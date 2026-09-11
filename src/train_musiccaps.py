from __future__ import annotations
import argparse, copy, json, os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from src.musiccaps_data import load_musiccaps_metadata, build_tag_vocab, PairedMusicCapsDataset, collate_paired
from src.splits import split_ids
from src.text_encoder import BertTextEncoder
from src.models import GraphSAGEEncoder, EarlyConcatFusion, CrossAttentionFusion, GatedFusion, MultimodalTagModel, TextOnlyTagModel, GraphTagModel
from src.metrics import calibrate_thresholds, multilabel_metrics, retrieval_metrics
from src.utils import load_config, seed_everything, ensure_dirs, write_json
from src.scaling import GraphFeatureScaler


def load_graphs(graph_dir):
    out={}
    for p in Path(graph_dir).glob('*.pt'):
        try: out[p.stem]=torch.load(p, map_location='cpu', weights_only=False)
        except Exception as e: print('skip',p,e)
    return out

@torch.no_grad()
def encode_texts(encoder, captions, device):
    H, cls, mask=encoder(captions); return H, cls, mask

def pos_weight_from_dataset(ds):
    ys=torch.stack([r[2] for r in ds.rows]); pos=ys.sum(0); neg=len(ds)-pos
    return (neg/(pos+1e-6)).clamp(1.0,20.0)

def eval_model(model, text_encoder, loader, device):
    model.eval(); text_encoder.eval(); probs=[]; ys=[]
    with torch.no_grad():
        for graph,caps,y,_ in loader:
            graph=graph.to(device); y=y.to(device); H,cls,mask=text_encoder(caps)
            logits,_,_=model(graph,H,cls,mask)
            probs.append(torch.sigmoid(logits).cpu()); ys.append(y.cpu())
    return torch.cat(ys).numpy(), torch.cat(probs).numpy()

def train_one(kind, cfg, train_ds, val_ds, test_ds, text_encoder, device, seed):
    seed_everything(seed)
    sample_graph=train_ds[0][0]; in_dim=sample_graph.x.shape[1]
    mcfg=cfg['model']; tcfg=cfg['training']; n_tags=len(train_ds[0][2])
    graph=GraphSAGEEncoder(in_dim,mcfg['graph_hidden'],mcfg['graph_layers'],mcfg['dropout'])
    if kind=='early': fusion=EarlyConcatFusion(mcfg['graph_hidden'],text_encoder.hidden_dim,mcfg['fusion_dim'],mcfg['dropout'])
    elif kind=='cross_attention': fusion=CrossAttentionFusion(mcfg['graph_hidden'],text_encoder.hidden_dim,mcfg['fusion_dim'],mcfg['attention_heads'],mcfg['dropout'])
    elif kind=='gated': fusion=GatedFusion(mcfg['graph_hidden'],text_encoder.hidden_dim,mcfg['fusion_dim'],mcfg['dropout'])
    else: raise ValueError(kind)
    model=MultimodalTagModel(graph,fusion,n_tags).to(device)
    text_trainable=[p for p in text_encoder.parameters() if p.requires_grad]
    params=list(model.parameters())+text_trainable
    groups=[{'params': model.parameters(), 'lr': tcfg['lr_heads']}]
    if text_trainable:
        groups.append({'params': text_trainable, 'lr': tcfg['lr_backbone']})
    opt=torch.optim.AdamW(groups,weight_decay=tcfg['weight_decay'])
    pw=pos_weight_from_dataset(train_ds).to(device) if tcfg['use_pos_weight'] else None
    loss_fn=torch.nn.BCEWithLogitsLoss(pos_weight=pw)
    tr=DataLoader(train_ds,batch_size=tcfg['batch_size'],shuffle=True,collate_fn=collate_paired)
    va=DataLoader(val_ds,batch_size=tcfg['batch_size']*2,shuffle=False,collate_fn=collate_paired)
    te=DataLoader(test_ds,batch_size=tcfg['batch_size']*2,shuffle=False,collate_fn=collate_paired)
    best=None; best_metric=-1; bad=0
    for epoch in range(1,tcfg['epochs']+1):
        model.train();
        if text_trainable: text_encoder.train()
        else: text_encoder.eval()
        total=0
        for graph_b,caps,y,_ in tr:
            graph_b=graph_b.to(device); y=y.to(device); H,cls,mask=text_encoder(caps)
            opt.zero_grad(set_to_none=True); logits,_,_=model(graph_b,H,cls,mask); loss=loss_fn(logits,y); loss.backward()
            torch.nn.utils.clip_grad_norm_(params,tcfg['grad_clip']); opt.step(); total+=loss.item()
        yv,pv=eval_model(model,text_encoder,va,device)
        th=calibrate_thresholds(yv,pv,tcfg['threshold_min'],tcfg['threshold_max'],tcfg['threshold_step'])
        vm=multilabel_metrics(yv,pv,th)
        if vm['macro_f1']>best_metric:
            best_metric=vm['macro_f1']; bad=0
            best={'model':copy.deepcopy(model.state_dict()),'thresholds':th.tolist(),'epoch':epoch}
        else:
            bad+=1
            if bad>=tcfg['patience']: break
    model.load_state_dict(best['model']); yt,pt=eval_model(model,text_encoder,te,device)
    tm=multilabel_metrics(yt,pt,np.array(best['thresholds']))
    return model,best,tm


@torch.no_grad()
def eval_text_only(model, text_encoder, loader, device):
    model.eval(); text_encoder.eval(); probs=[]; ys=[]
    for _,caps,y,_ in loader:
        y=y.to(device); _,cls,_=text_encoder(caps); probs.append(torch.sigmoid(model(cls)).cpu()); ys.append(y.cpu())
    return torch.cat(ys).numpy(), torch.cat(probs).numpy()


def train_text_only(cfg, train_ds, val_ds, test_ds, text_encoder, device, seed):
    seed_everything(seed); tcfg=cfg['training']; n_tags=len(train_ds[0][2])
    model=TextOnlyTagModel(text_encoder.hidden_dim,n_tags,hidden=cfg['model']['fusion_dim'],dropout=cfg['model']['dropout']).to(device)
    trainable=[p for p in text_encoder.parameters() if p.requires_grad]
    groups=[{'params':model.parameters(),'lr':tcfg['lr_heads']}]
    if trainable: groups.append({'params':trainable,'lr':tcfg['lr_backbone']})
    opt=torch.optim.AdamW(groups,weight_decay=tcfg['weight_decay'])
    loss_fn=torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight_from_dataset(train_ds).to(device) if tcfg['use_pos_weight'] else None)
    tr=DataLoader(train_ds,batch_size=tcfg['batch_size'],shuffle=True,collate_fn=collate_paired); va=DataLoader(val_ds,batch_size=tcfg['batch_size']*2,collate_fn=collate_paired); te=DataLoader(test_ds,batch_size=tcfg['batch_size']*2,collate_fn=collate_paired)
    best=None; bestm=-1; bad=0
    for epoch in range(tcfg['epochs']):
        model.train(); text_encoder.train() if trainable else text_encoder.eval()
        for _,caps,y,_ in tr:
            y=y.to(device); _,cls,_=text_encoder(caps); opt.zero_grad(set_to_none=True); logits=model(cls); loss=loss_fn(logits,y); loss.backward(); opt.step()
        yv,pv=eval_text_only(model,text_encoder,va,device); th=calibrate_thresholds(yv,pv,tcfg['threshold_min'],tcfg['threshold_max'],tcfg['threshold_step']); m=multilabel_metrics(yv,pv,th)['macro_f1']
        if m>bestm: bestm=m; best=(copy.deepcopy(model.state_dict()),th.tolist()); bad=0
        else: bad+=1
        if bad>=tcfg['patience']: break
    model.load_state_dict(best[0]); yt,pt=eval_text_only(model,text_encoder,te,device); return model,best,multilabel_metrics(yt,pt,np.array(best[1]))

@torch.no_grad()
def eval_graph_only(model, loader, device):
    model.eval(); probs=[]; ys=[]
    for graph,_,y,_ in loader:
        y=y.to(device); logits,_=model(graph.to(device)); probs.append(torch.sigmoid(logits).cpu()); ys.append(y.cpu())
    return torch.cat(ys).numpy(),torch.cat(probs).numpy()


def train_graph_only(cfg, train_ds, val_ds, test_ds, device, seed):
    seed_everything(seed); tcfg=cfg['training']; n_tags=len(train_ds[0][2]); in_dim=train_ds[0][0].x.shape[1]
    enc=GraphSAGEEncoder(in_dim,cfg['model']['graph_hidden'],cfg['model']['graph_layers'],cfg['model']['dropout']); model=GraphTagModel(enc,n_tags,cfg['model']['dropout']).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=tcfg['lr_heads'],weight_decay=tcfg['weight_decay']); loss_fn=torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight_from_dataset(train_ds).to(device) if tcfg['use_pos_weight'] else None)
    tr=DataLoader(train_ds,batch_size=tcfg['batch_size'],shuffle=True,collate_fn=collate_paired); va=DataLoader(val_ds,batch_size=tcfg['batch_size']*2,collate_fn=collate_paired); te=DataLoader(test_ds,batch_size=tcfg['batch_size']*2,collate_fn=collate_paired)
    best=None; bestm=-1; bad=0
    for epoch in range(tcfg['epochs']):
        model.train()
        for graph,_,y,_ in tr:
            graph=graph.to(device); y=y.to(device); opt.zero_grad(set_to_none=True); logits,_=model(graph); loss=loss_fn(logits,y); loss.backward(); opt.step()
        yv,pv=eval_graph_only(model,va,device); th=calibrate_thresholds(yv,pv,tcfg['threshold_min'],tcfg['threshold_max'],tcfg['threshold_step']); m=multilabel_metrics(yv,pv,th)['macro_f1']
        if m>bestm: bestm=m; best=(copy.deepcopy(model.state_dict()),th.tolist()); bad=0
        else: bad+=1
        if bad>=tcfg['patience']: break
    model.load_state_dict(best[0]); yt,pt=eval_graph_only(model,te,device); return model,best,multilabel_metrics(yt,pt,np.array(best[1]))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--config',default='config.yaml'); ap.add_argument('--top-k-tags',type=int,default=35); args=ap.parse_args()
    cfg=load_config(args.config); ensure_dirs(cfg['project']['output_dir'],cfg['project']['checkpoint_dir'],'data/splits')
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); print('device',device)
    df=load_musiccaps_metadata(cfg['datasets']['musiccaps_metadata']); graphs=load_graphs('data/processed/musiccaps_graphs')
    df=df[df['ytid'].astype(str).isin(graphs.keys())].copy(); split=split_ids(df['ytid'].astype(str),seed=cfg['project']['seeds'][0])
    write_json(split,'data/splits/musiccaps.json')
    scaler=GraphFeatureScaler.fit([graphs[i] for i in split['train'] if i in graphs])
    graphs={k:scaler.transform_graph(v) for k,v in graphs.items()}
    torch.save(scaler.state_dict(),'results/musiccaps_graph_scaler.pt')
    train_df=df[df.ytid.astype(str).isin(split['train'])]; val_df=df[df.ytid.astype(str).isin(split['val'])]; test_df=df[df.ytid.astype(str).isin(split['test'])]
    vocab=build_tag_vocab(train_df,top_k=args.top_k_tags,min_count=max(3,int(len(train_df)*0.005))); write_json(vocab,'results/tag_vocab.json')
    train_ds=PairedMusicCapsDataset(train_df,graphs,vocab); val_ds=PairedMusicCapsDataset(val_df,graphs,vocab); test_ds=PairedMusicCapsDataset(test_df,graphs,vocab)
    print('paired split sizes',len(train_ds),len(val_ds),len(test_ds),'tags',len(vocab))
    text=BertTextEncoder(cfg['text']['model_name'],cfg['text']['max_length'],cfg['text']['freeze_backbone'],cfg['text']['unfreeze_last_n_layers']).to(device)
    all_results={}
    # Independently trained single-modality baselines (not modality masking).
    for baseline in ('bert_only','gnn_only'):
        seed_metrics=[]; best_pack=None; best_score=-1
        for seed in cfg['project']['seeds']:
            if baseline=='bert_only': model,best,metrics=train_text_only(cfg,train_ds,val_ds,test_ds,text,device,seed)
            else: model,best,metrics=train_graph_only(cfg,train_ds,val_ds,test_ds,device,seed)
            seed_metrics.append(metrics)
            if metrics['macro_f1']>best_score: best_score=metrics['macro_f1']; best_pack=(copy.deepcopy(model.state_dict()),best)
        agg={k:{'mean':float(np.mean([m[k] for m in seed_metrics])),'std':float(np.std([m[k] for m in seed_metrics]))} for k in seed_metrics[0]}
        all_results[baseline]={'seeds':seed_metrics,'aggregate':agg}
        torch.save({'state_dict':best_pack[0],'thresholds':best_pack[1][1],'vocab':vocab,'config':cfg,'kind':baseline,'graph_scaler':scaler.state_dict()},Path(cfg['project']['checkpoint_dir'])/f'musiccaps_{baseline}_best.pt')
        print(baseline,agg)
    for kind in ('early','cross_attention','gated'):
        seed_metrics=[]; best_pack=None; best_score=-1
        for seed in cfg['project']['seeds']:
            # Fresh text encoder per seed if any BERT layer is trainable; otherwise shared frozen backbone is safe.
            if any(p.requires_grad for p in text.backbone.parameters()):
                text=BertTextEncoder(cfg['text']['model_name'],cfg['text']['max_length'],cfg['text']['freeze_backbone'],cfg['text']['unfreeze_last_n_layers']).to(device)
            model,best,metrics=train_one(kind,cfg,train_ds,val_ds,test_ds,text,device,seed); seed_metrics.append(metrics)
            if metrics['macro_f1']>best_score:
                best_score=metrics['macro_f1']; best_pack=(model.state_dict(),best)
        agg={k:{'mean':float(np.mean([m[k] for m in seed_metrics])),'std':float(np.std([m[k] for m in seed_metrics]))} for k in seed_metrics[0]}
        all_results[kind]={'seeds':seed_metrics,'aggregate':agg}
        torch.save({'state_dict':best_pack[0],'thresholds':best_pack[1]['thresholds'],'vocab':vocab,'config':cfg,'kind':kind,'graph_scaler':scaler.state_dict()},Path(cfg['project']['checkpoint_dir'])/f'musiccaps_{kind}_best.pt')
        print(kind,agg)
    write_json(all_results,'results/musiccaps_multiseed_metrics.json')
if __name__=='__main__': main()
