from __future__ import annotations
import argparse, copy, json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from src.musiccaps_data import load_musiccaps_metadata, PairedMusicCapsDataset, collate_paired, build_tag_vocab
from src.train_musiccaps import load_graphs
from src.splits import split_ids
from src.text_encoder import BertTextEncoder
from src.models import GraphSAGEEncoder, ContrastiveDualEncoder
from src.metrics import retrieval_metrics
from src.utils import load_config, seed_everything, write_json

@torch.no_grad()
def encode_graph_text(graph_enc,text_enc,loader,device):
    ga=[]; tt=[]
    graph_enc.eval(); text_enc.eval()
    for b,caps,_,_ in loader:
        b=b.to(device); H,cls,mask=text_enc(caps); g=graph_enc(b.x,b.edge_index,b.batch); ga.append(g); tt.append(cls)
    return torch.cat(ga),torch.cat(tt)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--config',default='config.yaml'); args=ap.parse_args(); cfg=load_config(args.config); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); seed_everything(cfg['project']['seeds'][0])
    df=load_musiccaps_metadata(cfg['datasets']['musiccaps_metadata']); graphs=load_graphs('data/processed/musiccaps_graphs'); df=df[df.ytid.astype(str).isin(graphs)]
    split=split_ids(df.ytid.astype(str),seed=cfg['project']['seeds'][0]); trdf=df[df.ytid.astype(str).isin(split['train'])]; tedf=df[df.ytid.astype(str).isin(split['test'])]
    vocab=build_tag_vocab(trdf,top_k=5,min_count=1)  # labels unused, keeps dataset interface simple
    tr=DataLoader(PairedMusicCapsDataset(trdf,graphs,vocab),batch_size=32,shuffle=False,collate_fn=collate_paired); te=DataLoader(PairedMusicCapsDataset(tedf,graphs,vocab),batch_size=64,shuffle=False,collate_fn=collate_paired)
    text=BertTextEncoder(cfg['text']['model_name'],cfg['text']['max_length'],True,0).to(device)
    in_dim=next(iter(graphs.values())).x.shape[1]; graph=GraphSAGEEncoder(in_dim,cfg['model']['graph_hidden'],cfg['model']['graph_layers'],cfg['model']['dropout']).to(device)
    # If supervised MusicCaps graph weights exist, load compatible graph submodule.
    sup=Path('checkpoints/musiccaps_cross_attention_best.pt')
    if sup.exists():
        pack=torch.load(sup,map_location='cpu',weights_only=False); sd={k.replace('graph.',''):v for k,v in pack['state_dict'].items() if k.startswith('graph.')}; graph.load_state_dict(sd,strict=False)
    gtr,ttr=encode_graph_text(graph,text,tr,device); gte,tte=encode_graph_text(graph,text,te,device)
    model=ContrastiveDualEncoder(graph.out_dim,text.hidden_dim,embed=128).to(device); opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
    ds=torch.utils.data.TensorDataset(gtr.detach().cpu(),ttr.detach().cpu()); dl=DataLoader(ds,batch_size=64,shuffle=True)
    best=None; bestloss=1e9
    for ep in range(30):
        model.train(); tot=0
        for g,t in dl:
            g,t=g.to(device),t.to(device); opt.zero_grad(); za,zt=model(g,t); loss=model.loss(za,zt); loss.backward(); opt.step(); tot+=loss.item()
        if tot<bestloss: bestloss=tot; best=copy.deepcopy(model.state_dict())
    model.load_state_dict(best); model.eval()
    with torch.no_grad(): za,zt=model(gte.to(device),tte.to(device)); m=retrieval_metrics(za,zt)
    torch.save({'state_dict':model.state_dict(),'metrics':m},'checkpoints/musiccaps_contrastive.pt'); write_json(m,'results/musiccaps_retrieval.json'); print(json.dumps(m,indent=2))
if __name__=='__main__': main()
