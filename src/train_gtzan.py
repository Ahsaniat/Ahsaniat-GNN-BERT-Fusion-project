from __future__ import annotations
import argparse, copy, json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Batch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from src.models import GraphSAGEEncoder, GraphGenreModel
from src.utils import load_config, seed_everything, write_json
from src.scaling import GraphFeatureScaler

class GraphDs(Dataset):
    def __init__(self,df,label2idx,scaler=None): self.df=df.reset_index(drop=True); self.l=label2idx; self.scaler=scaler
    def __len__(self): return len(self.df)
    def __getitem__(self,i):
        r=self.df.iloc[i]; g=torch.load(f"data/processed/gtzan_graphs/{r.track_id}.pt",map_location='cpu',weights_only=False)
        if self.scaler is not None: g=self.scaler.transform_graph(g)
        return g,self.l[r.genre],r.track_id

def collate_graph(items):
    g,y,ids=zip(*items); return Batch.from_data_list(g),torch.tensor(y),list(ids)

class MelDs(Dataset):
    def __init__(self,df,label2idx): self.df=df.reset_index(drop=True); self.l=label2idx
    def __len__(self): return len(self.df)
    def __getitem__(self,i):
        r=self.df.iloc[i]; x=np.load(f"data/processed/gtzan_mels/{r.track_id}.npy")
        x=(x-x.mean())/(x.std()+1e-6); return torch.tensor(x[None],dtype=torch.float32),self.l[r.genre]

class SmallMelCNN(nn.Module):
    def __init__(self,nc):
        super().__init__(); self.f=nn.Sequential(
            nn.Conv2d(1,16,5,stride=2,padding=2),nn.BatchNorm2d(16),nn.ReLU(),nn.MaxPool2d(2),
            nn.Conv2d(16,32,3,padding=1),nn.BatchNorm2d(32),nn.ReLU(),nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1),nn.BatchNorm2d(64),nn.ReLU(),nn.AdaptiveAvgPool2d((1,1)))
        self.h=nn.Linear(64,nc)
    def forward(self,x): return self.h(self.f(x).flatten(1))

def split_df(df,seed):
    tr,rest=train_test_split(df,test_size=.30,random_state=seed,stratify=df.genre)
    va,te=train_test_split(rest,test_size=.50,random_state=seed,stratify=rest.genre)
    return tr,va,te

@torch.no_grad()
def eval_graph(model,loader,device):
    model.eval(); yy=[]; pp=[]
    for b,y,_ in loader:
        p=model(b.to(device))[0].argmax(1).cpu(); yy.extend(y.numpy()); pp.extend(p.numpy())
    return {'accuracy':float(accuracy_score(yy,pp)),'macro_f1':float(f1_score(yy,pp,average='macro'))}

@torch.no_grad()
def eval_cnn(model,loader,device):
    model.eval(); yy=[]; pp=[]
    for x,y in loader:
        p=model(x.to(device)).argmax(1).cpu(); yy.extend(y.numpy()); pp.extend(p.numpy())
    return {'accuracy':float(accuracy_score(yy,pp)),'macro_f1':float(f1_score(yy,pp,average='macro'))}

def train_model(model,tr,va,device,epochs,lr,graph=False):
    opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=1e-4); best=None; bestm=-1; bad=0
    for ep in range(epochs):
        model.train()
        for batch in tr:
            opt.zero_grad(set_to_none=True)
            if graph:
                b,y,_=batch; logits=model(b.to(device))[0]; y=y.to(device)
            else:
                x,y=batch; logits=model(x.to(device)); y=y.to(device)
            loss=F.cross_entropy(logits,y); loss.backward(); opt.step()
        m=(eval_graph if graph else eval_cnn)(model,va,device)['macro_f1']
        if m>bestm: bestm=m; best=copy.deepcopy(model.state_dict()); bad=0
        else: bad+=1
        if bad>=6: break
    model.load_state_dict(best); return model

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--config',default='config.yaml'); args=ap.parse_args(); cfg=load_config(args.config)
    df=pd.read_csv('data/processed/gtzan_index.csv'); genres=sorted(df.genre.unique()); l={g:i for i,g in enumerate(genres)}; device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results={'GNN':[],'CNN':[]}
    for seed in cfg['project']['seeds']:
        seed_everything(seed); trdf,vadf,tedf=split_df(df,seed)
        raw_train=[torch.load(f"data/processed/gtzan_graphs/{r.track_id}.pt",map_location='cpu',weights_only=False) for _,r in trdf.iterrows()]
        scaler=GraphFeatureScaler.fit(raw_train)
        gtr=DataLoader(GraphDs(trdf,l,scaler),batch_size=32,shuffle=True,collate_fn=collate_graph); gva=DataLoader(GraphDs(vadf,l,scaler),batch_size=64,collate_fn=collate_graph); gte=DataLoader(GraphDs(tedf,l,scaler),batch_size=64,collate_fn=collate_graph)
        in_dim=GraphDs(trdf,l,scaler)[0][0].x.shape[1]; ge=GraphSAGEEncoder(in_dim,cfg['model']['graph_hidden'],cfg['model']['graph_layers'],cfg['model']['dropout']); gm=GraphGenreModel(ge,len(genres)).to(device)
        gm=train_model(gm,gtr,gva,device,cfg['training']['epochs'],1e-3,graph=True); results['GNN'].append(eval_graph(gm,gte,device))
        ctr=DataLoader(MelDs(trdf,l),batch_size=16,shuffle=True); cva=DataLoader(MelDs(vadf,l),batch_size=32); cte=DataLoader(MelDs(tedf,l),batch_size=32)
        cm=train_model(SmallMelCNN(len(genres)).to(device),ctr,cva,device,cfg['training']['epochs'],1e-3,graph=False); results['CNN'].append(eval_cnn(cm,cte,device))
        torch.save({'state_dict':gm.state_dict(),'genres':genres,'seed':seed,'graph_scaler':scaler.state_dict()},f'checkpoints/gtzan_gnn_seed{seed}.pt')
    agg={m:{k:{'mean':float(np.mean([r[k] for r in rs])),'std':float(np.std([r[k] for r in rs]))} for k in rs[0]} for m,rs in results.items()}
    write_json({'seeds':results,'aggregate':agg},'results/gtzan_metrics.json'); print(json.dumps(agg,indent=2))
if __name__=='__main__': main()
