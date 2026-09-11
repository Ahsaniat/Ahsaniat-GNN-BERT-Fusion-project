from __future__ import annotations
import argparse, os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import librosa
from tqdm import tqdm
from src.audio_features import AudioFeatureExtractor
from src.graph_builder import MusicStructureGraphBuilder
from src.utils import load_config, ensure_dirs


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--config',default='config.yaml'); args=ap.parse_args()
    cfg=load_config(args.config); a=cfg['audio']; g=cfg['graph']; root=Path(cfg['datasets']['gtzan_audio_dir'])
    graph_dir=Path('data/processed/gtzan_graphs'); mel_dir=Path('data/processed/gtzan_mels'); ensure_dirs(str(graph_dir),str(mel_dir),'results')
    ex=AudioFeatureExtractor(sr=a['sr'],n_mels=a['n_mels'],n_mfcc=a['n_mfcc']); gb=MusicStructureGraphBuilder(g['similarity_top_k'],g['similarity_min'],g['add_temporal_edges'])
    rows=[]
    files=sorted([p for p in root.glob('*/*') if p.suffix.lower() in {'.wav','.au','.mp3','.flac'}])
    for p in tqdm(files):
        try:
            y,_=librosa.load(str(p),sr=a['sr'],mono=True)
            x=ex.extract_segments(y,a['window_seconds'],a['hop_seconds']); graph=gb.build(x)
            track_id=f'{p.parent.name}__{p.stem}'; torch.save(graph,graph_dir/f'{track_id}.pt')
            mel=librosa.feature.melspectrogram(y=y,sr=a['sr'],n_mels=128,hop_length=512,power=2.0)
            mel=librosa.power_to_db(mel+1e-10,ref=np.max).astype(np.float32)
            # Fixed-width 30 s crop/pad for CNN baseline.
            target=1292
            if mel.shape[1]<target: mel=np.pad(mel,((0,0),(0,target-mel.shape[1])),constant_values=mel.min())
            else: mel=mel[:,:target]
            np.save(mel_dir/f'{track_id}.npy',mel)
            rows.append({'track_id':track_id,'genre':p.parent.name,'audio_path':str(p)})
        except Exception as e:
            print('skip',p,e)
    pd.DataFrame(rows).to_csv('data/processed/gtzan_index.csv',index=False)
    print('processed',len(rows),'GTZAN tracks')
if __name__=='__main__': main()
