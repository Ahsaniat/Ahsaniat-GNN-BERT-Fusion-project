from __future__ import annotations
import argparse, json, os
from pathlib import Path
import torch
from tqdm import tqdm
from src.audio_features import AudioFeatureExtractor
from src.graph_builder import MusicStructureGraphBuilder
from src.musiccaps_data import load_musiccaps_metadata
from src.utils import load_config, ensure_dirs


def find_audio(audio_dir, ytid):
    for ext in (".wav", ".mp3", ".flac", ".m4a", ".ogg"):
        p = Path(audio_dir) / f"{ytid}{ext}"
        if p.exists(): return str(p)
    return None


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--config',default='config.yaml'); args=ap.parse_args()
    cfg=load_config(args.config); a=cfg['audio']; g=cfg['graph']; d=cfg['datasets']
    df=load_musiccaps_metadata(d['musiccaps_metadata'])
    out=Path('data/processed/musiccaps_graphs'); ensure_dirs(str(out))
    ex=AudioFeatureExtractor(sr=a['sr'], n_mels=a['n_mels'], n_mfcc=a['n_mfcc'])
    gb=MusicStructureGraphBuilder(g['similarity_top_k'], g['similarity_min'], g['add_temporal_edges'])
    done=0; missing=[]
    for _,r in tqdm(df.iterrows(), total=len(df)):
        ytid=str(r['ytid']); p=find_audio(d['musiccaps_audio_dir'],ytid)
        if not p: missing.append(ytid); continue
        y=ex.load_clip(p, duration=a['clip_seconds'])
        x=ex.extract_segments(y,a['window_seconds'],a['hop_seconds'])
        graph=gb.build(x); graph.ytid=ytid
        torch.save(graph,out/f'{ytid}.pt'); done+=1
    Path('results').mkdir(exist_ok=True)
    json.dump({'graphs_created':done,'audio_missing':len(missing),'missing_ids':missing[:100]},open('results/preprocess_musiccaps.json','w'),indent=2)
    print(f'Created {done} paired MusicCaps graphs; missing audio for {len(missing)} rows.')
if __name__=='__main__': main()
