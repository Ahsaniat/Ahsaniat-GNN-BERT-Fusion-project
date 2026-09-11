from __future__ import annotations
import ast, os
from collections import Counter
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.data import Batch
from torch.utils.data import Dataset


def load_musiccaps_metadata(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"ytid", "caption", "aspect_list"}
    missing = required.difference(df.columns)
    if missing: raise ValueError(f"MusicCaps metadata missing columns: {sorted(missing)}")
    return df


def build_tag_vocab(train_df: pd.DataFrame, top_k=50, min_count=5):
    c = Counter()
    for s in train_df["aspect_list"].dropna():
        try: tags = ast.literal_eval(str(s))
        except Exception: tags = []
        for t in tags:
            t = str(t).strip().lower()
            if len(t) > 2: c[t] += 1
    return [t for t,n in c.most_common() if n >= min_count][:top_k]


def tags_to_multihot(aspect_list, vocab):
    idx = {t:i for i,t in enumerate(vocab)}; y = torch.zeros(len(vocab), dtype=torch.float32)
    try: tags = ast.literal_eval(str(aspect_list))
    except Exception: tags = []
    for t in tags:
        t = str(t).strip().lower()
        if t in idx: y[idx[t]] = 1.0
    return y


class PairedMusicCapsDataset(Dataset):
    def __init__(self, df, graphs_by_id, vocab):
        rows = []
        for _, r in df.iterrows():
            ytid = str(r["ytid"])
            if ytid in graphs_by_id:
                rows.append((ytid, str(r["caption"]), tags_to_multihot(r["aspect_list"], vocab)))
        self.rows = rows; self.graphs = graphs_by_id
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        ytid, caption, y = self.rows[i]
        return self.graphs[ytid], caption, y, ytid


def collate_paired(items):
    graphs, captions, ys, ids = zip(*items)
    return Batch.from_data_list(list(graphs)), list(captions), torch.stack(ys), list(ids)
