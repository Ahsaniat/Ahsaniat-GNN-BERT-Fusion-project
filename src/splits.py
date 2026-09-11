from __future__ import annotations
import json
import numpy as np
from sklearn.model_selection import train_test_split


def split_ids(ids, seed=42, train=0.70, val=0.15):
    ids = np.array(sorted(set(map(str, ids))))
    tr, rest = train_test_split(ids, train_size=train, random_state=seed, shuffle=True)
    val_frac_rest = val / (1.0 - train)
    va, te = train_test_split(rest, train_size=val_frac_rest, random_state=seed, shuffle=True)
    return {"train": tr.tolist(), "val": va.tolist(), "test": te.tolist()}


def save_split(split, path):
    with open(path, "w") as f: json.dump(split, f, indent=2)
