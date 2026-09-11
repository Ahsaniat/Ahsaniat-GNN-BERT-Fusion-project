from __future__ import annotations
import numpy as np
from sklearn.metrics import f1_score, average_precision_score


def calibrate_thresholds(y_true, probs, lo=0.10, hi=0.90, step=0.05):
    y_true = np.asarray(y_true); probs = np.asarray(probs)
    grid = np.arange(lo, hi + 1e-9, step)
    th = np.full(y_true.shape[1], 0.5, dtype=np.float32)
    for k in range(y_true.shape[1]):
        if y_true[:, k].sum() == 0:
            continue
        scores = [f1_score(y_true[:, k], probs[:, k] >= t, zero_division=0) for t in grid]
        th[k] = float(grid[int(np.argmax(scores))])
    return th


def multilabel_metrics(y_true, probs, thresholds=None):
    y_true = np.asarray(y_true); probs = np.asarray(probs)
    if thresholds is None: thresholds = np.full(y_true.shape[1], 0.5)
    pred = probs >= np.asarray(thresholds)[None, :]
    out = {
        "macro_f1": float(f1_score(y_true, pred, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(y_true, pred, average="micro", zero_division=0)),
    }
    valid = y_true.sum(axis=0) > 0
    if valid.any():
        out["macro_auprc"] = float(average_precision_score(y_true[:, valid], probs[:, valid], average="macro"))
    else:
        out["macro_auprc"] = float("nan")
    return out


def retrieval_metrics(za, zt, ks=(1,5,10)):
    import torch
    sim = zt @ za.T; n = sim.shape[0]
    ranks = sim.argsort(dim=1, descending=True)
    target = torch.arange(n, device=sim.device)[:, None]
    out = {}
    for k in ks:
        if k <= n:
            out[f"caption_to_audio_R@{k}"] = float((ranks[:, :k] == target).any(dim=1).float().mean())
    ranks2 = sim.T.argsort(dim=1, descending=True)
    for k in ks:
        if k <= n:
            out[f"audio_to_caption_R@{k}"] = float((ranks2[:, :k] == target).any(dim=1).float().mean())
    return out
