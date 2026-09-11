from __future__ import annotations
import torch
import torch.nn.functional as F
from torch_geometric.data import Data


class MusicStructureGraphBuilder:
    """Temporal + top-k acoustic recurrence graph.

    Top-k edges avoid density swings caused by a single global cosine threshold.
    """
    def __init__(self, top_k=3, min_similarity=0.60, add_temporal=True):
        self.top_k = int(top_k)
        self.min_similarity = float(min_similarity)
        self.add_temporal = bool(add_temporal)

    def build(self, x: torch.Tensor) -> Data:
        n = int(x.shape[0])
        edges = set()
        if n == 1:
            edges.add((0, 0))
        if self.add_temporal:
            for i in range(n - 1):
                edges.add((i, i + 1)); edges.add((i + 1, i))
        if n > 2 and self.top_k > 0:
            xn = F.normalize(x, p=2, dim=-1)
            sim = xn @ xn.T
            sim.fill_diagonal_(-1.0)
            for i in range(n):
                k = min(self.top_k, n - 1)
                vals, idx = torch.topk(sim[i], k=k)
                for v, j in zip(vals.tolist(), idx.tolist()):
                    if v >= self.min_similarity:
                        edges.add((i, j)); edges.add((j, i))
        if not edges:
            edges.add((0, 0))
        edge_index = torch.tensor(sorted(edges), dtype=torch.long).T.contiguous()
        return Data(x=x, edge_index=edge_index)
