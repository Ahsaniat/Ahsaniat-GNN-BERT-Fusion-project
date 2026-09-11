from __future__ import annotations
import torch


class GraphFeatureScaler:
    """Per-feature z-score scaler fitted only on training graph nodes."""
    def __init__(self, mean: torch.Tensor, std: torch.Tensor):
        self.mean = mean.float()
        self.std = std.float().clamp_min(1e-5)

    @classmethod
    def fit(cls, graphs):
        xs = [g.x.float() for g in graphs]
        x = torch.cat(xs, dim=0)
        return cls(x.mean(dim=0), x.std(dim=0, unbiased=False))

    def transform_graph(self, graph):
        g = graph.clone()
        g.x = (g.x.float() - self.mean) / self.std
        return g

    def state_dict(self):
        return {'mean': self.mean, 'std': self.std}

    @classmethod
    def from_state_dict(cls, d):
        return cls(d['mean'], d['std'])
