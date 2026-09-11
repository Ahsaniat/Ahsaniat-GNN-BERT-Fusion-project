from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, global_mean_pool


class GraphSAGEEncoder(nn.Module):
    def __init__(self, in_dim, hidden=128, layers=3, dropout=0.2):
        super().__init__()
        dims = [in_dim] + [hidden] * layers
        self.convs = nn.ModuleList([SAGEConv(dims[i], dims[i+1], aggr="mean") for i in range(layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(layers)])
        self.dropout = float(dropout)
        self.out_dim = hidden

    def forward(self, x, edge_index, batch):
        h = x
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            new = norm(conv(h, edge_index))
            new = F.gelu(new)
            if h.shape[-1] == new.shape[-1]:
                new = new + h
            h = F.dropout(new, p=self.dropout, training=self.training)
        return global_mean_pool(h, batch)


class EarlyConcatFusion(nn.Module):
    def __init__(self, dg=128, dt=768, out=128, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dg + dt, out), nn.LayerNorm(out), nn.GELU(), nn.Dropout(dropout))
        self.out_dim = out
    def forward(self, g, H, cls, mask):
        return self.net(torch.cat([g, cls], dim=-1))


class CrossAttentionFusion(nn.Module):
    def __init__(self, dg=128, dt=768, d=128, heads=4, dropout=0.2):
        super().__init__()
        self.q = nn.Linear(dg, d)
        self.k = nn.Linear(dt, d)
        self.v = nn.Linear(dt, d)
        self.attn = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.out = nn.Sequential(nn.Linear(dg + d, d), nn.LayerNorm(d), nn.GELU(), nn.Dropout(dropout))
        self.out_dim = d
    def forward(self, g, H, cls, mask):
        q = self.q(g).unsqueeze(1)
        k, v = self.k(H), self.v(H)
        key_padding_mask = ~mask.bool() if mask is not None else None
        ctx, _ = self.attn(q, k, v, key_padding_mask=key_padding_mask, need_weights=False)
        return self.out(torch.cat([g, ctx.squeeze(1)], dim=-1))


class GatedFusion(nn.Module):
    def __init__(self, dg=128, dt=768, d=128, dropout=0.2):
        super().__init__()
        self.ga = nn.Linear(dg, d)
        self.gt = nn.Linear(dt, d)
        self.gate = nn.Linear(dg + dt, d)
        self.norm = nn.LayerNorm(d)
        self.drop = nn.Dropout(dropout)
        self.out_dim = d
    def forward(self, g, H, cls, mask):
        a = torch.tanh(self.ga(g)); t = torch.tanh(self.gt(cls))
        gate = torch.sigmoid(self.gate(torch.cat([g, cls], dim=-1)))
        return self.drop(self.norm(gate * a + (1.0 - gate) * t))


class MultilabelHead(nn.Module):
    def __init__(self, in_dim, num_tags, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_dim, in_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(in_dim, num_tags))
    def forward(self, z): return self.net(z)


class EmotionHead(nn.Module):
    def __init__(self, in_dim, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_dim, 64), nn.GELU(), nn.Linear(64, 2))
    def forward(self, z):
        return 1.0 + 8.0 * torch.sigmoid(self.net(z))


class MultimodalTagModel(nn.Module):
    def __init__(self, graph_encoder, fusion, num_tags):
        super().__init__()
        self.graph = graph_encoder
        self.fusion = fusion
        self.head = MultilabelHead(fusion.out_dim, num_tags)
    def forward(self, graph_batch, H, cls, mask):
        g = self.graph(graph_batch.x, graph_batch.edge_index, graph_batch.batch)
        z = self.fusion(g, H, cls, mask)
        return self.head(z), z, g


class GraphGenreModel(nn.Module):
    def __init__(self, graph_encoder, num_classes):
        super().__init__(); self.graph = graph_encoder; self.head = nn.Linear(graph_encoder.out_dim, num_classes)
    def forward(self, batch):
        g = self.graph(batch.x, batch.edge_index, batch.batch)
        return self.head(g), g


class GraphTagModel(nn.Module):
    def __init__(self, graph_encoder, num_tags, dropout=0.2):
        super().__init__(); self.graph = graph_encoder; self.head = MultilabelHead(graph_encoder.out_dim, num_tags, dropout)
    def forward(self, batch):
        g = self.graph(batch.x, batch.edge_index, batch.batch)
        return self.head(g), g


class TextOnlyTagModel(nn.Module):
    def __init__(self, text_dim, num_tags, hidden=128, dropout=0.2):
        super().__init__(); self.proj = nn.Sequential(nn.Linear(text_dim, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout)); self.head = nn.Linear(hidden, num_tags)
    def forward(self, cls): return self.head(self.proj(cls))


class ContrastiveDualEncoder(nn.Module):
    def __init__(self, graph_dim=128, text_dim=768, embed=128, temperature=0.07):
        super().__init__()
        self.audio = nn.Sequential(nn.Linear(graph_dim, embed), nn.GELU(), nn.Linear(embed, embed))
        self.text = nn.Sequential(nn.Linear(text_dim, embed), nn.GELU(), nn.Linear(embed, embed))
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / temperature)))
    def forward(self, g, t):
        za = F.normalize(self.audio(g), dim=-1); zt = F.normalize(self.text(t), dim=-1)
        return za, zt
    def loss(self, za, zt):
        scale = self.logit_scale.exp().clamp(max=100)
        logits = scale * (za @ zt.T); y = torch.arange(logits.shape[0], device=logits.device)
        return 0.5 * (F.cross_entropy(logits, y) + F.cross_entropy(logits.T, y))
