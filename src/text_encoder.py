from __future__ import annotations
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


class BertTextEncoder(nn.Module):
    """DistilBERT token encoder returning token embeddings and an attention mask.

    The original notebook dropped the attention mask before cross-attention, so
    PAD tokens could receive fusion attention. This implementation preserves it.
    """
    def __init__(self, model_name="distilbert-base-uncased", max_length=96,
                 freeze_backbone=True, unfreeze_last_n_layers=0):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(model_name)
        self.max_length = int(max_length)
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
        if unfreeze_last_n_layers > 0 and hasattr(self.backbone, "transformer"):
            for layer in self.backbone.transformer.layer[-unfreeze_last_n_layers:]:
                for p in layer.parameters():
                    p.requires_grad = True
        self.hidden_dim = int(self.backbone.config.hidden_size)

    def tokenize(self, texts, device=None):
        batch = self.tokenizer(
            list(texts), padding=True, truncation=True,
            max_length=self.max_length, return_tensors="pt"
        )
        if device is not None:
            batch = {k: v.to(device) for k, v in batch.items()}
        return batch

    def forward(self, texts):
        device = next(self.backbone.parameters()).device
        batch = self.tokenize(texts, device=device)
        out = self.backbone(**batch)
        H = out.last_hidden_state
        cls = H[:, 0]
        return H, cls, batch["attention_mask"].bool()
