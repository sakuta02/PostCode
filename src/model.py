from __future__ import annotations

import torch.nn as nn

from backbones import DinoV2Backbone
from heads import AttnPoolHead, SegHead
from masks import N_CLASSES


class MultiTaskModel(nn.Module):
    def __init__(self, backbone_name: str, n_bins: int, hidden: int, dropout: float):
        super().__init__()
        self.backbone = DinoV2Backbone(backbone_name)
        self.head = AttnPoolHead(self.backbone.dim, n_bins, hidden, dropout)
        self.seg = SegHead(self.backbone.dim, N_CLASSES, hidden, dropout * 0.5)

    def forward(self, x):
        grid = self.backbone(x)
        return self.head(grid), self.seg(grid)
