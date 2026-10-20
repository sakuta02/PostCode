from __future__ import annotations

import os

import torch
import torch.nn as nn


class DinoV2Backbone(nn.Module):
    def __init__(self, name: str = "dinov2_vitb14"):
        super().__init__()
        repo = os.getenv("LOADEST_DINOV2_REPO")
        pretrained = os.getenv("LOADEST_SKIP_PRETRAINED") != "1"
        src, kw = (repo, {"source": "local"}) if repo else \
            ("facebookresearch/dinov2", {"trust_repo": True})
        self.net = torch.hub.load(src, name, pretrained=pretrained, **kw)
        self.dim = self.net.embed_dim

    def forward(self, x):
        tokens = self.net.forward_features(x)["x_norm_patchtokens"]
        b, n, d = tokens.shape
        side = int(round(n ** 0.5))
        return tokens.reshape(b, side, side, d)

    def block_groups(self):
        return [[self.net.patch_embed], *([b] for b in self.net.blocks), [self.net.norm]]


def freeze_all_but_last(backbone: nn.Module, n_unfrozen: int) -> None:
    for p in backbone.parameters():
        p.requires_grad = False
    if n_unfrozen <= 0:
        return
    for group in backbone.block_groups()[-n_unfrozen:]:
        for module in group:
            for p in module.parameters():
                p.requires_grad = True


def param_groups(backbone: nn.Module, head: nn.Module, backbone_lr: float,
                 head_lr: float, llrd: float, weight_decay: float):
    groups = backbone.block_groups()
    n = len(groups)
    out = []
    for depth, group in enumerate(groups):
        params = [p for m in group for p in m.parameters() if p.requires_grad]
        if params:
            out.append({"params": params, "lr": backbone_lr * llrd ** (n - 1 - depth),
                        "weight_decay": weight_decay})
    out.append({"params": list(head.parameters()), "lr": head_lr,
                "weight_decay": weight_decay})
    return out
