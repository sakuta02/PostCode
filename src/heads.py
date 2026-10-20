from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttnPoolHead(nn.Module):
    def __init__(self, dim: int, n_bins: int, hidden: int = 256, dropout: float = 0.2):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.score = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, 1),
        )
        self.mlp = nn.Sequential(
            nn.Linear(2 * dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, n_bins),
        )

    def forward(self, grid):
        b, h, w, d = grid.shape
        tokens = self.norm(grid.reshape(b, h * w, d))
        attn = torch.softmax(self.score(tokens).squeeze(-1), dim=1)
        pooled = torch.einsum("bn,bnd->bd", attn, tokens)
        return self.mlp(torch.cat([pooled, tokens.mean(dim=1)], dim=-1))


class SegHead(nn.Module):
    def __init__(self, dim: int, n_classes: int = 4, hidden: int = 256,
                 dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, hidden), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden, n_classes),
        )

    def forward(self, grid):
        return self.net(grid).permute(0, 3, 1, 2)


class DistributionLoss(nn.Module):
    def __init__(self, centers: torch.Tensor, lambda_l1: float = 0.05):
        super().__init__()
        self.register_buffer("centers", centers.float())
        self.lambda_l1 = lambda_l1

    def forward(self, logits, target_dist, y):
        logp = F.log_softmax(logits, dim=-1)
        kl = F.kl_div(logp, target_dist, reduction="batchmean")
        pred = (logp.exp() * self.centers).sum(dim=-1)
        return kl + self.lambda_l1 * (pred - y).abs().mean()


def seg_loss(logits, target_a, target_b=None, lam: float = 1.0,
             weight=None, sample_mask=None, ignore_index: int = 255):
    mixed = target_b is not None and lam < 1.0
    per_pixel = F.cross_entropy(logits, target_a, weight=weight,
                                ignore_index=ignore_index, reduction="none")
    valid = (target_a != ignore_index).float()
    if mixed:
        other = F.cross_entropy(logits, target_b, weight=weight,
                                ignore_index=ignore_index, reduction="none")
        per_pixel = lam * per_pixel + (1.0 - lam) * other
        valid = valid * (target_b != ignore_index).float()
    if sample_mask is not None:
        valid = valid * sample_mask.view(-1, 1, 1).float()

    denom = valid.sum()
    if denom < 1:
        return logits.sum() * 0.0
    return (per_pixel * valid).sum() / denom
