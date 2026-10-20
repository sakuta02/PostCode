from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PATCH = 14


@dataclass
class Config:
    train_dir: str = "data/train"
    out_dir: str = "outputs"
    coco_json: str = "annotations/masks.json"
    run_tag: str = "mt"
    num_workers: int = 2

    backbone: str = "dinov2_vitb14"
    img_size: int = 392
    batch_size: int = 8
    grad_accum: int = 1
    unfreeze_blocks: int = 4
    hidden: int = 256
    dropout: float = 0.2

    bin_step: float = 5.0
    label_sigma: float = 3.0
    lambda_l1: float = 0.05
    lambda_seg: float = 0.5
    lambda_seg_final: float = 0.1
    mask_scale: int = 4
    seg_grid_mult: int = 4

    epochs: int = 30
    backbone_lr: float = 2e-5
    head_lr: float = 1e-3
    llrd: float = 0.75
    weight_decay: float = 1e-4
    aug_strength: float = 1.0
    mixup_alpha: float = 0.4
    ema_decay: float = 0.99

    @property
    def images_dir(self) -> Path:
        return Path(self.train_dir) / "images"

    @property
    def labels_csv(self) -> Path:
        return Path(self.train_dir) / "train.csv"

    @property
    def ckpt_dir(self) -> Path:
        path = Path(self.out_dir) / "checkpoints"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def grid_side(self) -> int:
        return self.img_size // PATCH

    def apply_overrides(self, pairs) -> "Config":
        for pair in pairs or []:
            key, _, raw = pair.partition("=")
            key = key.strip()
            if not hasattr(self, key):
                raise KeyError(f"нет такого параметра: {key}")
            cur = getattr(self, key)
            if isinstance(cur, int):
                val = int(raw)
            elif isinstance(cur, float):
                val = float(raw)
            else:
                val = raw
            setattr(self, key, val)
        return self
