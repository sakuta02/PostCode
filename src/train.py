from __future__ import annotations

import argparse
import copy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from backbones import freeze_all_but_last, param_groups
from config import Config
from dataset import load_table
from heads import DistributionLoss, seg_loss
from joint_augment import JointTransform
from masks import IGNORE_INDEX, N_CLASSES, class_weights, load_polygons, rasterize
from model import MultiTaskModel
from target import bin_centers, soft_labels

DEFAULT_EPOCH = 17


class MaskDataset(Dataset):
    def __init__(self, df, polys, transform, dists, ys, mask_scale: int):
        self.paths = df["path"].tolist()
        self.ids = df["image_id"].astype(str).tolist()
        self.polys = polys
        self.transform = transform
        self.dists = dists
        self.ys = ys
        self.mask_scale = mask_scale

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        polys = self.polys.get(self.ids[i])
        with Image.open(self.paths[i]) as im:
            img = im.convert("RGB")
            mask = rasterize(polys, img.size, self.mask_scale) if polys else None
            x, grid = self.transform(img, mask)
        return (x, grid, torch.tensor(float(polys is not None)),
                torch.from_numpy(self.dists[i]), torch.tensor(self.ys[i], dtype=torch.float32))


class EMA:
    def __init__(self, model, decay=0.99):
        self.decay = decay
        self.step = 0
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        self.step += 1
        d = min(self.decay, (1 + self.step) / (10 + self.step))
        for s, p in zip(self.shadow.state_dict().values(), model.state_dict().values()):
            if s.dtype.is_floating_point:
                s.mul_(d).add_(p.detach(), alpha=1 - d)
            else:
                s.copy_(p)


def seg_lambda(cfg: Config, epoch: int) -> float:
    if cfg.epochs <= 1:
        return cfg.lambda_seg
    t = epoch / (cfg.epochs - 1)
    return cfg.lambda_seg + (cfg.lambda_seg_final - cfg.lambda_seg) * t


def load_masks(cfg: Config) -> dict:
    if not Path(cfg.coco_json).exists():
        print("разметки масок нет — сегментация в лосс не войдёт")
        return {}
    polys = load_polygons(cfg.coco_json)
    return {k: v for k, v in polys.items() if set(v) & {"cargo", "floor"}}


def seg_class_weights(cfg: Config, df, polys, limit: int = 400) -> np.ndarray:
    tf = JointTransform(cfg.img_size, cfg.grid_side, train=False, grid_mult=cfg.seg_grid_mult)
    paths = dict(zip(df["image_id"].astype(str), df["path"]))
    grids = []
    for image_id in [k for k in polys if k in paths][:limit]:
        with Image.open(paths[image_id]) as im:
            size = im.size
        mask = rasterize(polys[image_id], size, cfg.mask_scale)
        grids.append(tf.geometry(Image.new("RGB", size), mask)[1])
    return class_weights(grids) if grids else np.ones(N_CLASSES)


def train(cfg: Config, df, epoch: int = DEFAULT_EPOCH) -> Path:
    if epoch > cfg.epochs:
        raise ValueError(f"эпоха сохранения {epoch} больше epochs {cfg.epochs}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp = device == "cuda"
    centers = bin_centers(cfg.bin_step)
    y = df["load_pct"].values.astype(float)

    polys = load_masks(cfg)
    n_masked = int(df["image_id"].astype(str).isin(polys).sum())
    print(f"обучение на всех данных: {len(df)} фото (с маской {n_masked}), "
          f"эпох {epoch} из расписания на {cfg.epochs}")

    tf = JointTransform(cfg.img_size, cfg.grid_side, train=True,
                        strength=cfg.aug_strength, grid_mult=cfg.seg_grid_mult)
    ds = MaskDataset(df, polys, tf, soft_labels(y, centers, cfg.label_sigma), y, cfg.mask_scale)
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True,
                    num_workers=cfg.num_workers, pin_memory=amp)

    model = MultiTaskModel(cfg.backbone, len(centers), cfg.hidden, cfg.dropout).to(device)
    freeze_all_but_last(model.backbone, cfg.unfreeze_blocks)
    criterion = DistributionLoss(torch.tensor(centers, dtype=torch.float32), cfg.lambda_l1).to(device)
    w_seg = torch.tensor(seg_class_weights(cfg, df, polys), dtype=torch.float32, device=device)
    opt = torch.optim.AdamW(param_groups(
        model.backbone, nn.ModuleList([model.head, model.seg]),
        cfg.backbone_lr, cfg.head_lr, cfg.llrd, cfg.weight_decay))
    steps = cfg.epochs * max(1, len(dl) // cfg.grad_accum)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=[g["lr"] for g in opt.param_groups], total_steps=steps, pct_start=0.15)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    ema = EMA(model, cfg.ema_decay)

    for ep in range(epoch):
        model.train()
        lam_seg = seg_lambda(cfg, ep)
        opt.zero_grad(set_to_none=True)
        run_loss, n = 0.0, 0
        for step, (x, grid, has_mask, dist, yy) in enumerate(dl):
            x, grid = x.to(device, non_blocking=True), grid.to(device)
            has_mask, dist, yy = has_mask.to(device), dist.to(device), yy.to(device)

            lam, perm = 1.0, None
            if cfg.mixup_alpha > 0:
                lam = float(np.random.beta(cfg.mixup_alpha, cfg.mixup_alpha))
                perm = torch.randperm(x.size(0), device=x.device)
                x = lam * x + (1 - lam) * x[perm]
                dist = lam * dist + (1 - lam) * dist[perm]
                yy = lam * yy + (1 - lam) * yy[perm]

            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                logits, seg = model(x)
                loss_reg = criterion(logits, dist, yy)
                if perm is None:
                    loss_seg = seg_loss(seg, grid, weight=w_seg,
                                        sample_mask=has_mask, ignore_index=IGNORE_INDEX)
                else:
                    loss_seg = seg_loss(seg, grid, grid[perm], lam, w_seg,
                                        has_mask * has_mask[perm], IGNORE_INDEX)
                loss = loss_reg + lam_seg * loss_seg

            scaler.scale(loss / cfg.grad_accum).backward()
            if (step + 1) % cfg.grad_accum == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                if sched.last_epoch < steps - 1:
                    sched.step()
                ema.update(model)
            run_loss += float(loss.detach())
            n += 1
        print(f"  эпоха {ep + 1:>3}: лосс {run_loss / max(n, 1):.4f} (λ_seg {lam_seg:.2f})",
              flush=True)

    path = cfg.ckpt_dir / f"{cfg.run_tag}_full_ep{epoch}.pt"
    torch.save({"state_dict": {k: v.detach().cpu().clone()
                               for k, v in ema.shadow.state_dict().items()},
                "cfg": cfg.__dict__, "centers": centers, "epoch": epoch,
                "n_train": len(df), "full_data": True}, path)
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VAL",
                        help="переопределить параметр Config")
    parser.add_argument("--epoch", type=int, default=DEFAULT_EPOCH,
                        help="после какой эпохи сохранить веса")
    args = parser.parse_args()
    cfg = Config().apply_overrides(args.set)
    print(f"чекпоинт: {train(cfg, load_table(cfg), args.epoch)}")


if __name__ == "__main__":
    main()
