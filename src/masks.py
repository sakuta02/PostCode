from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

BACKGROUND = 0
PAINT_ORDER = ("frame", "floor", "cargo")
CLASS_NAMES = ("background",) + PAINT_ORDER
CLASS_IDS = {name: i + 1 for i, name in enumerate(PAINT_ORDER)}
N_CLASSES = len(CLASS_NAMES)
IGNORE_INDEX = 255


def load_polygons(coco_path, classes=PAINT_ORDER) -> dict:
    coco = json.loads(Path(coco_path).read_text(encoding="utf-8"))
    cats = {c["id"]: c["name"] for c in coco["categories"]}
    imgs = {i["id"]: i for i in coco["images"]}
    wanted = set(classes)

    out: dict[str, dict] = {}
    for ann in coco["annotations"]:
        name = cats.get(ann["category_id"])
        if name not in wanted:
            continue
        info = imgs.get(ann["image_id"])
        if info is None:
            continue
        stem = Path(info["file_name"]).stem
        for poly in ann.get("segmentation", []):
            pts = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
            if len(pts) >= 3:
                out.setdefault(stem, {}).setdefault(name, []).append(pts)
    return out


def rasterize(polys: dict, size_wh, scale: int = 4) -> Image.Image:
    w, h = size_wh
    mw, mh = max(1, w // scale), max(1, h // scale)
    mask = Image.new("L", (mw, mh), BACKGROUND)
    draw = ImageDraw.Draw(mask)
    for name in PAINT_ORDER:
        for pts in polys.get(name, []):
            flat = [(float(x) / scale, float(y) / scale) for x, y in pts]
            if len(flat) >= 3:
                draw.polygon(flat, fill=CLASS_IDS[name])
    return mask


def pad_to_square(mask: Image.Image, fill: int = BACKGROUND) -> Image.Image:
    w, h = mask.size
    if w == h:
        return mask
    side = max(w, h)
    canvas = Image.new("L", (side, side), fill)
    canvas.paste(mask, ((side - w) // 2, (side - h) // 2))
    return canvas


def majority_pool(mask: np.ndarray, out_size: int) -> np.ndarray:
    mask = np.asarray(mask)
    h, w = mask.shape
    if h % out_size or w % out_size:
        raise ValueError(f"маска {mask.shape} не кратна сетке {out_size}")
    bh, bw = h // out_size, w // out_size
    blocks = mask.reshape(out_size, bh, out_size, bw).transpose(0, 2, 1, 3)
    blocks = blocks.reshape(out_size, out_size, bh * bw)

    out = np.full((out_size, out_size), IGNORE_INDEX, dtype=np.uint8)
    valid = blocks != IGNORE_INDEX
    counts = np.stack([((blocks == c) & valid).sum(-1) for c in range(N_CLASSES)], -1)
    total = counts.sum(-1)
    filled = total > 0
    out[filled] = counts.argmax(-1)[filled].astype(np.uint8)
    return out


def occupied_ratio(mask: np.ndarray) -> float:
    cargo = float((mask == CLASS_IDS["cargo"]).sum())
    floor = float((mask == CLASS_IDS["floor"]).sum())
    denom = cargo + floor
    return 100.0 * cargo / denom if denom > 0 else float("nan")


def class_weights(masks, n_classes: int = N_CLASSES) -> np.ndarray:
    counts = np.zeros(n_classes, dtype=np.float64)
    for m in masks:
        m = np.asarray(m)
        vals, cnt = np.unique(m[m != IGNORE_INDEX], return_counts=True)
        counts[vals.astype(int)] += cnt
    freq = counts / max(counts.sum(), 1.0)
    w = 1.0 / np.sqrt(np.clip(freq, 1e-6, None))
    w[counts == 0] = 0.0
    nz = w > 0
    if nz.any():
        w[nz] /= w[nz].mean()
    return w


OVERLAY_COLORS = {1: (60, 120, 220), 2: (70, 190, 120), 3: (230, 110, 60)}


def overlay_mask(img: Image.Image, grid: np.ndarray, alpha: float = 0.45) -> Image.Image:
    w, h = img.size
    side = max(w, h)
    big = np.asarray(Image.fromarray(np.asarray(grid, dtype=np.uint8)).resize(
        (side, side), Image.NEAREST))
    ox, oy = (side - w) // 2, (side - h) // 2
    labels = big[oy:oy + h, ox:ox + w]

    base = np.asarray(img.convert("RGB"), dtype=np.float32)
    tint = base.copy()
    for cls, color in OVERLAY_COLORS.items():
        m = labels == cls
        if m.any():
            tint[m] = np.asarray(color, np.float32)
    return Image.fromarray((base * (1 - alpha) + tint * alpha).astype(np.uint8))
