from __future__ import annotations

from pathlib import Path

import pandas as pd
from PIL import Image

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class PadToSquare:
    def __init__(self, fill=(124, 116, 104)):
        self.fill = fill

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        if w == h:
            return img
        side = max(w, h)
        canvas = Image.new("RGB", (side, side), self.fill)
        canvas.paste(img, ((side - w) // 2, (side - h) // 2))
        return canvas


def load_table(cfg) -> pd.DataFrame:
    labels = pd.read_csv(cfg.labels_csv)
    labels["image_id"] = labels["image_id"].astype(str)
    labels["path"] = [str(cfg.images_dir / f"{i}.jpg") for i in labels["image_id"]]
    found = [Path(p).exists() for p in labels["path"]]
    if not any(found):
        raise FileNotFoundError(f"нет изображений из {cfg.labels_csv} в {cfg.images_dir}")
    return labels[found][["image_id", "path", "load_pct"]].reset_index(drop=True)
