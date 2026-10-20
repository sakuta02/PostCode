from __future__ import annotations

import io
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from errors import ApiError


@dataclass
class Prediction:
    load_pct: float
    low: float
    high: float
    confidence: str
    occupied_ratio: float | None
    mask_png: bytes | None
    model_version: str
    mode: str
    n_models: int


def confidence_label(low: float, high: float) -> str:
    width = high - low
    return "high" if width <= 15 else "medium" if width <= 30 else "low"


class TorchPredictor:
    def __init__(self, settings):
        self.s = settings
        self._lock = threading.RLock()
        self._cache: dict[str, tuple] = {}
        self._fast_path: Path = Path(settings.fast_path)
        self._fast_version = self._fast_path.stem
        self.device = self._resolve_device()
        os.environ.setdefault("LOADEST_SKIP_PRETRAINED", "1")

    def _resolve_device(self) -> str:
        if self.s.device != "auto":
            return self.s.device
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    def paths_for(self, mode: str) -> list[Path]:
        if mode == "fast":
            return [self._fast_path] if self._fast_path.exists() else []
        if mode == "accurate":
            return self.s.accurate_paths()
        raise ApiError("INVALID_ARGUMENT", f"Неизвестный режим {mode}",
                       details="допустимо: fast, accurate")

    def set_fast(self, path: Path, version: str) -> None:
        path = Path(path)
        if not path.exists():
            raise ApiError("MODEL_NOT_FOUND", details=str(path))
        try:
            self._load(path)
        except Exception as e:
            raise ApiError("MODEL_UNAVAILABLE",
                           details=f"не удалось загрузить {path.name}: {e}") from None
        with self._lock:
            old = self._fast_path
            self._fast_path, self._fast_version = path, version
        if old != path:
            self.evict(old)

    def evict(self, path: Path) -> None:
        path = Path(path)
        with self._lock:
            if path != self._fast_path and path not in self.s.accurate_paths():
                self._cache.pop(str(path), None)

    @property
    def fast_version(self) -> str:
        return self._fast_version

    @property
    def fast_path(self) -> Path:
        return self._fast_path

    def version_for(self, mode: str) -> str:
        return self._fast_version if mode == "fast" else f"ensemble-{len(self.s.accurate_paths())}"

    def _load(self, path: Path):
        key = str(path)
        if key in self._cache:
            return self._cache[key]
        import torch

        from config import PATCH
        from model import MultiTaskModel

        blob = torch.load(path, map_location="cpu", weights_only=False)
        c = blob.get("cfg", {})
        centers = np.asarray(blob["centers"], dtype=np.float64)
        model = MultiTaskModel(c.get("backbone", "dinov2_vitb14"), len(centers),
                               c.get("hidden", 256), c.get("dropout", 0.2))
        model.load_state_dict(blob["state_dict"])
        model.to(self.device).eval()

        img_size = int(c.get("img_size", c.get("e2e_img_size", 392)))
        lam = c.get("lambda_seg")
        seg_ok = (lam is None or float(lam) > 0) and c.get("coco_json", "?") != ""
        meta = SimpleNamespace(centers=centers, img_size=img_size, grid=img_size // PATCH,
                               mult=int(c.get("seg_grid_mult", 4)),
                               step=float(c.get("bin_step", 5.0)), seg_ok=seg_ok)
        self._cache[key] = (model, meta)
        return self._cache[key]

    def warmup(self) -> None:
        for p in self.paths_for("fast"):
            self._load(p)

    def _forward(self, img, paths: list[Path]):
        import torch

        from joint_augment import JointTransform

        flips = (False, True) if self.s.tta else (False,)
        probs_sum, n, seg_sum, seg_n, meta0 = None, 0, None, 0, None
        prepared: dict[tuple, "torch.Tensor"] = {}

        with self._lock:
            for path in paths:
                model, meta = self._load(path)
                meta0 = meta0 or meta
                if len(meta.centers) != len(meta0.centers):
                    raise ApiError("INTERNAL_ERROR", details="модели ансамбля с разной сеткой бинов")
                key = (meta.img_size, meta.grid, meta.mult)
                if key not in prepared:
                    xs = [JointTransform(meta.img_size, meta.grid, train=False,
                                         grid_mult=meta.mult, hflip=f)(img, None)[0] for f in flips]
                    prepared[key] = torch.stack(xs).to(self.device)
                x = prepared[key]
                with torch.no_grad(), torch.autocast(
                        "cuda", dtype=torch.float16, enabled=self.device == "cuda"):
                    logits, seg = model(x)
                p = torch.softmax(logits.float(), dim=-1).cpu().numpy()
                probs_sum = p.sum(0) if probs_sum is None else probs_sum + p.sum(0)
                n += len(p)
                if meta.seg_ok:
                    sp = torch.softmax(seg[0].float(), dim=0).cpu().numpy()
                    if seg_sum is None or seg_sum.shape == sp.shape:
                        seg_sum = sp if seg_sum is None else seg_sum + sp
                        seg_n += 1
        return (probs_sum / n)[None], seg_sum, seg_n, meta0

    @staticmethod
    def _point(probs, meta) -> float:
        from decode import decode_median
        from target import round_to_grid

        return float(round_to_grid(np.clip(decode_median(probs, meta.centers), 0, 100),
                                   meta.step)[0])

    def predict_paths(self, img, paths: list[Path]) -> float:
        probs, _, _, meta = self._forward(img, [Path(p) for p in paths])
        return self._point(probs, meta)

    def predict(self, img, mode: str, return_mask: bool = False) -> Prediction:
        paths = self.paths_for(mode)
        if not paths:
            raise ApiError("MODEL_UNAVAILABLE",
                           details=f"режим {mode}: веса не найдены в {self.s.models_dir}")
        from decode import decode_quantile
        from masks import occupied_ratio, overlay_mask

        probs, seg_sum, seg_n, meta = self._forward(img, paths)
        load = self._point(probs, meta)
        q = (1 - self.s.interval_level) / 2
        low = float(np.clip(decode_quantile(probs, meta.centers, q)[0], 0, 100))
        high = float(np.clip(decode_quantile(probs, meta.centers, 1 - q)[0], 0, 100))
        low, high = min(low, load), max(high, load)

        occ, mask_png = None, None
        if seg_sum is not None:
            grid = (seg_sum / seg_n).argmax(0).astype(np.uint8)
            r = occupied_ratio(grid)
            occ = float(r) if np.isfinite(r) else None
            if return_mask:
                vis = overlay_mask(img, grid)
                vis.thumbnail((1024, 1024))
                buf = io.BytesIO()
                vis.save(buf, format="PNG", optimize=True)
                mask_png = buf.getvalue()

        return Prediction(load_pct=load, low=round(low, 1), high=round(high, 1),
                          confidence=confidence_label(low, high),
                          occupied_ratio=None if occ is None else round(occ, 1),
                          mask_png=mask_png, model_version=self.version_for(mode), mode=mode,
                          n_models=len(paths))

    def info(self) -> dict:
        fast, acc = self.paths_for("fast"), self.s.accurate_paths()
        return {
            "device": self.device,
            "modes": {"fast": {"available": bool(fast), "models": len(fast)},
                      "accurate": {"available": bool(acc), "models": len(acc)}},
            "status": "ok" if fast else "degraded",
            "active_model": self._fast_version,
        }
