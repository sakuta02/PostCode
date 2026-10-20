from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _env(name: str, default: str) -> str:
    return os.getenv(f"LOADEST_{name}", default)


def _flag(name: str, default: bool) -> bool:
    return _env(name, "1" if default else "0").strip().lower() in {"1", "true", "yes"}


@dataclass
class Settings:
    models_dir: Path = field(default_factory=lambda: Path(_env("MODELS_DIR", "models")))
    fast_checkpoint: str = field(default_factory=lambda: _env("FAST_CKPT", "fast_model.pt"))
    accurate_glob: str = field(default_factory=lambda: _env("ACCURATE_GLOB", "*_fold*.pt"))
    default_mode: str = field(default_factory=lambda: _env("DEFAULT_MODE", "fast"))

    data_dir: Path = field(default_factory=lambda: Path(_env("DATA_DIR", "var")))
    train_dir: Path = field(default_factory=lambda: Path(_env("TRAIN_DIR", "data/train")))
    coco_json: str = field(default_factory=lambda: _env(
        "COCO_JSON", str(PROJECT_ROOT / "annotations" / "masks.json")))

    tta: bool = field(default_factory=lambda: _flag("TTA", True))
    exif_transpose: bool = field(default_factory=lambda: _flag("EXIF_TRANSPOSE", True))
    device: str = field(default_factory=lambda: _env("DEVICE", "auto"))

    max_body_mb: float = field(default_factory=lambda: float(_env("MAX_BODY_MB", "25")))
    max_image_mb: float = field(default_factory=lambda: float(_env("MAX_IMAGE_MB", "15")))
    max_pixels: int = field(default_factory=lambda: int(_env("MAX_PIXELS", str(64_000_000))))
    min_side: int = field(default_factory=lambda: int(_env("MIN_SIDE", "64")))

    retrain_async: bool = field(default_factory=lambda: _flag("RETRAIN_ASYNC", True))
    retrain_gate_tol: float = field(default_factory=lambda: float(_env("RETRAIN_GATE_TOL", "2.0")))
    retrain_control_n: int = field(default_factory=lambda: int(_env("RETRAIN_CONTROL_N", "60")))

    dedup: bool = field(default_factory=lambda: _flag("DEDUP", True))
    api_key: str = field(default_factory=lambda: _env("API_KEY", ""))
    interval_level: float = 0.8

    def __post_init__(self):
        for name in ("models_dir", "data_dir", "train_dir"):
            setattr(self, name, Path(getattr(self, name)))

    @property
    def fast_path(self) -> Path:
        return self.models_dir / "fast" / self.fast_checkpoint

    def accurate_paths(self) -> list[Path]:
        return sorted((self.models_dir / "ensemble").glob(self.accurate_glob))

    @property
    def db_path(self) -> Path:
        return self.data_dir / "loadest.sqlite3"

    @property
    def images_dir(self) -> Path:
        return self.data_dir / "images"

    @property
    def trained_models_dir(self) -> Path:
        return self.data_dir / "models"

    @property
    def jobs_dir(self) -> Path:
        return self.data_dir / "jobs"

    @property
    def schema_path(self) -> Path:
        return PROJECT_ROOT / "schema" / "loadest_api.xsd"

    def ensure_dirs(self) -> None:
        for d in (self.data_dir, self.images_dir, self.trained_models_dir, self.jobs_dir):
            d.mkdir(parents=True, exist_ok=True)
