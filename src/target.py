from __future__ import annotations

import numpy as np


def bin_centers(step: float = 5.0, lo: float = 0.0, hi: float = 100.0) -> np.ndarray:
    n = int(round((hi - lo) / step)) + 1
    return lo + step * np.arange(n, dtype=np.float64)


def soft_labels(y, centers: np.ndarray, sigma: float = 5.0) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64).reshape(-1, 1)
    d = (centers.reshape(1, -1) - y) / max(sigma, 1e-6)
    p = np.exp(-0.5 * d * d)
    p /= p.sum(axis=1, keepdims=True)
    return p.astype(np.float32)


def round_to_grid(pred, step: float = 5.0, lo: float = 0.0, hi: float = 100.0):
    return np.clip(np.round(np.asarray(pred, float) / step) * step, lo, hi)
