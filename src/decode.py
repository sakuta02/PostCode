from __future__ import annotations

import numpy as np


def decode_quantile(probs: np.ndarray, centers: np.ndarray, q: float = 0.5) -> np.ndarray:
    probs = np.asarray(probs, dtype=np.float64)
    cdf = np.cumsum(probs, axis=1)
    cdf /= cdf[:, -1:]
    idx = np.argmax(cdf >= q, axis=1)
    rows = np.arange(len(idx))
    prev = np.where(idx > 0, cdf[rows, np.maximum(idx - 1, 0)], 0.0)
    p = probs[rows, idx]
    frac = np.clip(np.where(p > 1e-12, (q - prev) / np.maximum(p, 1e-12), 0.5), 0, 1)
    step = float(centers[1] - centers[0])
    return np.clip(centers[idx] - step / 2 + frac * step, centers[0], centers[-1])


def decode_median(probs: np.ndarray, centers: np.ndarray) -> np.ndarray:
    return decode_quantile(probs, centers, 0.5)
