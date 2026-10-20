from __future__ import annotations

import numpy as np
from PIL import Image

MIN_SIDE_OK = 480
DARK_LEVEL = 20
BRIGHT_LEVEL = 245
MAX_DARK_SHARE = 50.0
MIN_BRIGHTNESS = 12.0
MAX_BRIGHT_SHARE = 25.0
MAX_BRIGHTNESS = 80.0
MIN_SHARPNESS = 15.0

MESSAGES = {
    "LOW_RESOLUTION": "Низкое разрешение: мелкие детали груза могут быть не видны",
    "TOO_DARK": "Фото слишком тёмное: границы груза и пола плохо различимы",
    "OVEREXPOSED": "Фото пересвечено: часть кадра засвечена",
    "BLURRY": "Фото смазано или не в фокусе",
}


def assess(img: Image.Image) -> dict:
    width, height = img.size
    gray = img.convert("L")
    gray.thumbnail((1024, 1024))
    g = np.asarray(gray, dtype=np.float32)

    brightness = float(g.mean() / 255 * 100)
    dark_share = float((g < DARK_LEVEL).mean() * 100)
    bright_share = float((g > BRIGHT_LEVEL).mean() * 100)
    lap = g[1:-1, :-2] + g[1:-1, 2:] + g[:-2, 1:-1] + g[2:, 1:-1] - 4 * g[1:-1, 1:-1]
    sharpness = float(lap.var())

    issues = []
    if min(width, height) < MIN_SIDE_OK:
        issues.append("LOW_RESOLUTION")
    if brightness < MIN_BRIGHTNESS or dark_share > MAX_DARK_SHARE:
        issues.append("TOO_DARK")
    if bright_share > MAX_BRIGHT_SHARE or brightness > MAX_BRIGHTNESS:
        issues.append("OVEREXPOSED")
    if sharpness < MIN_SHARPNESS:
        issues.append("BLURRY")

    return {
        "status": "warning" if issues else "ok",
        "width": width, "height": height,
        "brightness": round(brightness, 1),
        "dark_share": round(dark_share, 1),
        "bright_share": round(bright_share, 1),
        "sharpness": round(sharpness, 1),
        "issues": [(code, MESSAGES[code]) for code in issues],
    }
