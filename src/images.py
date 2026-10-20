from __future__ import annotations

import base64
import binascii
import hashlib
import io
import re

from PIL import Image, ImageFile, ImageOps, UnidentifiedImageError

from errors import ApiError

ImageFile.LOAD_TRUNCATED_IMAGES = False
SUPPORTED = {"JPEG", "PNG", "WEBP", "BMP"}

try:
    from pillow_heif import register_heif_opener

    register_heif_opener()
    SUPPORTED |= {"HEIC", "HEIF"}
except ImportError:
    pass

_HINTS = {
    "HEIC": "Фото в HEIC (формат iPhone). Сохраните его как JPEG — "
            "в настройках камеры «Наиболее совместимый».",
    "PDF": "Передан PDF-документ, а не фотография.",
    "GIF": "GIF не поддерживается, используйте JPEG или PNG.",
    "TIFF": "TIFF не поддерживается, используйте JPEG или PNG.",
    "AVIF": "AVIF не поддерживается, используйте JPEG или PNG.",
}
_HEIF_BRANDS = (b"heic", b"heix", b"heim", b"heis", b"hevc", b"hevx", b"mif1", b"msf1")


def sniff(data: bytes) -> str | None:
    head = data[:32]
    if head.startswith(b"\xff\xd8\xff"):
        return "JPEG"
    if head.startswith(b"\x89PNG\r\n\x1a\n"):
        return "PNG"
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return "WEBP"
    if head.startswith(b"BM"):
        return "BMP"
    if head.startswith(b"GIF8"):
        return "GIF"
    if head.startswith(b"%PDF"):
        return "PDF"
    if head[:4] in (b"II*\x00", b"MM\x00*"):
        return "TIFF"
    if head[4:8] == b"ftyp" and head[8:12] in (b"avif", b"avis"):
        return "AVIF"
    if head[4:8] == b"ftyp" and head[8:12] in _HEIF_BRANDS:
        return "HEIC"
    return None


def decode_base64(text: str | None) -> bytes:
    if not text or not text.strip():
        raise ApiError("IMAGE_EMPTY")
    try:
        return base64.b64decode(re.sub(r"\s+", "", text), validate=True)
    except (binascii.Error, ValueError) as e:
        raise ApiError("BASE64_INVALID", details=str(e)) from None


def load_image(data: bytes, settings) -> tuple[Image.Image, dict]:
    if not data:
        raise ApiError("IMAGE_EMPTY")
    limit = settings.max_image_mb * 1024 * 1024
    if len(data) > limit:
        raise ApiError("IMAGE_TOO_LARGE",
                       details=f"{len(data)/1e6:.1f} МБ при лимите {settings.max_image_mb:g} МБ")

    kind = sniff(data)
    if kind is None:
        raise ApiError("IMAGE_UNSUPPORTED_FORMAT",
                       "Данные не распознаны как изображение",
                       details="поддерживаются JPEG, PNG, WEBP, BMP")
    if kind not in SUPPORTED:
        raise ApiError("IMAGE_UNSUPPORTED_FORMAT",
                       f"Формат {kind} не поддерживается", details=_HINTS.get(kind))

    Image.MAX_IMAGE_PIXELS = settings.max_pixels
    try:
        with Image.open(io.BytesIO(data)) as probe:
            probe.verify()
        img = Image.open(io.BytesIO(data))
        img.load()
    except Image.DecompressionBombError as e:
        raise ApiError("IMAGE_TOO_LARGE", "Слишком много пикселей", details=str(e)) from None
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as e:
        raise ApiError("IMAGE_CORRUPTED",
                       f"Файл похож на {kind}, но не декодируется", details=str(e)) from None

    fmt = (img.format or kind).upper()
    if fmt not in SUPPORTED:
        raise ApiError("IMAGE_UNSUPPORTED_FORMAT", f"Формат {fmt} не поддерживается")

    if settings.exif_transpose:
        try:
            img = ImageOps.exif_transpose(img)
        except Exception:
            pass
    img = img.convert("RGB")

    w, h = img.size
    if min(w, h) < settings.min_side:
        raise ApiError("IMAGE_TOO_SMALL",
                       details=f"{w}x{h}, минимальная сторона {settings.min_side} пикселей")

    return img, {
        "format": fmt, "width": w, "height": h, "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }
