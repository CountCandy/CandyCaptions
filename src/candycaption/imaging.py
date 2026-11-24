"""Image preparation: show the model what the trainer will see.

Images are turned upright (EXIF) and any transparency is filled with the
dataset's background color before they are downscaled and sent.
"""

from __future__ import annotations

import base64
import hashlib
import io
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

Image.MAX_IMAGE_PIXELS = None
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def parse_color(value: str) -> tuple[int, int, int]:
    value = (value or "#ffffff").lstrip("#")
    if len(value) != 6:
        return (255, 255, 255)
    try:
        return tuple(int(value[i:i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]
    except ValueError:
        return (255, 255, 255)


def has_transparency(img: Image.Image) -> bool:
    if img.mode == "P" and "transparency" in img.info:
        img = img.convert("RGBA")
    if img.mode in ("RGBA", "LA", "PA"):
        return img.getchannel("A").getextrema()[0] < 255
    return False


def exif_orientation(img: Image.Image) -> int:
    try:
        return int(img.getexif().get(0x0112, 1) or 1)
    except Exception:
        return 1


def upright_filled(img: Image.Image, background: str = "#ffffff") -> Image.Image:
    """Upright RGB copy with transparency filled."""
    img = ImageOps.exif_transpose(img)
    if img.mode == "P" and "transparency" in img.info:
        img = img.convert("RGBA")
    if img.mode in ("RGBA", "LA", "PA"):
        rgba = img.convert("RGBA")
        canvas = Image.new("RGBA", rgba.size, parse_color(background) + (255,))
        canvas.alpha_composite(rgba)
        return canvas.convert("RGB")
    return img.convert("RGB")


def prepare_bytes(path: Path, long_side: int, background: str = "#ffffff") -> bytes:
    """JPEG bytes of the image as the model should see it."""
    with Image.open(path) as opened:
        img = upright_filled(opened, background)
    if long_side and max(img.size) > long_side:
        img.thumbnail((long_side, long_side), Image.LANCZOS)
    buffer = io.BytesIO()
    # High quality and full color resolution, so the copy the model sees adds no compression marks of its own.
    img.save(buffer, "JPEG", quality=96, subsampling=0)
    return buffer.getvalue()


def data_url(jpeg: bytes) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(jpeg).decode("ascii")


def prepare_data_url(path: Path, long_side: int, background: str = "#ffffff") -> str:
    return data_url(prepare_bytes(path, long_side, background))


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def dhash(img: Image.Image) -> str:
    """64-bit difference hash; resized or recompressed copies land a few bits apart."""
    small = upright_filled(img).convert("L").resize((9, 8), Image.LANCZOS)
    pixels = small.tobytes()
    bits = 0
    for row in range(8):
        for col in range(8):
            bits = (bits << 1) | (pixels[row * 9 + col] > pixels[row * 9 + col + 1])
    return f"{bits:016x}"


def hamming(a: str, b: str) -> int:
    return bin(int(a, 16) ^ int(b, 16)).count("1")


def tiny_test_image() -> str:
    """A small image with two colored squares, for Test connection."""
    img = Image.new("RGB", (64, 64), (255, 255, 255))
    for x in range(8, 30):
        for y in range(8, 30):
            img.putpixel((x, y), (220, 30, 40))
    for x in range(34, 56):
        for y in range(34, 56):
            img.putpixel((x, y), (30, 60, 220))
    buffer = io.BytesIO()
    img.save(buffer, "JPEG", quality=90)
    return data_url(buffer.getvalue())


STATS_VERSION = 3
SIGNATURE_SIDE = 32      # a tiny grayscale copy kept with the measurements, for finding look-alikes


def _grid_ratio(gray: np.ndarray, axis: int) -> float:
    """Edge strength on JPEG's 8-pixel grid against edge strength elsewhere. 1.0 means no grid."""
    edges = np.abs(np.diff(gray, axis=axis)).mean(axis=1 - axis)
    on = edges[7::8]
    if len(on) < 8:
        return 1.0
    off = np.delete(edges, np.s_[7::8])
    return float(on.mean() / max(float(off.mean()), 1e-6))


def measure(path: Path, background: str = "#ffffff") -> dict[str, float]:
    """Pixel measurements of one image. Cheap enough to run on every scan."""
    with Image.open(path) as opened:
        img = upright_filled(opened, background)
    width, height = img.size
    full = np.asarray(img.convert("L"), dtype=np.float32)
    blocking = max(_grid_ratio(full, 0), _grid_ratio(full, 1))

    # Sharpness and grain are compared at one common size, so a 4K image and a 1080p one are judged alike.
    mid = img.copy()
    mid.thumbnail((1536, 1536), Image.BILINEAR, reducing_gap=2.0)
    luma = np.asarray(mid.convert("L"), dtype=np.float32)
    laplace = 4 * luma[1:-1, 1:-1] - luma[:-2, 1:-1] - luma[2:, 1:-1] - luma[1:-1, :-2] - luma[1:-1, 2:]
    blurred = (luma[:-2, :-2] + luma[:-2, 1:-1] + luma[:-2, 2:] + luma[1:-1, :-2] + luma[1:-1, 1:-1] + luma[1:-1, 2:]
               + luma[2:, :-2] + luma[2:, 1:-1] + luma[2:, 2:]) / 9
    grain = float(np.median(np.abs(luma[1:-1, 1:-1] - blurred)))

    small = mid.copy()
    small.thumbnail((512, 512), Image.BILINEAR, reducing_gap=2.0)
    rgb = np.asarray(small, dtype=np.float32)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    rg, yb = r - g, 0.5 * (r + g) - b
    colorfulness = math.sqrt(float(rg.std()) ** 2 + float(yb.std()) ** 2) + 0.3 * math.sqrt(float(rg.mean()) ** 2 + float(yb.mean()) ** 2)
    hsv = np.asarray(small.convert("HSV"), dtype=np.float32)
    weight = hsv[..., 1] * hsv[..., 2]
    hues, _ = np.histogram(hsv[..., 0], bins=12, range=(0, 256), weights=weight)
    hues = hues / max(float(hues.sum()), 1e-6)
    light = 0.299 * r + 0.587 * g + 0.114 * b
    detail = float(np.abs(np.diff(light, axis=0)).mean() + np.abs(np.diff(light, axis=1)).mean())
    return {
        "v": STATS_VERSION,
        "megapixels": round(width * height / 1e6, 3),
        "sharpness": round(math.log10(float(laplace.var()) + 1e-3), 4),
        "blocking": round(blocking, 4),
        "grain": round(math.log10(grain + 0.05), 4),
        "colorfulness": round(colorfulness, 3),
        "saturation": round(float(hsv[..., 1].mean()), 3),
        "brightness": round(float(light.mean()), 3),
        "contrast": round(float(light.std()), 3),
        "detail": round(detail, 3),
        "hues": [round(float(h), 4) for h in hues],
        "sig": base64.b64encode(small.convert("L").resize((SIGNATURE_SIDE, SIGNATURE_SIDE), Image.BOX).tobytes()).decode("ascii"),
    }
