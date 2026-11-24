"""Image preparation: show the model what the trainer will see.

Images are turned upright (EXIF) and any transparency is filled with the
dataset's background color before they are downscaled and sent.
"""

from __future__ import annotations

import base64
import hashlib
import io
from pathlib import Path

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
