"""A dataset folder: scanning, caption and tag files, health checks and fixes."""

from __future__ import annotations

import re
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps

from . import tokens
from .config import data_dir
from .imaging import IMAGE_SUFFIXES, dhash, exif_orientation, file_sha, hamming, has_transparency, upright_filled
from .store import Store

# check key -> (plain name, candy name, why it matters)
HEALTH_CHECKS = {
    "too_small": ("Too small", "Fun Size", "BluFlow can drop sources below its size floor."),
    "unreadable": ("Unreadable", "Crushed", "A broken file stops latent caching."),
    "sideways": ("Sideways", "Tipped Over", "The file stores its pixels rotated and BluFlow ignores the flag that says how to turn them."),
    "transparency": ("Transparency", "Clear Wrapper", "BluFlow drops transparency without filling it, so see-through areas train as whatever color is hidden under them."),
    "extreme_shape": ("Extreme shape", "Candy Cane", "Very wide or tall images get cropped or shrunk hard in buckets."),
    "exact_duplicate": ("Exact duplicate", "Double Wrapped", "The same image twice doubles its weight."),
    "near_duplicate": ("Near duplicate", "Twin Pack", "A resized or recompressed copy doubles its weight too."),
    "missing_files": ("Missing files", "Unwrapped", "An image without tags or a description trains on less."),
    "too_many_tokens": ("Too many tokens", "Overstuffed", "Tags plus description do not fit in the trainer's caption limit."),
    "name_clash": ("Name clash", "Mixed-Up Wrappers", "Two images with the same name would share one tag file and one description."),
}


class Project:
    def __init__(self, folder: str | Path):
        self.folder = Path(folder).expanduser().resolve()
        if not self.folder.is_dir():
            raise FileNotFoundError(f"{self.folder} is not a folder.")
        self.store = Store(self.folder)
        self.id = self.store.id
        self._pairs: dict[str, list[tuple[str, str]]] = {}  # image rel -> [(check, other rel)]
        self._token_cache: dict[str, int] = {}
        self.expect_captions = False
        self.expect_tags = False

    # -- paths and files ----------------------------------------------------

    def path(self, image: dict[str, Any]) -> Path:
        return self.folder / image["rel"]

    def caption_path(self, image: dict[str, Any]) -> Path:
        path = self.path(image)
        return path.with_name(path.stem + "_nl.txt")

    def tags_path(self, image: dict[str, Any]) -> Path:
        return self.path(image).with_suffix(".txt")

    @staticmethod
    def _read(path: Path) -> str:
        try:
            return path.read_text(encoding="utf-8-sig").strip()
        except (OSError, UnicodeDecodeError):
            return ""

    def read_caption(self, image: dict[str, Any]) -> str:
        return self._read(self.caption_path(image))

    def read_tags(self, image: dict[str, Any]) -> str:
        return self._read(self.tags_path(image))

    def has_caption(self, image: dict[str, Any]) -> bool:
        return self.caption_path(image).exists()

    def write_caption(self, image: dict[str, Any], text: str, reason: str) -> None:
        """Write <stem>_nl.txt (UTF-8), keeping the previous version in history."""
        path = self.caption_path(image)
        if path.exists():
            old = self._read(path)
            if old == text.strip():
                return
            self.store.add_history(image["id"], "caption", old, reason)
        path.write_text(text.strip() + "\n", encoding="utf-8")
        self.store.update_image(image["id"], caption=text.strip())

    def write_tags(self, image: dict[str, Any], text: str, reason: str) -> None:
        path = self.tags_path(image)
        if path.exists():
            old = self._read(path)
            if old == text.strip():
                return
            self.store.add_history(image["id"], "tags", old, reason)
        path.write_text(text.strip() + "\n", encoding="utf-8")

    # -- scanning -----------------------------------------------------------

    def scan(self) -> list[dict[str, Any]]:
        """Find the images, read what the health checks need, and pair up duplicates."""
        settings = self.store.settings()
        walker = self.folder.rglob("*") if settings.get("recursive") else self.folder.iterdir()
        files = sorted(p for p in walker if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
        present = set()
        for path in files:
            rel = path.relative_to(self.folder).as_posix()
            present.add(rel)
            existing = self.store.image_by_rel(rel)
            stat = path.stat()
            if existing and existing["size"] == stat.st_size and existing["mtime"] == stat.st_mtime and existing["health"]:
                continue
            image_id = existing["id"] if existing else self.store.add_image(rel)
            self.store.update_image(image_id, size=stat.st_size, mtime=stat.st_mtime, **self._inspect(path))
        self.store.mark_missing(present)
        images = self.store.images()
        self._pair_up(images)
        self.refresh_expectations(images)
        return images

    def refresh_expectations(self, images: list[dict[str, Any]] | None = None) -> None:
        images = images if images is not None else self.store.images()
        self.expect_captions = any(self.has_caption(image) for image in images)
        # A folder with no tag files at all is a dataset without tags, not one with tags missing.
        self.expect_tags = any(self.tags_path(image).exists() for image in images)

    @staticmethod
    def _inspect(path: Path) -> dict[str, Any]:
        info: dict[str, Any] = {"width": 0, "height": 0, "sha": None, "phash": None,
                                "health": {"unreadable": None, "orientation": 1, "transparent": False}}
        try:
            info["sha"] = file_sha(path)
            with Image.open(path) as img:
                orientation = exif_orientation(img)
                img.load()
                upright = ImageOps.exif_transpose(img)
                info["width"], info["height"] = upright.size
                info["health"]["orientation"] = orientation
                info["health"]["transparent"] = has_transparency(img)
                info["phash"] = dhash(img)
        except Exception as exc:
            info["health"]["unreadable"] = f"{type(exc).__name__}: {exc}"[:200]
        return info

    def _pair_up(self, images: list[dict[str, Any]]) -> None:
        pairs: dict[str, list[tuple[str, str]]] = defaultdict(list)
        by_sha, by_stem = defaultdict(list), defaultdict(list)
        for image in images:
            if image["sha"]:
                by_sha[image["sha"]].append(image["rel"])
            rel = Path(image["rel"])
            by_stem[(rel.parent / rel.stem).as_posix().lower()].append(image["rel"])
        for group in by_sha.values():
            for rel in group:
                pairs[rel].extend(("exact_duplicate", other) for other in group if other != rel)
        for group in by_stem.values():
            for rel in group:
                pairs[rel].extend(("name_clash", other) for other in group if other != rel)
        hashed = [(i["rel"], i["phash"], i["sha"]) for i in images if i["phash"]]
        for a in range(len(hashed)):
            for b in range(a + 1, len(hashed)):
                if hashed[a][2] != hashed[b][2] and hamming(hashed[a][1], hashed[b][1]) <= 4:
                    pairs[hashed[a][0]].append(("near_duplicate", hashed[b][0]))
                    pairs[hashed[b][0]].append(("near_duplicate", hashed[a][0]))
        self._pairs = pairs

    # -- health -------------------------------------------------------------

    def token_count(self, image: dict[str, Any], app_settings: dict[str, Any],
                    tags: str | None = None, caption: str | None = None) -> int:
        """Tags plus description, as the trainer's mixed mode can combine them."""
        tags = self.read_tags(image) if tags is None else tags
        caption = self.read_caption(image) if caption is None else caption
        total = 0
        for text in (tags, caption):
            if text not in self._token_cache:
                if len(self._token_cache) > 50000:
                    self._token_cache.clear()
                self._token_cache[text] = tokens.count(text, app_settings)
            total += self._token_cache[text]
        return total

    def health(self, image: dict[str, Any], app_settings: dict[str, Any]) -> list[dict[str, str]]:
        """Problems with one image, each as {check, detail}."""
        settings = self.store.settings()
        facts = image.get("health") or {}
        out = []
        if facts.get("unreadable"):
            return [{"check": "unreadable", "detail": facts["unreadable"]}]
        width, height = image["width"] or 0, image["height"] or 0
        floor = float(settings.get("min_area_mp", 0.5)) * 1_000_000
        if width * height < floor:
            out.append({"check": "too_small", "detail": f"{width}×{height} is {width * height / 1e6:.2f} MP, under the {floor / 1e6:.2f} MP floor"})
        if facts.get("orientation", 1) != 1:
            out.append({"check": "sideways", "detail": f"EXIF orientation {facts['orientation']}"})
        if facts.get("transparent"):
            out.append({"check": "transparency", "detail": "has see-through pixels"})
        if width and height and max(width / height, height / width) > 3:
            out.append({"check": "extreme_shape", "detail": f"aspect ratio {width}:{height} is outside 1:3 to 3:1"})
        for check, other in self._pairs.get(image["rel"], []):
            out.append({"check": check, "detail": f"matches {other}"})
        # A missing file only counts once this folder has that kind of file at all.
        wanted = [("tags", self.tags_path(image))] if self.expect_tags else []
        if self.expect_captions:
            wanted.append(("description", self.caption_path(image)))
        missing = [name for name, path in wanted if not path.exists()]
        if missing:
            out.append({"check": "missing_files", "detail": "no " + " or ".join(missing) + " file"})
        limit = int(app_settings.get("token_limit", 512))
        count = self.token_count(image, app_settings)
        if count > limit:
            out.append({"check": "too_many_tokens", "detail": f"{count} tokens, over the {limit}-token limit"})
        return out

    # -- fixes --------------------------------------------------------------

    def _backup(self, image: dict[str, Any]) -> Path:
        target = data_dir() / "backups" / self.id / image["rel"]
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copy2(self.path(image), target)
        return target

    def fix(self, image: dict[str, Any], kind: str) -> None:
        """Save an upright copy ('sideways') or an upright, filled copy ('transparency') in place.

        The original is copied to the app's backup folder first.
        """
        path = self.path(image)
        self._backup(image)
        with Image.open(path) as opened:
            fmt = opened.format or "PNG"
            if kind == "transparency":
                fixed = upright_filled(opened, self.store.settings().get("background", "#ffffff"))
            else:
                fixed = ImageOps.exif_transpose(opened)
                fixed.load()
        options: dict[str, Any] = {}
        if fmt == "JPEG":
            fixed = fixed.convert("RGB")
            options = {"quality": 95, "subsampling": 0}
        elif fmt == "WEBP":
            options = {"quality": 95}
        fixed.save(path, fmt, **options)
        stat = path.stat()
        self.store.update_image(image["id"], size=stat.st_size, mtime=stat.st_mtime, **self._inspect(path))
        self._pair_up(self.store.images())

    # -- find and replace ---------------------------------------------------

    def replace_preview(self, find: str, replace: str, *, regex: bool = False, case: bool = False,
                        whole: bool = False) -> list[dict[str, Any]]:
        if not find:
            return []
        pattern = find if regex else re.escape(find)
        if whole:
            pattern = r"(?<!\w)(?:" + pattern + r")(?!\w)"
        compiled = re.compile(pattern, 0 if case else re.I)
        changes = []
        for image in self.store.images():
            before = self.read_caption(image)
            if not before:
                continue
            after = compiled.sub(replace if regex else replace.replace("\\", "\\\\"), before)
            after = re.sub(r" {2,}", " ", after).strip()
            if after != before:
                changes.append({"id": image["id"], "rel": image["rel"], "before": before, "after": after})
        return changes

    def replace_apply(self, changes: list[dict[str, Any]], only: set[int] | None = None) -> int:
        done = 0
        for change in changes:
            if only is not None and change["id"] not in only:
                continue
            image = self.store.image(change["id"])
            if image:
                self.write_caption(image, change["after"], "find and replace")
                done += 1
        return done
