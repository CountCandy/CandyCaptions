"""Token counter for the trainer's caption limit.

Order of preference: a local tokenizer folder from the settings (BluFlow
bundles the MOSS/Qwen tokenizer), then the cached download in the app's data
folder, then a first-run download. Only when none of those works does the
count fall back to an approximation, and the result says so.
"""

from __future__ import annotations

import logging
import re
import threading
from pathlib import Path
from typing import Any

from .config import data_dir

logger = logging.getLogger(__name__)
_lock = threading.Lock()
_cache: dict[tuple[str, str], Any] = {}


def _find_tokenizer_json(folder: Path) -> Path | None:
    if folder.is_file() and folder.name == "tokenizer.json":
        return folder
    if folder.is_dir():
        direct = folder / "tokenizer.json"
        if direct.exists():
            return direct
        for found in folder.rglob("tokenizer.json"):
            return found
    return None


def _load(settings: dict[str, Any]):
    """Returns (tokenizer or None, source description)."""
    from tokenizers import Tokenizer

    local = (settings.get("tokenizer_path") or "").strip()
    if local:
        found = _find_tokenizer_json(Path(local).expanduser())
        if found:
            return Tokenizer.from_file(str(found)), f"local folder ({found.parent})"
        logger.warning("No tokenizer.json under %s", local)

    cache = data_dir() / "tokenizer"
    cached = cache / "tokenizer.json"
    if cached.exists():
        return Tokenizer.from_file(str(cached)), "cached download"

    repo = (settings.get("tokenizer_repo") or "").strip()
    if repo:
        try:
            from huggingface_hub import hf_hub_download

            path = hf_hub_download(repo, "tokenizer.json", local_dir=str(cache))
            return Tokenizer.from_file(path), f"downloaded from {repo}"
        except Exception as exc:  # offline, blocked, or the repo moved
            logger.warning("Tokenizer download from %s failed: %s", repo, exc)
    return None, "approximate (no tokenizer available)"


def get_tokenizer(settings: dict[str, Any]):
    key = (settings.get("tokenizer_path") or "", settings.get("tokenizer_repo") or "")
    with _lock:
        if key not in _cache:
            try:
                _cache[key] = _load(settings)
            except Exception as exc:
                logger.warning("Tokenizer failed to load: %s", exc)
                _cache[key] = (None, "approximate (tokenizer failed to load)")
        return _cache[key]


def reset() -> None:
    with _lock:
        _cache.clear()


def approximate(text: str) -> int:
    # Roughly 1.3 tokens per word plus one per punctuation mark.
    words = len(re.findall(r"\w+", text))
    marks = len(re.findall(r"[^\w\s]", text))
    return round(words * 1.3 + marks)


def count(text: str, settings: dict[str, Any]) -> int:
    if not text:
        return 0
    tokenizer, _ = get_tokenizer(settings)
    if tokenizer is None:
        return approximate(text)
    return len(tokenizer.encode(text, add_special_tokens=False).ids)


def status(settings: dict[str, Any]) -> dict[str, Any]:
    tokenizer, source = get_tokenizer(settings)
    return {"exact": tokenizer is not None, "source": source}
