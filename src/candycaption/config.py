"""The app's own data folder and its JSON settings files.

Everything the app remembers lives under one folder (``data/`` next to the
launcher by default, or ``CANDYCAPTION_DATA``), never inside a dataset, so the
trainer only ever sees images, tags and descriptions.
"""

from __future__ import annotations

import copy
import json
import os
import tempfile
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
STATIC_DIR = REPO_ROOT / "static"


def data_dir() -> Path:
    root = os.environ.get("CANDYCAPTION_DATA")
    path = Path(root) if root else REPO_ROOT / "data"
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_json(name: str, default: Any) -> Any:
    path = data_dir() / name
    if not path.exists():
        return copy.deepcopy(default)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return copy.deepcopy(default)


def save_json(name: str, value: Any) -> None:
    path = data_dir() / name
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


# Score weights are a starting point (design doc, open question "Score weights").
DEFAULT_SETTINGS: dict[str, Any] = {
    "backend": "lmstudio",
    "base_url": "http://localhost:1234/v1",
    "api_key": "",
    "model": "",
    "verifier_model": "",
    # What the connected server was seen to support. Filled by Test connection.
    "caps": {},
    "request_timeout": 300,
    "tokenizer_path": "",
    "tokenizer_repo": "Qwen/Qwen3.5-27B",
    "token_limit": 512,
    "score_weights": {"uncertain": 0.3, "checks": 0.4, "tags": 0.2, "template": 0.1},
    "uncertain_min_alt": 0.15,
    "check_threshold": 0.6,
    "recent": [],
}


def load_settings() -> dict[str, Any]:
    settings = copy.deepcopy(DEFAULT_SETTINGS)
    saved = load_json("settings.json", {})
    for key, value in saved.items():
        if isinstance(value, dict) and isinstance(settings.get(key), dict) and key != "caps":
            settings[key].update(value)
        else:
            settings[key] = value
    return settings


def save_settings(settings: dict[str, Any]) -> None:
    save_json("settings.json", settings)
