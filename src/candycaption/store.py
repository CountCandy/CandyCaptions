"""Project store: one SQLite file per dataset, kept in the app's data folder.

The ``_nl.txt`` files in the dataset folder stay the source of truth. The
store only describes them: run snapshots, each image's fields, scores,
highlights and edit history.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

from .config import data_dir
from .prefills import DEFAULT_SAMPLING

DEFAULT_DATASET: dict[str, Any] = {
    "prefill_id": "style",
    "extra_instructions": "",
    "trigger": "",
    "preset": "Balanced",
    "sampling": dict(DEFAULT_SAMPLING),
    "parallel": 4,
    "overwrite": False,
    "field_checks": "flagged",   # off | flagged | all
    "background": "#ffffff",
    "min_area_mp": 0.5,
    "recursive": False,
    # Label Maker: tags that always lead a tag file, and tags that are never written to one.
    "tags_trigger_first": True,
    "tags_first": "",
    "tags_never": "",
}

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS images (
    id INTEGER PRIMARY KEY, rel TEXT UNIQUE, width INTEGER, height INTEGER,
    size INTEGER, mtime REAL, sha TEXT, phash TEXT, health TEXT,
    status TEXT DEFAULT 'pending', error TEXT, fields TEXT, caption TEXT,
    snapshot_id INTEGER, run_id INTEGER, signals TEXT, score REAL DEFAULT 0,
    reviewed INTEGER DEFAULT 0, seconds REAL, check_seconds REAL, raw TEXT, missing INTEGER DEFAULT 0, note TEXT, style TEXT, tagging TEXT
);
CREATE TABLE IF NOT EXISTS snapshots (id INTEGER PRIMARY KEY, created REAL, data TEXT);
CREATE TABLE IF NOT EXISTS history (
    id INTEGER PRIMARY KEY, image_id INTEGER, kind TEXT, old TEXT, reason TEXT, created REAL
);
CREATE TABLE IF NOT EXISTS testset (image_id INTEGER PRIMARY KEY, facts TEXT);
CREATE TABLE IF NOT EXISTS test_runs (id INTEGER PRIMARY KEY, created REAL, label TEXT, snapshot TEXT, results TEXT);
"""

MAX_TEST_RUNS = 8
JSON_COLUMNS = {"fields", "signals", "health", "style", "tagging"}


def natural_key(name: str) -> list:
    """Sort key that reads numbers as numbers, so "ta (2)" comes before "ta (16)"."""
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", name)]


def project_id(folder: Path) -> str:
    return hashlib.sha1(str(folder.resolve()).lower().encode("utf-8")).hexdigest()[:16]


class Store:
    def __init__(self, folder: Path):
        self.folder = folder.resolve()
        self.id = project_id(self.folder)
        root = data_dir() / "projects"
        root.mkdir(parents=True, exist_ok=True)
        self.path = root / f"{self.id}.sqlite"
        self._lock = threading.RLock()
        self._db = sqlite3.connect(self.path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        with self._lock:
            self._db.executescript(SCHEMA)
            # Stores made before per-image notes existed get the column added.
            have = {row["name"] for row in self._db.execute("PRAGMA table_info(images)")}
            for column in ("note", "style", "tagging"):
                if column not in have:
                    self._db.execute(f"ALTER TABLE images ADD COLUMN {column} TEXT")
            self._db.commit()
        self.set_meta("folder", str(self.folder))

    def close(self) -> None:
        with self._lock:
            self._db.close()

    # -- low level ----------------------------------------------------------

    def _run(self, sql: str, args: tuple = ()) -> sqlite3.Cursor:
        with self._lock:
            cursor = self._db.execute(sql, args)
            self._db.commit()
            return cursor

    def _all(self, sql: str, args: tuple = ()) -> list[sqlite3.Row]:
        with self._lock:
            return self._db.execute(sql, args).fetchall()

    def get_meta(self, key: str, default: Any = None) -> Any:
        rows = self._all("SELECT value FROM meta WHERE key=?", (key,))
        return json.loads(rows[0]["value"]) if rows else default

    def set_meta(self, key: str, value: Any) -> None:
        self._run("INSERT INTO meta(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                  (key, json.dumps(value)))

    # -- dataset settings ---------------------------------------------------

    def settings(self) -> dict[str, Any]:
        merged = copy.deepcopy(DEFAULT_DATASET)
        saved = self.get_meta("settings", {})
        for key, value in saved.items():
            if key == "sampling":
                merged["sampling"].update(value or {})
            else:
                merged[key] = value
        return merged

    def save_settings(self, settings: dict[str, Any]) -> None:
        self.set_meta("settings", settings)

    # -- images -------------------------------------------------------------

    @staticmethod
    def _image(row: sqlite3.Row) -> dict[str, Any]:
        item = dict(row)
        for key in JSON_COLUMNS:
            item[key] = json.loads(item[key]) if item.get(key) else None
        return item

    def images(self, include_missing: bool = False) -> list[dict[str, Any]]:
        where = "" if include_missing else "WHERE missing=0"
        rows = [self._image(r) for r in self._all(f"SELECT * FROM images {where}")]
        return sorted(rows, key=lambda image: natural_key(image["rel"]))

    def image(self, image_id: int) -> dict[str, Any] | None:
        rows = self._all("SELECT * FROM images WHERE id=?", (image_id,))
        return self._image(rows[0]) if rows else None

    def image_by_rel(self, rel: str) -> dict[str, Any] | None:
        rows = self._all("SELECT * FROM images WHERE rel=?", (rel,))
        return self._image(rows[0]) if rows else None

    def add_image(self, rel: str) -> int:
        self._run("INSERT OR IGNORE INTO images(rel) VALUES(?)", (rel,))
        return self._all("SELECT id FROM images WHERE rel=?", (rel,))[0]["id"]

    def update_image(self, image_id: int, **values: Any) -> None:
        if not values:
            return
        for key in JSON_COLUMNS & set(values):
            values[key] = json.dumps(values[key]) if values[key] is not None else None
        columns = ", ".join(f"{key}=?" for key in values)
        self._run(f"UPDATE images SET {columns} WHERE id=?", (*values.values(), image_id))

    def mark_missing(self, present: set[str]) -> None:
        for row in self._all("SELECT id, rel, missing FROM images"):
            gone = int(row["rel"] not in present)
            if gone != row["missing"]:
                self._run("UPDATE images SET missing=? WHERE id=?", (gone, row["id"]))

    # -- snapshots and history ----------------------------------------------

    def add_snapshot(self, data: dict[str, Any]) -> int:
        return self._run("INSERT INTO snapshots(created,data) VALUES(?,?)", (time.time(), json.dumps(data))).lastrowid

    def snapshot(self, snapshot_id: int | None) -> dict[str, Any] | None:
        if not snapshot_id:
            return None
        rows = self._all("SELECT data FROM snapshots WHERE id=?", (snapshot_id,))
        return json.loads(rows[0]["data"]) if rows else None

    def add_history(self, image_id: int, kind: str, old: str, reason: str) -> None:
        self._run("INSERT INTO history(image_id,kind,old,reason,created) VALUES(?,?,?,?,?)",
                  (image_id, kind, old, reason, time.time()))

    def history(self, image_id: int) -> list[dict[str, Any]]:
        return [dict(r) for r in self._all(
            "SELECT * FROM history WHERE image_id=? ORDER BY id DESC LIMIT 50", (image_id,))]

    # -- test bench ---------------------------------------------------------

    def testset(self) -> dict[int, list[str]]:
        return {r["image_id"]: json.loads(r["facts"] or "[]") for r in self._all("SELECT * FROM testset")}

    def pin(self, image_id: int, pinned: bool) -> None:
        if pinned:
            self._run("INSERT OR IGNORE INTO testset(image_id,facts) VALUES(?,?)", (image_id, "[]"))
        else:
            self._run("DELETE FROM testset WHERE image_id=?", (image_id,))

    def set_facts(self, image_id: int, facts: list[str]) -> None:
        self._run("UPDATE testset SET facts=? WHERE image_id=?", (json.dumps(facts), image_id))

    def add_test_run(self, label: str, snapshot: dict[str, Any], results: list[dict[str, Any]]) -> int:
        run_id = self._run("INSERT INTO test_runs(created,label,snapshot,results) VALUES(?,?,?,?)",
                           (time.time(), label, json.dumps(snapshot), json.dumps(results))).lastrowid
        # A hard limit: only the newest versions are kept.
        self._run("DELETE FROM test_runs WHERE id NOT IN (SELECT id FROM test_runs ORDER BY id DESC LIMIT ?)", (MAX_TEST_RUNS,))
        return run_id

    def delete_test_runs(self, run_id: int | None = None) -> None:
        if run_id is None:
            self._run("DELETE FROM test_runs")
        else:
            self._run("DELETE FROM test_runs WHERE id=?", (run_id,))

    def test_runs(self) -> list[dict[str, Any]]:
        out = []
        for row in self._all("SELECT * FROM test_runs ORDER BY id DESC LIMIT ?", (MAX_TEST_RUNS,)):
            out.append({"id": row["id"], "created": row["created"], "label": row["label"],
                        "snapshot": json.loads(row["snapshot"]), "results": json.loads(row["results"])})
        return out

    def test_run(self, run_id: int) -> dict[str, Any] | None:
        return next((r for r in self.test_runs() if r["id"] == run_id), None)
