"""Label Maker: tag files written from the image, then checked by the model.

Two separate readers, on purpose. A tagger (WD EVA02 by default) is a small
classifier that looks at the image and gives Danbooru-style tags with a
confidence each. It runs inside this app, not on the model server. The model
on the server can then look at the same image and strike or correct tags that
are clearly wrong. Every change it makes is recorded with the image, so it can
be seen and undone.
"""

from __future__ import annotations

import asyncio
import csv
import fnmatch
import json
import logging
import threading
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image

from .client import ModelClient, ModelServerError, image_part, text_part
from .config import data_dir
from .dataset import Project
from .imaging import prepare_data_url, upright_filled
from .pipeline import build_snapshot
from .prefills import KNOWN_FACTS

logger = logging.getLogger(__name__)

DEFAULT_REPO = "SmilingWolf/wd-eva02-large-tagger-v3"
MODEL_FILE, TAGS_FILE = "model.onnx", "selected_tags.csv"
GENERAL, CHARACTER, RATING = 0, 4, 9          # the categories in a WD tagger's tag list
# Faces drawn with punctuation keep their underscores; every other tag is written with spaces.
KEEP_UNDERSCORES = {"0_0", "(o)_(o)", "+_+", "+_-", "._.", "<o>_<o>", "<|>_<|>", "=_=", ">_<", "3_3", "6_9", ">_o",
                    "@_@", "^_^", "o_o", "u_u", "x_x", "|_|", "||_||"}


class TaggerError(Exception):
    """The tagger cannot be used. The message says why, in plain words."""


def tidy(name: str) -> str:
    return name if name in KEEP_UNDERSCORES else name.replace("_", " ")


def split_tags(text: str) -> list[str]:
    return [t.strip() for t in text.replace("\n", ",").split(",") if t.strip()]


def _same(tag: str) -> str:
    """How a tag is compared: "Red_Hair" and "red hair" are the same tag."""
    return " ".join(tag.lower().replace("_", " ").split())


def tag_rules(ds: dict[str, Any]) -> tuple[list[str], list[str]]:
    """(tags that always lead the file, patterns for tags that are never written) for this dataset."""
    first = split_tags(ds.get("trigger") or "") if ds.get("tags_trigger_first", True) else []
    first += [t for t in split_tags(ds.get("tags_first") or "") if _same(t) not in {_same(f) for f in first}]
    return first, split_tags(ds.get("tags_never") or "")


def arrange(tags: list[str], first: list[str], never: list[str]) -> list[str]:
    """Apply the dataset's lists: drop the never tags, then put the first tags in front.

    A never entry may use * for "anything", so "* hair" drops every hair tag. A first tag is always
    kept, even when a never pattern would match it.
    """
    lead = {_same(t) for t in first}
    patterns = [_same(p) for p in never]
    out, seen = list(first), set(lead)
    for tag in tags:
        key = _same(tag)
        if key in seen or any(fnmatch.fnmatchcase(key, pattern) for pattern in patterns):
            continue
        seen.add(key)
        out.append(tag)
    return out


def apply_rules(project: Project) -> int:
    """Bring every existing tag file in line with the dataset's lists. Returns how many files changed."""
    first, never = tag_rules(project.store.settings())
    changed = 0
    for image in project.store.images():
        if not project.tags_path(image).exists():
            continue
        old = split_tags(project.read_tags(image))
        new = arrange(old, first, never)
        if new != old:
            project.write_tags(image, ", ".join(new), "the dataset's tag lists were applied")
            changed += 1
    return changed


class Tagger:
    """One loaded tagger: an ONNX model and the list of tags it knows."""

    def __init__(self, folder: Path, source: str, device: str = "auto"):
        try:
            import onnxruntime as ort
        except ImportError as exc:  # pragma: no cover - the dependency is installed with the app
            raise TaggerError("The tagger needs the onnxruntime package. Run `uv sync` in the app folder.") from exc
        model, tags = folder / MODEL_FILE, folder / TAGS_FILE
        if not model.is_file() or not tags.is_file():
            raise TaggerError(f"{folder} needs both {MODEL_FILE} and {TAGS_FILE}.")
        with open(tags, newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.names = [tidy(row["name"]) for row in rows]
        self.categories = [int(row["category"]) for row in rows]
        self.vocabulary = {name.lower() for name in self.names}
        wanted = ("CPUExecutionProvider",) if device == "cpu" else ("CUDAExecutionProvider", "CPUExecutionProvider")
        providers = [p for p in wanted if p in ort.get_available_providers()]
        if "CUDAExecutionProvider" in providers and hasattr(ort, "preload_dlls"):
            try:
                ort.preload_dlls()     # finds the CUDA libraries that were installed beside onnxruntime
            except Exception as exc:  # noqa: BLE001 - the card is then simply not used
                logger.warning("Could not load the CUDA libraries, so the tagger will use the CPU: %s", exc)
        try:
            # If the graphics card cannot be used, onnxruntime falls back to the next provider by itself.
            self.session = ort.InferenceSession(str(model), providers=providers)
        except Exception as exc:  # noqa: BLE001 - a damaged or half-downloaded file
            raise TaggerError(f"The tagger model in {folder} could not be loaded ({type(exc).__name__}). "
                              "Delete that folder and download it again.") from exc
        entry = self.session.get_inputs()[0]
        self.input_name = entry.name
        # WD taggers take a square picture, height x width x color, in blue-green-red order.
        self.size = int(entry.shape[1]) if isinstance(entry.shape[1], int) else 448
        self.source = source
        self.device = "GPU" if self.session.get_providers()[0].startswith("CUDA") else "CPU"
        self.wanted_gpu = device != "cpu"
        self._lock = threading.Lock()

    def _prepare(self, path: Path, background: str) -> np.ndarray:
        with Image.open(path) as opened:
            image = upright_filled(opened, background).convert("RGB")
        side = max(image.size)
        square = Image.new("RGB", (side, side), (255, 255, 255))
        square.paste(image, ((side - image.width) // 2, (side - image.height) // 2))
        if side != self.size:
            square = square.resize((self.size, self.size), Image.BICUBIC)
        return np.asarray(square, dtype=np.float32)[:, :, ::-1][None]

    def tag(self, path: Path, general: float = 0.35, character: float = 0.85, background: str = "#ffffff") -> dict[str, Any]:
        """{"tags": [...], "scores": {tag: confidence}, "rating": "..."} with characters first, then by confidence."""
        with self._lock:
            scores = self.session.run(None, {self.input_name: self._prepare(path, background)})[0][0]
        found: list[tuple[int, float, str]] = []
        rating, rating_score = "", -1.0
        for name, category, score in zip(self.names, self.categories, scores):
            score = float(score)
            if category == RATING:
                if score > rating_score:
                    rating, rating_score = name, score
            elif category == CHARACTER and score >= character:
                found.append((0, -score, name))
            elif category == GENERAL and score >= general:
                found.append((1, -score, name))
        found.sort()
        return {"tags": [name for _, _, name in found], "scores": {name: round(-s, 3) for _, s, name in found}, "rating": rating}


_loaded: dict[tuple[str, str, str], Tagger] = {}
_load_lock = threading.Lock()


def tagger_folder(settings: dict[str, Any]) -> tuple[Path, str]:
    """(folder the files are or will be in, where they come from)."""
    local = (settings.get("tagger_path") or "").strip()
    if local:
        return Path(local), f"local folder {local}"
    repo = (settings.get("tagger_repo") or DEFAULT_REPO).strip()
    return data_dir() / "tagger" / repo.replace("/", "--"), repo


def status(settings: dict[str, Any]) -> dict[str, Any]:
    """Whether the tagger is on disk, without loading or downloading anything."""
    folder, source = tagger_folder(settings)
    ready = (folder / MODEL_FILE).is_file() and (folder / TAGS_FILE).is_file()
    device = "cpu" if settings.get("tagger_device") == "cpu" else "auto"
    loaded = _loaded.get((str(folder), source, device))
    return {"ready": ready, "device_setting": device,
            # Known only once the tagger has been loaded: which of the two it is really running on.
            "device": loaded.device if loaded else None, "source": source, "folder": str(folder), "local": bool((settings.get("tagger_path") or "").strip()),
            "general": float(settings.get("tagger_general", 0.35)), "character": float(settings.get("tagger_character", 0.85))}


def get_tagger(settings: dict[str, Any]) -> Tagger:
    """Load the tagger, downloading it the first time. Slow the first time, instant after."""
    folder, source = tagger_folder(settings)
    device = "cpu" if settings.get("tagger_device") == "cpu" else "auto"
    key = (str(folder), source, device)
    with _load_lock:
        if key in _loaded:
            return _loaded[key]
        if not ((folder / MODEL_FILE).is_file() and (folder / TAGS_FILE).is_file()):
            if (settings.get("tagger_path") or "").strip():
                raise TaggerError(f"The tagger folder {folder} needs both {MODEL_FILE} and {TAGS_FILE}.")
            try:
                from huggingface_hub import hf_hub_download

                folder.mkdir(parents=True, exist_ok=True)
                for name in (TAGS_FILE, MODEL_FILE):
                    hf_hub_download(source, name, local_dir=str(folder))
            except Exception as exc:  # noqa: BLE001 - offline, blocked, or the repository moved
                logger.warning("Tagger download from %s failed: %s", source, exc)
                raise TaggerError(
                    f"The tagger could not be downloaded from {source} ({type(exc).__name__}). Check the internet "
                    f"connection, or download {MODEL_FILE} and {TAGS_FILE} yourself and set their folder as the "
                    "tagger folder in the Candy Shop tab.") from exc
        _loaded[key] = Tagger(folder, source, device)
        return _loaded[key]


# -- the model's check -----------------------------------------------------

CHECK_SYSTEM = (
    "You are checking a list of automatic tags against one image. The tags came from an image tagger, which is "
    "usually right and sometimes wrong.\n"
    "- Put a tag in \"remove\" only when the image clearly shows it is wrong: the thing is not there, or it is "
    "plainly a different color, number or kind.\n"
    "- When a wrong tag has an obvious right one of the same sort (red hair that is clearly orange hair, 1girl "
    "that is clearly 2girls), put the pair in \"replace\" instead of \"remove\". Write the new tag in the same "
    "short tag style as the others.\n"
    "- If you are not sure, keep the tag. A tag for something small, partly hidden or hard to see stays.\n"
    "- Never judge a tag by taste or wording. Do not add tags for things the list misses; that is not the job.\n"
    "Reply with one JSON object: {\"remove\": [...], \"replace\": [{\"from\": ..., \"to\": ...}]}. "
    "Both lists are empty when every tag is acceptable."
)


def check_schema(tags: list[str]) -> dict[str, Any]:
    """The reply may only name tags that are in the list, which is enforced where the server supports it."""
    one = {"type": "string", "enum": tags}
    return {
        "type": "object",
        "properties": {
            "remove": {"type": "array", "items": one},
            "replace": {"type": "array", "items": {
                "type": "object", "properties": {"from": one, "to": {"type": "string"}},
                "required": ["from", "to"], "additionalProperties": False}},
        },
        "required": ["remove", "replace"], "additionalProperties": False,
    }


def check_messages(tags: list[str], image_url: str, note: str = "") -> list[dict[str, Any]]:
    text = "Tags to check: " + ", ".join(tags)
    if note.strip():
        text += "\n\n" + KNOWN_FACTS.format(note=note.strip())
    return [{"role": "system", "content": CHECK_SYSTEM},
            {"role": "user", "content": [text_part(text), image_part(image_url)]}]


def apply_check(tags: list[str], reply: str, protected: set[str], vocabulary: set[str] | None) -> dict[str, Any]:
    """Turn the model's reply into changes, keeping only the ones that are safe to make.

    Returns {"tags", "removed", "replaced", "refused"}. A reply that would strip more than half the
    tags is refused whole: that is a model misreading the image or the task, not a tagger that bad.
    """
    out = {"tags": list(tags), "removed": [], "replaced": [], "refused": ""}
    try:
        start, end = reply.find("{"), reply.rfind("}")
        data = json.loads(reply[start:end + 1])
        remove = [str(t).strip() for t in (data.get("remove") or [])]
        replace = [(str(p.get("from", "")).strip(), tidy(str(p.get("to", "")).strip().lower()))
                   for p in (data.get("replace") or []) if isinstance(p, dict)]
    except (ValueError, AttributeError, TypeError):
        out["refused"] = "The model's reply could not be read, so no tag was changed."
        return out
    have = {t.lower(): t for t in tags}
    guarded = {t.lower() for t in protected}
    swaps: dict[str, str] = {}
    for old, new in replace:
        key = old.lower()
        if key not in have or key in guarded or not new or new == key:
            continue
        if new in have:
            remove.append(old)                    # the right tag is already there: just drop the wrong one
        elif (vocabulary is None and len(new.split()) <= 4) or (vocabulary is not None and new in vocabulary):
            swaps[key] = new                      # only real tags may come in
        else:
            remove.append(old)
    drop = {t.lower() for t in remove if t.lower() in have and t.lower() not in guarded} - set(swaps)
    if len(drop) + len(swaps) > max(1, len(tags) // 2):
        out["refused"] = (f"The model wanted to change {len(drop) + len(swaps)} of {len(tags)} tags. That is more "
                          "than half, so nothing was changed. Check this image by hand.")
        return out
    kept: list[str] = []
    for tag in tags:
        key = tag.lower()
        if key in drop:
            out["removed"].append(tag)
        elif key in swaps:
            out["replaced"].append([tag, swaps[key]])
            kept.append(swaps[key])
        else:
            kept.append(tag)
    out["tags"] = kept
    return out


async def check_tags(client: ModelClient, snapshot: dict[str, Any], image_url: str, tags: list[str], note: str,
                     protected: set[str], vocabulary: set[str] | None) -> dict[str, Any]:
    """One request with one image: which of these tags are clearly wrong?"""
    reply = await client.complete(
        snapshot["model"], check_messages(tags, image_url, note),
        sampling={**snapshot["sampling"], "temperature": 0.0}, max_tokens=120 + 12 * len(tags),
        schema=("tag_check", check_schema(tags)), think=True)
    if not reply.text:
        raise ModelServerError("reply", reply.why_empty() or "The model gave no answer about the tags.")
    return apply_check(tags, reply.text, protected, vocabulary)


def undo_change(project: Project, image: dict[str, Any], tag: str) -> None:
    """Put back a tag the model removed, or the original of one it replaced."""
    record = dict(image.get("tagging") or {})
    tags = split_tags(project.read_tags(image))
    removed = [t for t in record.get("removed") or [] if t != tag]
    replaced = []
    for old, new in record.get("replaced") or []:
        if old == tag:
            tags = [old if t == new else t for t in tags]
        else:
            replaced.append([old, new])
    if len(removed) != len(record.get("removed") or []) and tag not in tags:
        tags.append(tag)
    project.write_tags(image, ", ".join(tags), "restored a tag the model had changed")
    project.store.update_image(image["id"], tagging={**record, "removed": removed, "replaced": replaced})


class TagJob:
    """Tag the images, then let the model check the tags. Either half can run alone."""

    def __init__(self, project: Project, app_settings: dict[str, Any], publish: Callable[[dict[str, Any]], None],
                 client_factory: Callable[[dict[str, Any]], ModelClient] = ModelClient,
                 tagger_factory: Callable[[dict[str, Any]], Any] = get_tagger):
        self.project, self.app, self.publish = project, app_settings, publish
        self.client_factory, self.tagger_factory = client_factory, tagger_factory
        self.state, self.message, self.step = "idle", "", ""
        self.steps = self.step_number = self.step_done = self.step_total = 0
        self.written = self.changed = self.failed = 0
        self.note = ""
        self._stop = False
        self._task: asyncio.Task | None = None

    @property
    def busy(self) -> bool:
        return self.state == "running"

    def status(self) -> dict[str, Any]:
        return {"state": self.state, "message": self.message, "step": self.step, "steps": self.steps,
                "step_number": self.step_number, "step_done": self.step_done, "step_total": self.step_total,
                "written": self.written, "changed": self.changed, "failed": self.failed, "note": self.note}

    def _tell(self) -> None:
        self.publish({"type": "tag_progress", **self.status()})

    def start(self, scope: str = "missing", check: bool = True) -> None:
        if self.busy:
            raise RuntimeError("Tagging is already running.")
        self.state, self._stop = "running", False
        self._task = asyncio.create_task(self._run(scope, check))

    def stop(self) -> None:
        self._stop = True

    async def wait(self) -> None:
        if self._task:
            await self._task

    def _images(self) -> list[dict[str, Any]]:
        return [i for i in self.project.store.images() if not (i.get("health") or {}).get("unreadable")]

    async def _run(self, scope: str, check: bool) -> None:
        """scope: "missing" tags images with no tag file, "all" tags every image again, "none" only checks."""
        project, store = self.project, self.project.store
        ds = store.settings()
        background = ds.get("background", "#ffffff")
        general, character = float(self.app.get("tagger_general", 0.35)), float(self.app.get("tagger_character", 0.85))
        first, never = tag_rules(ds)
        images = self._images()
        to_tag = [] if scope == "none" else [i for i in images if scope == "all" or not project.tags_path(i).exists()]
        self.steps = (1 if to_tag else 0) + (1 if check else 0)
        self.step_number = 0
        vocabulary: set[str] | None = None
        try:
            if to_tag:
                self.step, self.step_done, self.step_total, self.step_number = "Loading the tagger", 0, len(to_tag), 1
                self._tell()
                tagger = await asyncio.to_thread(self.tagger_factory, self.app)
                vocabulary = getattr(tagger, "vocabulary", None)
                self.step = f"Tagging each image (on the {getattr(tagger, 'device', 'CPU')})"
                if getattr(tagger, "wanted_gpu", False) and getattr(tagger, "device", "CPU") == "CPU":
                    self.note = ("The tagger is running on the CPU because the graphics card could not be used. It still "
                                 "works, only slower. An up-to-date NVIDIA driver is what it usually needs.")
                for image in to_tag:
                    if self._stop:
                        break
                    try:
                        result = await asyncio.to_thread(tagger.tag, project.path(image), general, character, background)
                        project.write_tags(image, ", ".join(arrange(result["tags"], first, never)), "tagged again by the tagger")
                        store.update_image(image["id"], tagging={
                            "by": getattr(tagger, "source", "tagger"), "at": time.time(), "count": len(result["tags"]),
                            "rating": result.get("rating", ""), "removed": [], "replaced": [], "checked_by": "", "refused": ""})
                        self.written += 1
                    except (OSError, ValueError) as exc:
                        self.failed += 1
                        logger.warning("Tagging %s failed: %s", image["rel"], exc)
                    self.step_done += 1
                    self._tell()
            # Tag files from elsewhere follow the dataset's lists too, before the model looks at them.
            await asyncio.to_thread(apply_rules, project)
            if check and not self._stop:
                await self._check(scope, {i["id"] for i in to_tag}, vocabulary, background)
            parts = []
            if to_tag:
                parts.append(f"{self.written} tag file{'s' if self.written != 1 else ''} written")
            if check:
                parts.append(f"{self.changed} image{'s' if self.changed != 1 else ''} had a tag struck or corrected")
            if self.failed:
                parts.append(f"{self.failed} could not be done")
            self.state = "stopped" if self._stop else "done"
            self.message = ("Stopped: " if self._stop else "Labels on: ") + (", ".join(parts) if parts else "nothing needed doing") + "."
        except TaggerError as exc:
            self.state, self.message = "failed", str(exc)
        except ModelServerError as exc:
            self.state, self.message = "failed", str(exc)
        except Exception as exc:  # noqa: BLE001
            self.state, self.message = "failed", f"Unexpected error ({type(exc).__name__}: {exc})."
        finally:
            self._tell()

    async def _check(self, scope: str, just_tagged: set[int], vocabulary: set[str] | None, background: str) -> None:
        project, store = self.project, self.project.store
        snapshot = build_snapshot(project, self.app)
        if not snapshot["model"]:
            raise ModelServerError("reply", "Pick a Captioner model in the Candy Shop tab before checking tags.")
        if vocabulary is None:
            # Checking tags that came from elsewhere: use the tagger's list of real tags if it is already on disk.
            try:
                vocabulary = (await asyncio.to_thread(self.tagger_factory, self.app)).vocabulary if status(self.app)["ready"] else None
            except TaggerError:
                vocabulary = None
        todo = []
        for image in self._images():
            if not project.tags_path(image).exists():
                continue
            # Everything just tagged, and every tag file the model has not looked at yet (yours included).
            if image["id"] in just_tagged or not (image.get("tagging") or {}).get("checked_by"):
                todo.append(image)
        self.step, self.step_done, self.step_total = "The model is checking the tags", 0, len(todo)
        self.step_number += 1
        self._tell()
        if not todo:
            return
        protected = set(tag_rules(store.settings())[0]) | set(split_tags(snapshot.get("trigger") or ""))
        long_side = int(snapshot["sampling"].get("long_side", 1536))
        gate = asyncio.Semaphore(max(1, min(8, int(store.settings().get("parallel", 4)))))
        client = self.client_factory(self.app)

        async def one(image: dict[str, Any]) -> None:
            async with gate:
                if self._stop:
                    return
                tags = split_tags(project.read_tags(image))
                if tags:
                    try:
                        url = await asyncio.to_thread(prepare_data_url, project.path(image), long_side, background)
                        result = await check_tags(client, snapshot, url, tags, image.get("note") or "", protected, vocabulary)
                    except ModelServerError as exc:
                        if exc.kind == "connect":
                            raise
                        result = {"tags": tags, "removed": [], "replaced": [], "refused": str(exc)}
                        self.failed += 1
                    if result["removed"] or result["replaced"]:
                        project.write_tags(image, ", ".join(result["tags"]), "tags checked by the model")
                        self.changed += 1
                    record = dict(store.image(image["id"]).get("tagging") or {})
                    record.update(removed=result["removed"], replaced=result["replaced"], refused=result["refused"],
                                  checked_by=snapshot["model"], checked_at=time.time())
                    store.update_image(image["id"], tagging=record)
                self.step_done += 1
                self._tell()

        try:
            await asyncio.gather(*(one(image) for image in todo))
        finally:
            await client.close()
