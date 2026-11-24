"""Flavor Profile (Style): which images do not fit the rest of the set.

Two passes, both relative to the set itself and never to a fixed number:

* Macro: source quality measured from the pixels (sharpness, resolution,
  compression blocking, grain, black-and-white). No model, always on.
* Style lens: the model writes a short style note per image, the notes are
  summed up into one style card for the set, and each image is then checked
  against the card with a yes or no question, one image per request. Pixel
  measurements of palette and contrast give a second, model-free opinion.

Every score is a rank within this dataset. Nothing here says an image is
wrong, only that it is the least like the others.
"""

from __future__ import annotations

import asyncio
import math
import statistics
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

from . import signals as sig
from .client import ModelClient, ModelServerError, image_part, text_part
from .dataset import Project
from .imaging import measure, prepare_data_url
from .pipeline import build_snapshot

MIN_SET = 8          # below this there is no "rest of the set" to compare with

NOTE_SYSTEM = (
    "Describe only how this image is made, not what it shows: the medium (3D render, digital painting, cel shading, "
    "pencil, photo, pixel art and so on), line work, shading, color handling, level of detail, texture and any "
    "post-processing. One sentence, 30 words at most. Never mention the subject, the setting or any names. "
    "Write in American English."
)
NOTE_SCHEMA = {"type": "object", "properties": {"style": {"type": "string"}}, "required": ["style"], "additionalProperties": False}
CARD_SYSTEM = (
    "You are given style notes, one per image, all from one dataset. Write the rendering style most of the images "
    "share, in two or three sentences: medium, line work, shading, color handling, level of detail and "
    "post-processing. Ignore notes that disagree with the majority. Describe only how the images are made, never "
    "what they show. Reply with the description only, in American English."
)
MATCH_SYSTEM = "You judge whether an image is made in a given rendering style. Answer with one word: yes or no."
EXPLAIN_SYSTEM = (
    "You compare how images are made, not what they show. The first image is the one in question. The other images "
    "are typical of the set it came from. In two sentences, say how the first differs from the others in rendering "
    "style (medium, line work, shading, color handling, detail, post-processing). If it does not differ, say so."
)


# -- comparing within the set ----------------------------------------------

def robust_z(values: list[float], floor: float) -> list[float]:
    """How far each value sits from the set's median, in units that one odd value cannot stretch."""
    median = statistics.median(values)
    spread = 1.4826 * statistics.median(abs(v - median) for v in values)
    scale = max(spread, floor)
    return [(v - median) / scale for v in values]


def macro_flags(stats: list[dict[str, float]]) -> list[list[dict[str, Any]]]:
    """Per image, the ways its source quality stands apart from the rest of the set."""
    out: list[list[dict[str, Any]]] = [[] for _ in stats]
    if len(stats) < MIN_SET:
        return out
    sharp = robust_z([s["sharpness"] for s in stats], 0.12)
    size = robust_z([math.log2(max(s["megapixels"], 1e-3)) for s in stats], 0.25)
    block = robust_z([s["blocking"] for s in stats], 0.03)
    grain = robust_z([s["grain"] for s in stats], 0.08)
    typical_color = statistics.median(s["colorfulness"] for s in stats)
    for i, s in enumerate(stats):
        if sharp[i] < -3:
            out[i].append({"check": "soft", "name": "Softer", "z": round(sharp[i], 1), "detail": "much softer (blurrier) than the rest of the set"})
        if size[i] < -3:
            out[i].append({"check": "small", "name": "Lower resolution", "z": round(size[i], 1),
                           "detail": f"{s['megapixels']:.1f} megapixels, well under the rest of the set"})
        if block[i] > 3 and s["blocking"] > 1.08:
            out[i].append({"check": "blocky", "name": "More compressed", "z": round(block[i], 1),
                           "detail": "clearer JPEG block pattern than the rest of the set"})
        if grain[i] > 3.5:
            out[i].append({"check": "grainy", "name": "Grainier", "z": round(grain[i], 1), "detail": "noticeably noisier or grainier than the rest of the set"})
        if s["colorfulness"] < 4 and typical_color > 12:
            out[i].append({"check": "mono", "name": "Black and white", "z": 9.0, "detail": "black and white in a set that is in color"})
    return out


PIXEL_FEATURES = [("colorfulness", 4.0, "more colorful", "less colorful"), ("saturation", 8.0, "more saturated", "less saturated"),
                  ("brightness", 10.0, "brighter", "darker"), ("contrast", 5.0, "higher contrast", "flatter contrast"),
                  ("detail", 1.5, "busier fine detail", "smoother, less fine detail")]


def pixel_outliers(stats: list[dict[str, float]]) -> list[dict[str, Any]]:
    """How far each image's palette, light and texture sit from its nearest neighbors in the set."""
    n = len(stats)
    if n < MIN_SET:
        return [{"z": 0.0, "reasons": []} for _ in stats]
    columns = [robust_z([s[name] for s in stats], floor) for name, floor, _, _ in PIXEL_FEATURES]
    points = np.array(columns, dtype=np.float64).T / math.sqrt(len(PIXEL_FEATURES))
    hues = np.array([s["hues"] for s in stats], dtype=np.float64)
    distance = np.sqrt(((points[:, None, :] - points[None, :, :]) ** 2).sum(axis=2)
                       + (2.5 * np.abs(hues[:, None, :] - hues[None, :, :]).sum(axis=2)) ** 2)
    np.fill_diagonal(distance, np.inf)
    k = max(2, min(5, n // 5))
    nearest = np.sort(distance, axis=1)[:, :k].mean(axis=1)
    z = robust_z([float(v) for v in nearest], 0.15)
    typical_hues = np.median(hues, axis=0)
    out = []
    for i in range(n):
        reasons = [more if columns[c][i] > 0 else less for c, (_, _, more, less) in enumerate(PIXEL_FEATURES) if abs(columns[c][i]) > 2.5]
        if float(np.abs(hues[i] - typical_hues).sum()) > 0.9:
            reasons.append("a different palette")
        out.append({"z": round(z[i], 2), "reasons": reasons})
    return out


def stands_out(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The images a person should look at, worst first. An empty list means nothing stands out."""
    picked = []
    for row in rows:
        model = row.get("match")
        by_model = model is not None and model < 0.5
        by_pixels = row["pixels"]["z"] > 3.5 and bool(row["pixels"]["reasons"])
        if by_model or by_pixels:
            picked.append(row)
    picked.sort(key=lambda r: (r["match"] if r.get("match") is not None else 1.0, -r["pixels"]["z"]))
    return picked


# -- the model's part ------------------------------------------------------

def _sampling(snapshot: dict[str, Any]) -> dict[str, Any]:
    return dict(snapshot["sampling"], temperature=0.0)


async def style_note(client: ModelClient, snapshot: dict[str, Any], image_url: str) -> str:
    messages = [{"role": "system", "content": NOTE_SYSTEM}, {"role": "user", "content": [image_part(image_url)]}]
    reply = await client.complete(snapshot["model"], messages, sampling=_sampling(snapshot), max_tokens=120,
                                  schema=("style_note", NOTE_SCHEMA))
    text = reply.text.strip()
    try:
        import json

        text = str(json.loads(text).get("style", "")).strip()
    except (ValueError, AttributeError):
        text = text.strip("{}\" ")
    if not text:
        raise ModelServerError("reply", reply.why_empty() or "The model wrote no style note.")
    return " ".join(text.split())


async def style_card(client: ModelClient, snapshot: dict[str, Any], notes: list[str]) -> str:
    # A spread of at most 80 notes is plenty to see what the set has in common, and always fits the context.
    step = max(1, math.ceil(len(notes) / 80))
    listing = "\n".join(f"- {note}" for note in notes[::step])
    messages = [{"role": "system", "content": CARD_SYSTEM}, {"role": "user", "content": f"Style notes:\n{listing}"}]
    reply = await client.complete(snapshot["model"], messages, sampling=_sampling(snapshot), max_tokens=220)
    card = " ".join(reply.text.split())
    if not card:
        raise ModelServerError("reply", reply.why_empty() or "The model wrote no style card.")
    return card


async def style_match(client: ModelClient, snapshot: dict[str, Any], image_url: str, card: str) -> tuple[float, bool]:
    """Probability that the image is made in the set's style. One image per request, image first."""
    question = f"The style most images in this set share: {card}\nIs this image made in that same rendering style? Answer yes or no."
    messages = [{"role": "system", "content": MATCH_SYSTEM},
                {"role": "user", "content": [image_part(image_url), text_part(question)]}]
    reply = await client.complete(snapshot["verifier_model"], messages, sampling=_sampling(snapshot), max_tokens=4,
                                  want_logprobs=True, top_logprobs=10)
    return sig.yes_probability(reply.text, reply.tokens)


async def explain(client: ModelClient, snapshot: dict[str, Any], urls: list[str]) -> str:
    parts = [image_part(url) for url in urls]
    parts.append(text_part("How does the first image differ from the others in rendering style?"))
    reply = await client.complete(snapshot["verifier_model"], [{"role": "system", "content": EXPLAIN_SYSTEM},
                                                               {"role": "user", "content": parts}],
                                  sampling=_sampling(snapshot), max_tokens=160)
    text = " ".join(reply.text.split())
    if not text:
        raise ModelServerError("reply", reply.why_empty() or "The model gave no explanation.")
    return text


# -- putting it together ---------------------------------------------------

def usable(project: Project, chosen_only: bool = True) -> list[dict[str, Any]]:
    """Readable, measured images; only the chosen ones when the user picked a subset to check."""
    images = [i for i in project.store.images() if not (i.get("health") or {}).get("unreadable") and (i.get("health") or {}).get("stats")]
    chosen = project.store.get_meta("style_selection")
    if chosen_only and chosen:
        picked = [i for i in images if i["id"] in set(chosen)]
        if picked:
            return picked
    return images


def targets(project: Project) -> list[dict[str, Any]]:
    """Images the user marked as exactly the style they are aiming for."""
    wanted = set(project.store.get_meta("style_targets") or [])
    return [i for i in usable(project, chosen_only=False) if i["id"] in wanted]


def note_of(image: dict[str, Any]) -> str | None:
    return (image.get("style") or {}).get("note") or ((image.get("fields") or {}).get("style") if image.get("fields") else None)


def set_choice(project: Project, key: str, ids: list[int] | None) -> None:
    """Change which images are checked, or which are the targets. Either makes the old results stale."""
    known = {i["id"] for i in project.store.images()}
    project.store.set_meta(key, sorted(set(ids or []) & known) or None)
    project.store.set_meta("style", None)


def profile(project: Project) -> dict[str, Any]:
    """Everything the tab shows. The pixel passes are computed fresh; the model's part is read from the store."""
    images = usable(project)
    everything = usable(project, chosen_only=False)
    meta = project.store.get_meta("style") or {}
    aims = targets(project)
    aim_ids = {i["id"] for i in aims}
    stats = [i["health"]["stats"] for i in images]
    macro = macro_flags(stats)
    pixels = pixel_outliers(stats)
    rows = []
    for image, flags, px in zip(images, macro, pixels):
        style = image.get("style") or {}
        # A check only counts against the style card it was made with.
        current = meta.get("at") is not None and style.get("card_at") == meta.get("at")
        rows.append({
            "id": image["id"], "rel": image["rel"], "macro": flags, "pixels": px, "note": note_of(image),
            "match": style.get("match") if current else None, "measured": style.get("measured") if current else None,
            "why": style.get("why") if current else None, "target": image["id"] in aim_ids,
        })
    checked = sum(1 for r in rows if r["match"] is not None)
    return {
        "enough": len(rows) >= MIN_SET, "min_set": MIN_SET, "count": len(rows), "total": len(everything),
        "selection": project.store.get_meta("style_selection") if len(rows) < len(everything) else None,
        "targets": [{"id": i["id"], "rel": i["rel"]} for i in aims],
        "can_run": len(rows) >= MIN_SET or (bool(aims) and len(rows) >= 1),
        "card": meta.get("card"), "made": meta, "checked": checked,
        # Listed by file name, like every other list in the app.
        "macro": [r for r in rows if r["macro"]],
        "odd": stands_out(rows) if checked or len(rows) >= MIN_SET else [],
        "rows": rows,
    }


class StyleJob:
    """Notes, card, then one match check per image, with live progress."""

    def __init__(self, project: Project, app_settings: dict[str, Any], publish: Callable[[dict[str, Any]], None],
                 client_factory: Callable[[dict[str, Any]], ModelClient] = ModelClient):
        self.project, self.app, self.publish, self.client_factory = project, app_settings, publish, client_factory
        self.state, self.message, self.step = "idle", "", ""
        self.done = self.total = 0
        # Progress within the step under way, so "33 notes, then 33 checks" never reads as 66 images.
        self.steps = self.step_number = self.step_done = self.step_total = 0
        self._task: asyncio.Task | None = None

    @property
    def busy(self) -> bool:
        return self.state == "running"

    def status(self) -> dict[str, Any]:
        return {"state": self.state, "message": self.message, "step": self.step, "done": self.done, "total": self.total,
                "steps": self.steps, "step_number": self.step_number, "step_done": self.step_done, "step_total": self.step_total}

    def _tell(self) -> None:
        self.publish({"type": "style_progress", **self.status()})

    def start(self, redo: bool = False) -> None:
        if self.busy:
            raise RuntimeError("The style check is already running.")
        self._task = asyncio.create_task(self._run(redo))

    async def wait(self) -> None:
        if self._task:
            await self._task

    async def _each(self, images: list[dict[str, Any]], step: str, work) -> None:
        """Run `work` over the images, a few at a time, counting progress."""
        if not images:
            return
        self.step, self.step_done, self.step_total = step, 0, len(images)
        self.step_number += 1
        self._tell()
        parallel = max(1, min(8, int(self.project.store.settings().get("parallel", 4))))
        gate = asyncio.Semaphore(parallel)

        async def one(image: dict[str, Any]) -> None:
            async with gate:
                await work(image)
                self.done += 1
                self.step_done += 1
                self._tell()

        await asyncio.gather(*(one(image) for image in images))

    async def _run(self, redo: bool) -> None:
        self.state, self.message, self.done = "running", "", 0
        self.step_number = self.step_done = self.step_total = 0
        store = self.project.store
        snapshot = build_snapshot(self.project, self.app)
        long_side = int(snapshot["sampling"].get("long_side", 1536))
        images = usable(self.project)
        aims = targets(self.project)
        # With targets, the style is written from their notes alone, so only they need one.
        # Without, every image needs a note, because the style is what most of them share.
        noted = {i["id"]: i for i in (aims or images)}
        need_note = [i for i in noted.values() if redo or not note_of(i)]
        self.total = len(need_note) + len(images)
        self.steps = 2 if need_note else 1
        self._tell()
        client = self.client_factory(self.app)

        def url(image: dict[str, Any]) -> str:
            return prepare_data_url(self.project.path(image), long_side, snapshot.get("background", "#ffffff"))

        try:
            async def write_note(image: dict[str, Any]) -> None:
                note = await style_note(client, snapshot, await asyncio.to_thread(url, image))
                store.update_image(image["id"], style={**(image.get("style") or {}), "note": note})

            await self._each(need_note, "Writing a style note for each image", write_note)

            images = usable(self.project)
            aims = targets(self.project)
            # With targets, the style is theirs alone. Without, it is what most of the set shares.
            notes = [note_of(i) for i in (aims or images)]
            self.step = "Summing up the target style" if aims else "Summing up the set's style"
            self._tell()
            card = await style_card(client, snapshot, [n for n in notes if n])
            made = {"card": card, "at": time.time(), "model": snapshot["model"], "verifier_model": snapshot["verifier_model"],
                    "count": len(images), "targets": len(aims)}
            store.set_meta("style", made)

            async def check(image: dict[str, Any]) -> None:
                match, measured = await style_match(client, snapshot, await asyncio.to_thread(url, image), card)
                style = {**(store.image(image["id"]).get("style") or {}), "match": round(match, 4), "measured": measured,
                         "card_at": made["at"]}
                style.pop("why", None)      # an explanation written against an older card no longer applies
                store.update_image(image["id"], style=style)

            await self._each(images, "Checking each image against the target style" if aims else "Checking each image against the set's style", check)
            odd = len(stands_out(profile(self.project)["rows"]))
            self.state = "done"
            self.message = (f"Batch tasted: {odd} piece{'s' if odd != 1 else ''} taste{'s' if odd == 1 else ''} different."
                            if odd else "Batch tasted: nothing stands out.")
        except ModelServerError as exc:
            self.state, self.message = "failed", str(exc)
        except Exception as exc:  # noqa: BLE001
            self.state, self.message = "failed", f"Unexpected error ({type(exc).__name__}: {exc})."
        finally:
            await client.close()
            self._tell()


async def explain_image(project: Project, app_settings: dict[str, Any], image: dict[str, Any], client: ModelClient) -> str:
    """Show the image next to three typical ones and ask how it differs. Four images in one request."""
    snapshot = build_snapshot(project, app_settings)
    aims = [i for i in targets(project) if i["id"] != image["id"]]
    if aims:
        typical = aims[:3]
    else:
        rows = [r for r in profile(project)["rows"] if r["id"] != image["id"]]
        rows.sort(key=lambda r: (-(r["match"] if r["match"] is not None else 0.5), r["pixels"]["z"]))
        typical = [project.store.image(r["id"]) for r in rows[:3]]
    if not typical:
        raise ModelServerError("reply", "There are no other images to compare with.")
    # Smaller copies: four images share one request, and this is about overall look, not small detail.
    urls = [await asyncio.to_thread(prepare_data_url, project.path(i), 896, snapshot.get("background", "#ffffff"))
            for i in [image] + typical]
    why = await explain(client, snapshot, urls)
    project.store.update_image(image["id"], style={**(image.get("style") or {}), "why": why})
    return why


async def ringer(project: Project, app_settings: dict[str, Any], path: Path, client: ModelClient) -> dict[str, Any]:
    """Where would an image you know does not belong rank? It is measured but never added to the dataset."""
    snapshot = build_snapshot(project, app_settings)
    background = snapshot.get("background", "#ffffff")
    stats = await asyncio.to_thread(measure, path, background)
    current = profile(project)
    images = usable(project)
    all_stats = [i["health"]["stats"] for i in images] + [stats]
    px = pixel_outliers(all_stats)
    macro = macro_flags(all_stats)[-1]
    pixel_rank = 1 + sum(1 for other in px[:-1] if other["z"] > px[-1]["z"])
    result: dict[str, Any] = {"count": len(images), "pixels": px[-1], "pixel_rank": pixel_rank, "macro": macro,
                              "match": None, "model_rank": None}
    if current["card"] and current["checked"]:
        url = await asyncio.to_thread(prepare_data_url, path, int(snapshot["sampling"].get("long_side", 1536)), background)
        match, measured = await style_match(client, snapshot, url, current["card"])
        result.update(match=round(match, 4), measured=measured,
                      model_rank=1 + sum(1 for r in current["rows"] if r["match"] is not None and r["match"] < match))
    return result
