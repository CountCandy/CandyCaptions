"""The app server: everything the browser page talks to.

The browser never talks to the model server directly. Every model request
goes through here and holds exactly one image.
"""

from __future__ import annotations

import asyncio
import copy
import io
import json
import logging
import os
import re
import string
import time
from pathlib import Path
from typing import Any

from fastapi import Body, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from . import tokens
from .backends import BACKENDS, effective_caps
from .client import ModelClient, ModelServerError, normalize_base_url
from .config import STATIC_DIR, data_dir, load_settings, save_settings
from .connect import test_connection
from .dataset import HEALTH_CHECKS, Project
from .filters import FilterError, compile_filter
from .imaging import prepare_bytes, tiny_test_image, upright_filled
from .pipeline import CaptionError, Captioner, build_snapshot, refresh_template
from .prefills import (BASE_FIELDS, DEFAULT_SAMPLING, default_prefill, get_prefill, load_prefills, load_presets,
                       save_prefills, save_presets)
from .runner import Broadcaster, Runner
from .signals import combine
from .store import natural_key
from .testbench import promote, request_preview, run_bench, summarize
from .wordlists import DEFAULT_WORDLISTS, load_wordlists, save_wordlists

logger = logging.getLogger(__name__)
app = FastAPI(title="CandyCaption")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


class State:
    project: Project | None = None
    runner: Runner | None = None
    broadcaster = Broadcaster()
    base_url = "http://127.0.0.1:8200"
    bench_busy = False


state = State()


@app.middleware("http")
async def remember_address(request: Request, call_next):
    # The app's own address, for servers that need images passed by URL.
    state.base_url = str(request.base_url).rstrip("/")
    if state.runner:
        state.runner.image_base_url = state.base_url
    return await call_next(request)


def fail(status: int, message: str) -> HTTPException:
    return HTTPException(status_code=status, detail=message)


@app.exception_handler(ModelServerError)
async def model_error(_: Request, exc: ModelServerError):
    return JSONResponse({"detail": str(exc)}, status_code=502)


@app.exception_handler(CaptionError)
async def caption_error(_: Request, exc: CaptionError):
    return JSONResponse({"detail": str(exc)}, status_code=422)


def project() -> Project:
    if state.project is None:
        raise fail(409, "No dataset is open. Point me at a folder of images in the Pantry tab.")
    return state.project


def image_or_404(image_id: int) -> dict[str, Any]:
    image = project().store.image(image_id)
    if not image:
        raise fail(404, "That image is not in this dataset.")
    return image


def field_names() -> dict[str, str]:
    names = {key: name for key, name, _ in BASE_FIELDS}
    if state.project:
        prefill = get_prefill(state.project.store.settings()["prefill_id"])
        names.update({f["key"]: f["name"] for f in prefill["fields"]})
    return names


def captioner(client: ModelClient, settings: dict[str, Any]) -> Captioner:
    return Captioner(project(), client, settings, build_snapshot(project(), settings), load_wordlists(), state.base_url)


# -- page ------------------------------------------------------------------

@app.get("/")
async def index():
    return FileResponse(STATIC_DIR / "index.html", headers={"Cache-Control": "no-cache"})


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/api/events")
async def events(request: Request):
    queue = state.broadcaster.subscribe()

    async def stream():
        try:
            if state.runner:
                yield f"data: {json.dumps({'type': 'state', **state.runner.status()})}\n\n"
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=15)
                    yield f"data: {json.dumps(event)}\n\n"
                except asyncio.TimeoutError:
                    yield ": keep-alive\n\n"
                if await request.is_disconnected():
                    break
        finally:
            state.broadcaster.unsubscribe(queue)

    return StreamingResponse(stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})


# -- connect ---------------------------------------------------------------

def settings_payload(settings: dict[str, Any]) -> dict[str, Any]:
    return {"settings": settings, "backends": BACKENDS, "caps": effective_caps(settings),
            "default_sampling": DEFAULT_SAMPLING}


@app.get("/api/settings")
async def get_settings():
    return settings_payload(load_settings())


@app.put("/api/settings")
async def put_settings(body: dict = Body(...)):
    settings = load_settings()
    connection_changed = any(body.get(k, settings.get(k)) != settings.get(k) for k in ("backend", "base_url"))
    tokenizer_changed = any(body.get(k, settings.get(k)) != settings.get(k) for k in ("tokenizer_path", "tokenizer_repo"))
    for key, value in body.items():
        if key in settings and key != "caps":
            settings[key] = value
    settings["base_url"] = normalize_base_url(settings["base_url"])
    if connection_changed:
        settings["caps"] = {}  # what one server supports says nothing about another
    save_settings(settings)
    if tokenizer_changed:
        tokens.reset()
        if state.project:
            state.project._token_cache.clear()
    return settings_payload(settings)


@app.post("/api/connect/models")
async def connect_models():
    client = ModelClient(load_settings())
    try:
        return {"models": await client.list_models()}
    finally:
        await client.close()


@app.get("/api/probe-image")
async def probe_image():
    import base64
    return Response(base64.b64decode(tiny_test_image().split(",", 1)[1]), media_type="image/jpeg")


@app.post("/api/connect/test")
async def connect_test():
    settings = load_settings()
    result = await test_connection(settings, f"{state.base_url}/api/probe-image")
    if result["ok"]:
        settings["caps"] = result["caps"]
        if not settings.get("model") and result["models"]:
            settings["model"] = result["models"][0]
        save_settings(settings)
    return {**result, **settings_payload(settings)}


@app.get("/api/tokenizer")
async def tokenizer_status():
    return await asyncio.to_thread(tokens.status, load_settings())


# -- prefills, presets, word lists -----------------------------------------

@app.get("/api/prefills")
async def get_prefills():
    return {"prefills": load_prefills(), "presets": load_presets(), "wordlists": load_wordlists()}


@app.put("/api/prefills/{prefill_id}")
async def put_prefill(prefill_id: str, body: dict = Body(...)):
    keys = [f.get("key") for f in body.get("fields") or []]
    if not keys or any(not re.fullmatch(r"[a-z][a-z0-9_]*", k or "") for k in keys) or len(set(keys)) != len(keys):
        raise fail(422, "Every field needs its own key made of lowercase letters, digits or underscores.")
    prefills = load_prefills()
    body["id"] = prefill_id
    for index, prefill in enumerate(prefills):
        if prefill["id"] == prefill_id:
            body["builtin"] = prefill.get("builtin", False)
            prefills[index] = body
            break
    else:
        prefills.append(body)
    save_prefills(prefills)
    return {"prefills": prefills}


@app.post("/api/prefills/{prefill_id}/duplicate")
async def duplicate_prefill(prefill_id: str, body: dict = Body(...)):
    prefills = load_prefills()
    source = copy.deepcopy(get_prefill(prefill_id, prefills))
    name = (body.get("name") or f"{source['name']} copy").strip()
    new_id = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "variant"
    while any(p["id"] == new_id for p in prefills):
        new_id += "-2"
    source.update(id=new_id, name=name, builtin=False)
    prefills.append(source)
    save_prefills(prefills)
    return {"prefills": prefills, "id": new_id}


@app.post("/api/prefills/{prefill_id}/reset")
async def reset_prefill(prefill_id: str, body: dict = Body(default={})):
    default = default_prefill(prefill_id)
    if default is None:
        raise fail(404, "Only the Style, Character and Concept prefills have defaults to reset to.")
    current = get_prefill(prefill_id)
    if body.get("keep_lengths", True):
        # A reset is for getting the latest wording; the word range and preset you chose stay.
        for key in ("min_words", "max_words", "preset"):
            default[key] = current.get(key, default[key])
    prefills = [default if p["id"] == prefill_id else p for p in load_prefills()]
    save_prefills(prefills)
    return {"prefills": prefills}


@app.delete("/api/prefills/{prefill_id}")
async def delete_prefill(prefill_id: str):
    prefills = load_prefills()
    if get_prefill(prefill_id, prefills).get("builtin"):
        raise fail(409, "The built-in prefills can be reset but not deleted.")
    prefills = [p for p in prefills if p["id"] != prefill_id]
    save_prefills(prefills)
    return {"prefills": prefills}


@app.put("/api/presets")
async def put_presets(body: dict = Body(...)):
    save_presets(body.get("presets") or [])
    return {"presets": load_presets()}


@app.put("/api/wordlists")
async def put_wordlists(body: dict = Body(...)):
    if body.get("reset"):
        save_wordlists(copy.deepcopy(DEFAULT_WORDLISTS))
    else:
        lists = body.get("wordlists")
        if not isinstance(lists, dict) or "categories" not in lists or "tag_map" not in lists:
            raise fail(422, "Word lists need a \"categories\" part and a \"tag_map\" part.")
        save_wordlists(lists)
    return {"wordlists": load_wordlists()}


@app.get("/api/share/export")
async def share_export():
    payload = {"candycaption": 1, "prefills": load_prefills(), "presets": [p for p in load_presets() if not p.get("builtin")],
               "wordlists": load_wordlists()}
    return Response(json.dumps(payload, indent=2, ensure_ascii=False), media_type="application/json",
                    headers={"Content-Disposition": 'attachment; filename="candycaption-recipes.json"'})


@app.post("/api/share/import")
async def share_import(body: dict = Body(...)):
    if "prefills" not in body and "presets" not in body and "wordlists" not in body:
        raise fail(422, "That file has no prefills, presets or word lists in it.")
    if body.get("prefills"):
        merged = {p["id"]: p for p in load_prefills()}
        for prefill in body["prefills"]:
            if isinstance(prefill, dict) and prefill.get("id") and prefill.get("fields"):
                prefill["builtin"] = merged.get(prefill["id"], {}).get("builtin", False)
                merged[prefill["id"]] = prefill
        save_prefills(list(merged.values()))
    if body.get("presets"):
        names = {p["name"] for p in body["presets"]}
        save_presets([p for p in load_presets() if p["name"] not in names] + body["presets"])
    if isinstance(body.get("wordlists"), dict) and "categories" in body["wordlists"]:
        save_wordlists(body["wordlists"])
    return await get_prefills()


# -- dataset ---------------------------------------------------------------

def dataset_payload() -> dict[str, Any]:
    if state.project is None:
        return {"open": False, "recent": load_settings().get("recent", [])}
    proj, settings = state.project, load_settings()
    images, summary = [], {key: 0 for key in HEALTH_CHECKS}
    pins = proj.store.testset()
    all_images = proj.store.images()
    proj.refresh_expectations(all_images)
    for image in all_images:
        badges = proj.health(image, settings)
        for check in {b["check"] for b in badges}:
            summary[check] += 1
        images.append({
            "id": image["id"], "rel": image["rel"], "width": image["width"], "height": image["height"],
            "status": image["status"], "has_caption": proj.has_caption(image), "has_tags": proj.tags_path(image).exists(),
            "health": badges, "pinned": image["id"] in pins, "reviewed": bool(image["reviewed"]),
        })
    return {
        "open": True, "folder": str(proj.folder), "settings": proj.store.settings(), "images": images,
        "health_summary": summary, "health_checks": {k: {"name": v[0], "candy": v[1], "why": v[2]} for k, v in HEALTH_CHECKS.items()},
        "tokenizer": tokens.status(settings), "token_limit": settings.get("token_limit", 512),
        "captioned": sum(1 for i in images if i["has_caption"]), "recent": settings.get("recent", []),
    }


@app.get("/api/dataset")
async def get_dataset():
    return await asyncio.to_thread(dataset_payload)


@app.post("/api/dataset/open")
async def open_dataset(body: dict = Body(...)):
    if state.runner and state.runner.busy:
        raise fail(409, "A run is going. Stop it before opening another dataset.")
    folder = (body.get("path") or "").strip().strip('"')
    if not folder:
        raise fail(422, "Type or paste the path of a folder of images.")
    try:
        proj = Project(folder)
    except FileNotFoundError as exc:
        raise fail(404, str(exc))
    await asyncio.to_thread(proj.scan)
    if state.project:
        state.project.store.close()
    state.project = proj
    state.runner = Runner(proj, load_settings(), publish=state.broadcaster.publish, image_base_url=state.base_url)
    settings = load_settings()
    settings["recent"] = ([str(proj.folder)] + [r for r in settings.get("recent", []) if r != str(proj.folder)])[:8]
    save_settings(settings)
    return await asyncio.to_thread(dataset_payload)


@app.post("/api/dataset/rescan")
async def rescan_dataset():
    await asyncio.to_thread(project().scan)
    return await asyncio.to_thread(dataset_payload)


@app.put("/api/dataset/settings")
async def put_dataset_settings(body: dict = Body(...)):
    proj = project()
    settings = proj.store.settings()
    rescan = body.get("recursive", settings["recursive"]) != settings["recursive"]
    for key, value in body.items():
        if key == "sampling" and isinstance(value, dict):
            settings["sampling"].update(value)
        elif key in settings:
            settings[key] = value
    settings["parallel"] = max(1, min(8, int(settings["parallel"])))
    proj.store.save_settings(settings)
    if rescan:
        await asyncio.to_thread(proj.scan)
    return {"settings": settings}


@app.post("/api/dataset/fix")
async def fix_images(body: dict = Body(...)):
    proj, settings = project(), load_settings()
    kind = body.get("kind")
    if kind not in ("sideways", "transparency"):
        raise fail(422, "Only sideways and transparency have a one-click fix.")
    if body.get("id") is not None:
        targets = [image_or_404(int(body["id"]))]
    else:
        targets = [i for i in proj.store.images() if any(b["check"] == kind for b in proj.health(i, settings))]
    fixed, errors = 0, []
    for image in targets:
        try:
            await asyncio.to_thread(proj.fix, image, kind)
            fixed += 1
        except Exception as exc:
            errors.append(f"{image['rel']}: {exc}")
    _clear_thumbs(proj)
    return {"fixed": fixed, "errors": errors, "backup": str(data_dir() / "backups" / proj.id)}


@app.get("/api/browse")
async def browse(path: str = "", roots: bool = False):
    """Folders under a path, for the folder picker."""
    start = load_settings().get("start_folder") or ""
    if not path and not roots and start and Path(start).is_dir():
        path = start
    if not path:
        places = [str(Path.home())]
        if os.name == "nt":
            places += [f"{letter}:\\" for letter in string.ascii_uppercase if Path(f"{letter}:\\").exists()]
        else:
            places.append("/")
        return {"path": "", "parent": None, "start_folder": start, "folders": [{"name": r, "path": r} for r in places]}
    here = Path(path).expanduser()
    if not here.is_dir():
        raise fail(404, f"{here} is not a folder.")
    folders = []
    try:
        for child in sorted(here.iterdir(), key=lambda p: natural_key(p.name)):
            if child.is_dir() and not child.name.startswith((".", "$")):
                folders.append({"name": child.name, "path": str(child)})
    except PermissionError:
        pass
    parent = str(here.parent) if here.parent != here else ""
    return {"path": str(here), "parent": parent, "start_folder": start, "folders": folders}


def _thumb_dir(proj: Project) -> Path:
    path = data_dir() / "thumbs" / proj.id
    path.mkdir(parents=True, exist_ok=True)
    return path


def _clear_thumbs(proj: Project) -> None:
    for thumb in _thumb_dir(proj).glob("*.jpg"):
        thumb.unlink(missing_ok=True)


def _thumb(proj: Project, image: dict[str, Any]) -> bytes:
    cached = _thumb_dir(proj) / f"{image['id']}-{int(image['mtime'] or 0)}.jpg"
    if cached.exists():
        return cached.read_bytes()
    with Image.open(proj.path(image)) as opened:
        img = upright_filled(opened, proj.store.settings().get("background", "#ffffff"))
    img.thumbnail((360, 360), Image.LANCZOS)
    buffer = io.BytesIO()
    img.save(buffer, "JPEG", quality=85)
    cached.write_bytes(buffer.getvalue())
    return buffer.getvalue()


@app.get("/api/image/{image_id}")
async def get_image(image_id: int, size: str = "thumb"):
    proj = project()
    image = image_or_404(image_id)
    try:
        if size == "full":
            return FileResponse(proj.path(image))
        if size == "prepared":
            ds = proj.store.settings()
            data = await asyncio.to_thread(prepare_bytes, proj.path(image), int(ds["sampling"].get("long_side", 1536)), ds["background"])
        else:
            data = await asyncio.to_thread(_thumb, proj, image)
    except Exception:
        raise fail(415, "This image cannot be read.")
    return Response(data, media_type="image/jpeg", headers={"Cache-Control": "max-age=60"})


# -- run -------------------------------------------------------------------

def runner() -> Runner:
    project()
    assert state.runner is not None
    return state.runner


@app.get("/api/run")
async def run_status():
    if state.runner is None:
        return {"state": "idle", "total": 0, "done": 0, "failed": 0, "skipped": 0, "in_flight": [], "recent": [], "message": ""}
    return state.runner.status()


@app.post("/api/run/start")
async def run_start(body: dict = Body(default={})):
    run = runner()
    settings = load_settings()
    if not settings.get("model"):
        raise fail(409, "No model is chosen. Pick one in the Candy Shop tab first.")
    if run.busy or state.bench_busy:
        raise fail(409, "A run is already going.")
    run.app = settings
    ids = body.get("image_ids")
    run.start(mode=body.get("mode", "caption"), image_ids=set(ids) if ids else None,
              extra=body.get("extra") or None, resume=bool(body.get("resume")))
    await asyncio.sleep(0)
    return run.status()


@app.post("/api/run/{action}")
async def run_control(action: str):
    run = runner()
    if action == "pause":
        run.pause()
    elif action == "resume":
        run.resume()
    elif action == "stop":
        run.stop()
    elif action == "forget":
        project().store.set_meta("open_run", None)
    else:
        raise fail(404, "Unknown run control.")
    return run.status()


# -- review ----------------------------------------------------------------

def review_row(proj: Project, image: dict[str, Any], settings: dict[str, Any], names: dict[str, str],
               detail: bool = False) -> dict[str, Any]:
    caption, tags = proj.read_caption(image), proj.read_tags(image)
    score, reasons = combine(image["signals"] or {}, settings.get("score_weights") or {})
    for reason in reasons:
        reason["field_name"] = names.get(reason["field"]) if reason["field"] else None
    row = {
        "id": image["id"], "rel": image["rel"], "caption": caption, "tags": tags, "score": score, "reasons": reasons,
        "reviewed": bool(image["reviewed"]), "status": image["status"], "error": image["error"],
        "words": len(caption.split()), "tokens": proj.token_count(image, settings, tags, caption),
        "fields": image["fields"], "note": image.get("note") or "",
    }
    if detail:
        snapshot = proj.store.snapshot(image["snapshot_id"])
        prefill = (snapshot or {}).get("prefill") or get_prefill(proj.store.settings()["prefill_id"])
        row.update(
            field_defs=prefill["fields"], signals=image["signals"], history=proj.store.history(image["id"]),
            health=proj.health(image, settings), width=image["width"], height=image["height"],
            snapshot={"prefill": prefill["name"], "model": (snapshot or {}).get("model"), "preset": (snapshot or {}).get("preset"),
                      "extra_instructions": (snapshot or {}).get("extra_instructions", "")} if snapshot else None,
            token_limit=settings.get("token_limit", 512), tokenizer=tokens.status(settings),
        )
    return row


@app.get("/api/review")
async def review_list(q: str = "", sort: str = "name"):
    proj, settings, names = project(), load_settings(), field_names()
    try:
        matches = compile_filter(q)
    except FilterError as exc:
        raise fail(422, str(exc))

    def build() -> dict[str, Any]:
        rows, totals = [], {"done": 0, "reviewed": 0, "flagged": 0, "failed": 0}
        show_reviewed = "reviewed:" in q.lower()
        show_all = "status:" in q.lower()
        for image in proj.store.images():
            has_caption = proj.has_caption(image)
            if image["status"] == "failed" and has_caption:
                # A description exists on disk, and the files are the source of truth.
                proj.store.update_image(image["id"], status="done", error=None)
                image.update(status="done", error=None)
            if image["status"] == "failed":
                totals["failed"] += 1
            if not has_caption and not (show_all or image["status"] == "failed"):
                continue
            row = review_row(proj, image, settings, names)
            if has_caption:
                totals["done"] += 1
                totals["reviewed"] += row["reviewed"]
                totals["flagged"] += bool(row["reasons"]) and not row["reviewed"]
            try:
                if not matches(row):
                    continue
            except FilterError:
                continue
            if row["reviewed"] and not show_reviewed:
                continue
            row.pop("fields")
            row.pop("tags")
            rows.append(row)
        if sort == "name":
            rows.sort(key=lambda r: natural_key(r["rel"]))
        else:
            rows.sort(key=lambda r: (r["status"] != "failed", -r["score"], natural_key(r["rel"])))
        return {"rows": rows, "totals": totals}

    return await asyncio.to_thread(build)


@app.get("/api/review/{image_id}")
async def review_detail(image_id: int):
    return review_row(project(), image_or_404(image_id), load_settings(), field_names(), detail=True)


def _rescore(proj: Project, image: dict[str, Any], settings: dict[str, Any], signals: dict[str, Any], **extra: Any) -> None:
    proj.store.update_image(image["id"], signals=signals, score=combine(signals, settings.get("score_weights") or {})[0], **extra)


@app.put("/api/review/{image_id}/caption")
async def edit_caption(image_id: int, body: dict = Body(...)):
    proj = project()
    image = image_or_404(image_id)
    text = " ".join((body.get("text") or "").split())
    if not text:
        raise fail(422, "A caption cannot be empty. Delete the file by hand if you really want none.")
    proj.write_caption(image, text, "edited by hand")
    if image["status"] != "done":
        proj.store.update_image(image_id, status="done", error=None)
    return await review_detail(image_id)


@app.put("/api/review/{image_id}/tags")
async def edit_tags(image_id: int, body: dict = Body(...)):
    proj, settings = project(), load_settings()
    image = image_or_404(image_id)
    proj.write_tags(image, body.get("text") or "", "edited by hand")
    if image["fields"]:
        cap = Captioner(proj, None, settings, build_snapshot(proj, settings), load_wordlists())  # type: ignore[arg-type]
        _rescore(proj, image, settings, cap.signals_for(image, image["fields"], None, None, image["signals"]))
    return await review_detail(image_id)


@app.put("/api/review/{image_id}/fields")
async def edit_fields(image_id: int, body: dict = Body(...)):
    """Editing a field recomposes the description."""
    proj, settings = project(), load_settings()
    image = image_or_404(image_id)
    old = image["fields"] or {}
    fields = {**old, **{k: str(v).strip() or "none" for k, v in (body.get("fields") or {}).items()}}
    changed = {k for k in fields if fields[k] != old.get(k)}
    if not changed:
        return await review_detail(image_id)
    signals = image["signals"] or {}
    checks = signals.get("checks") or {}
    if checks.get("values"):
        # A check answered a claim that no longer stands.
        checks["values"] = {k: v for k, v in checks["values"].items() if k not in changed}
        checks["items"] = [i for i in checks.get("items", []) if i["field"] not in changed]
    image["signals"] = signals
    proj.store.add_history(image_id, "fields", json.dumps(old, ensure_ascii=False), "edited by hand")
    client = ModelClient(settings)
    try:
        await captioner(client, settings).recompose(image, fields, reason="field edited by hand")
    finally:
        await client.close()
    return await review_detail(image_id)


@app.post("/api/review/{image_id}/approve")
async def approve(image_id: int, body: dict = Body(default={})):
    image_or_404(image_id)
    project().store.update_image(image_id, reviewed=int(body.get("reviewed", True)))
    return {"ok": True}


@app.post("/api/review/{image_id}/regenerate")
async def regenerate(image_id: int, body: dict = Body(default={})):
    run = runner()
    image_or_404(image_id)
    if run.busy or state.bench_busy:
        raise fail(409, "A run is going. Regenerate once it has finished or paused.")
    run.app = load_settings()
    # The note is kept with the image; sending an empty one clears it.
    note = " ".join((body.get("extra") or "").split())
    project().store.update_image(image_id, note=note)
    run.start(image_ids={image_id})
    await run.wait()
    image = image_or_404(image_id)
    if image["status"] == "failed":
        raise fail(422, image["error"] or "The image could not be captioned.")
    if run.state == "paused":
        raise fail(502, run.message)
    return await review_detail(image_id)


@app.post("/api/review/{image_id}/chat")
async def review_chat(image_id: int, body: dict = Body(...)):
    """One turn of review chat. The conversation lives in the browser and is gone when the chat closes."""
    settings = load_settings()
    image = image_or_404(image_id)
    if not image["fields"]:
        raise fail(409, "This image has no stored fields yet. Regenerate it first.")
    client = ModelClient(settings)
    try:
        return await captioner(client, settings).chat(image, image["fields"], project().read_caption(image), body.get("messages") or [])
    finally:
        await client.close()


@app.post("/api/review/{image_id}/chat/accept")
async def chat_accept(image_id: int, body: dict = Body(...)):
    """Save the corrected fields and caption, then rerun this image's field checks."""
    proj, settings = project(), load_settings()
    image = image_or_404(image_id)
    fields, caption = body.get("fields"), " ".join((body.get("caption") or "").split())
    if not isinstance(fields, dict) or not caption:
        raise fail(422, "There is no proposed correction to accept.")
    proj.store.add_history(image_id, "fields", json.dumps(image["fields"] or {}, ensure_ascii=False), "review chat")
    proj.write_caption(image, caption, "review chat")
    client = ModelClient(settings)
    try:
        cap = captioner(client, settings)
        checks = None
        if proj.store.settings().get("field_checks") != "off":
            try:
                checks = await cap.check(await asyncio.to_thread(cap.image_url, image), fields)
            except ModelServerError:
                checks = None
        signals = cap.signals_for(image, fields, None, checks, image["signals"])
        if checks is None:
            signals["checks"] = {"available": False, "items": [], "values": {}}
        _rescore(proj, image, settings, signals, fields=fields, caption=caption)
    finally:
        await client.close()
    return await review_detail(image_id)


@app.post("/api/rules/promote")
async def promote_rule(body: dict = Body(...)):
    """Turn a correction into a dataset rule. This is the only way information moves between images."""
    proj = project()
    line = " ".join((body.get("line") or "").split())
    if not line:
        raise fail(422, "Write the rule first.")
    settings = proj.store.settings()
    lines = [l for l in settings["extra_instructions"].splitlines() if l.strip()]
    if line not in lines:
        lines.append(line)
    settings["extra_instructions"] = "\n".join(lines)
    proj.store.save_settings(settings)
    return {"settings": settings}


@app.post("/api/rules/matches")
async def rule_matches(body: dict = Body(...)):
    word = (body.get("word") or "").strip()
    if not word:
        return {"matches": []}
    pattern = re.compile(r"(?<!\w)" + re.escape(word) + r"(?!\w)", re.I)
    matches = []
    for image in project().store.images():
        if any(pattern.search(str(v)) for v in (image["fields"] or {}).values()):
            matches.append({"id": image["id"], "rel": image["rel"]})
    return {"matches": matches}


# -- find and replace ------------------------------------------------------

def _replace_args(body: dict[str, Any]) -> dict[str, Any]:
    return {"regex": bool(body.get("regex")), "case": bool(body.get("case")), "whole": bool(body.get("whole"))}


@app.post("/api/replace/preview")
async def replace_preview(body: dict = Body(...)):
    try:
        changes = await asyncio.to_thread(project().replace_preview, body.get("find") or "", body.get("replace") or "", **_replace_args(body))
    except re.error as exc:
        raise fail(422, f"That pattern is not valid: {exc}")
    return {"changes": changes}


@app.post("/api/replace/apply")
async def replace_apply(body: dict = Body(...)):
    proj = project()
    try:
        changes = proj.replace_preview(body.get("find") or "", body.get("replace") or "", **_replace_args(body))
    except re.error as exc:
        raise fail(422, f"That pattern is not valid: {exc}")
    only = set(body["ids"]) if body.get("ids") is not None else None
    done = proj.replace_apply(changes, only)
    await asyncio.to_thread(refresh_template, proj, load_settings())
    return {"changed": done}


# -- test bench ------------------------------------------------------------

def bench_payload() -> dict[str, Any]:
    proj = project()
    pins = []
    for image_id, facts in proj.store.testset().items():
        image = proj.store.image(image_id)
        if image and not image["missing"]:
            pins.append({"id": image_id, "rel": image["rel"], "facts": facts})
    pins.sort(key=lambda pin: natural_key(pin["rel"]))
    return {"pins": pins, "runs": [summarize(r) for r in proj.store.test_runs()], "busy": state.bench_busy}


@app.get("/api/test")
async def bench_get():
    return bench_payload()


@app.post("/api/test/pin")
async def bench_pin(body: dict = Body(...)):
    image_or_404(int(body["id"]))
    project().store.pin(int(body["id"]), bool(body.get("pinned", True)))
    return bench_payload()


@app.put("/api/test/facts")
async def bench_facts(body: dict = Body(...)):
    facts = [" ".join(str(f).split()) for f in body.get("facts") or [] if str(f).strip()]
    project().store.set_facts(int(body["id"]), facts)
    return bench_payload()


@app.post("/api/test/run")
async def bench_run(body: dict = Body(default={})):
    proj, settings = project(), load_settings()
    if not settings.get("model"):
        raise fail(409, "No model is chosen. Pick one in the Candy Shop tab first.")
    if not proj.store.testset():
        raise fail(409, "The Sampler Box is empty. Pin a few images first.")
    if state.bench_busy or (state.runner and state.runner.busy):
        raise fail(409, "Something is already running.")
    label = (body.get("label") or "").strip() or time.strftime("Version %H:%M:%S")
    state.bench_busy = True
    client = ModelClient(settings)
    try:
        ids = [await run_bench(proj, settings, client, label, state.broadcaster.publish)]
        if body.get("twice"):
            ids.append(await run_bench(proj, settings, client, label + " (second run)", state.broadcaster.publish))
    finally:
        state.bench_busy = False
        await client.close()
    return {"run_ids": ids, **bench_payload()}


@app.get("/api/test/run/{run_id}")
async def bench_run_detail(run_id: int):
    run = project().store.test_run(run_id)
    if not run:
        raise fail(404, "That test run is gone.")
    names = {f["key"]: f["name"] for f in run["snapshot"]["prefill"]["fields"]}
    for result in run["results"]:
        for reason in result.get("reasons") or []:
            reason["field_name"] = names.get(reason.get("field"))
        result.pop("signals", None)
    return {**run, "summary": summarize(run)}


@app.post("/api/test/request")
async def bench_request(body: dict = Body(...)):
    settings = load_settings()
    return await request_preview(project(), settings, image_or_404(int(body["id"])))


@app.post("/api/test/promote")
async def bench_promote(body: dict = Body(...)):
    run = project().store.test_run(int(body["run_id"]))
    if not run:
        raise fail(404, "That test run is gone.")
    return {"settings": promote(project(), run)}
