"""The caption queue: parallel requests, pause and resume, the retry list and live progress.

The app is the queue. Up to ``parallel`` images are in flight at once and each
one is still its own self-contained request. Pause lets the images in flight
finish. A failed image goes to the retry list instead of stopping the run, and
if the model server stops answering the run pauses itself.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from typing import Any, Callable

from .client import ModelClient, ModelServerError
from .dataset import Project
from .pipeline import CaptionError, Captioner, build_snapshot, refresh_template
from .signals import combine
from .wordlists import load_wordlists


logger = logging.getLogger(__name__)


class Broadcaster:
    """Fan-out of live events to every connected browser tab."""

    def __init__(self) -> None:
        self._queues: set[asyncio.Queue] = set()

    def subscribe(self) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue(maxsize=2000)
        self._queues.add(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue) -> None:
        self._queues.discard(queue)

    def publish(self, event: dict[str, Any]) -> None:
        for queue in list(self._queues):
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                pass


class Runner:
    def __init__(self, project: Project, app_settings: dict[str, Any], *,
                 publish: Callable[[dict[str, Any]], None] | None = None,
                 client_factory: Callable[[dict[str, Any]], ModelClient] | None = None,
                 image_base_url: str | None = None):
        self.project = project
        self.app = app_settings
        self.publish = publish or (lambda event: None)
        self.client_factory = client_factory or ModelClient
        self.image_base_url = image_base_url
        self.state = "idle"
        self.message = ""
        self.mode = "caption"
        self.run_id: int | None = None
        self.total = self.done = self.failed = self.skipped = 0
        self.in_flight: dict[int, dict[str, Any]] = {}
        self.recent: deque[dict[str, Any]] = deque(maxlen=8)
        self._finishes: deque[float] = deque(maxlen=10)
        self._first_done = False
        self._seconds = self._check_seconds = self._look_seconds = 0.0
        self._tokens = 0
        self._go = asyncio.Event()
        self._stop = False
        self._task: asyncio.Task | None = None
        self._queue: asyncio.Queue | None = None

    # -- what the UI sees ---------------------------------------------------

    def status(self) -> dict[str, Any]:
        finished = self.done + self.failed
        left = max(0, self.total - finished - self.skipped)
        eta = None
        if len(self._finishes) >= 2 and left:
            span = self._finishes[-1] - self._finishes[0]
            if span > 0:
                eta = left / ((len(self._finishes) - 1) / span)
        return {
            "state": self.state, "message": self.message, "mode": self.mode, "run_id": self.run_id,
            "total": self.total, "done": self.done, "failed": self.failed, "skipped": self.skipped, "left": left,
            "seconds_per_image": round(self._seconds / self.done, 2) if self.done else None,
            "check_seconds_per_image": round(self._check_seconds / self.done, 2) if self.done else None,
            "check_seconds": round(self._check_seconds, 1),
            "tokens_per_second": round(self._tokens / self._look_seconds, 1) if self._look_seconds else None,
            "eta_seconds": round(eta) if eta is not None else None,
            "in_flight": list(self.in_flight.values()),
            "recent": list(self.recent),
            "interrupted": self.project.store.get_meta("open_run") if self.state in ("idle", "done", "stopped") else None,
        }

    def _emit(self, kind: str, **data: Any) -> None:
        self.publish({"type": kind, **data})

    def _emit_state(self) -> None:
        self._emit("state", **self.status())

    @property
    def busy(self) -> bool:
        return self.state in ("running", "pausing", "paused", "stopping")

    # -- controls -----------------------------------------------------------

    def start(self, *, mode: str = "caption", image_ids: list[int] | None = None, extra: str | None = None,
              resume: bool = False) -> None:
        if self.busy:
            raise RuntimeError("A run is already going.")
        self._task = asyncio.create_task(self._run(mode, image_ids, extra, resume))

    async def wait(self) -> None:
        if self._task:
            await self._task

    def pause(self) -> None:
        if self.state == "running":
            self._go.clear()
            self.state = "pausing" if self.in_flight else "paused"
            self.message = "Pausing after the images in flight finish." if self.in_flight else "Paused."
            self._emit_state()

    def resume(self) -> None:
        if self.state in ("paused", "pausing"):
            self._finishes.clear()
            self._first_done = False
            self.state, self.message = "running", ""
            self._go.set()
            self._emit_state()

    def stop(self) -> None:
        if self.busy:
            self._stop = True
            self.state = "stopping"
            self.message = "Stopping after the images in flight finish."
            self._go.set()
            self._emit_state()

    # -- the run ------------------------------------------------------------

    def _plan(self, mode: str, image_ids: list[int] | None, resume_run: int | None) -> list[dict[str, Any]]:
        settings = self.project.store.settings()
        images = self.project.scan()
        todo = []
        self.skipped = 0
        for image in images:
            if (image.get("health") or {}).get("unreadable"):
                continue
            if image_ids is not None:
                if image["id"] in image_ids:
                    todo.append(image)
                continue
            if mode == "compose":
                if image.get("fields"):
                    todo.append(image)
                continue
            if mode == "retry":
                if image["status"] == "failed":
                    todo.append(image)
                continue
            already = self.project.has_caption(image)
            if resume_run is not None and image.get("run_id") == resume_run and image["status"] == "done":
                self.skipped += 1
            elif already and not settings.get("overwrite") and not self.project.caption_is_stale(image):
                self.skipped += 1
            else:
                todo.append(image)
        return todo

    async def _run(self, mode: str, image_ids: list[int] | None, extra: str | None, resume: bool) -> None:
        store = self.project.store
        self.mode = mode
        self.state, self.message = "running", ""
        self.done = self.failed = 0
        self.in_flight.clear()
        self._finishes.clear()
        self._first_done = False
        self._seconds = self._check_seconds = self._look_seconds = 0.0
        self._tokens = 0
        self._stop = False
        self._go.set()

        open_run = store.get_meta("open_run") if resume else None
        if open_run and store.snapshot(open_run["run_id"]):
            snapshot_id, snapshot = open_run["run_id"], store.snapshot(open_run["run_id"])
        else:
            open_run = None
            snapshot = build_snapshot(self.project, self.app)
            snapshot_id = store.add_snapshot(snapshot)
        self.run_id = snapshot_id

        todo = await asyncio.to_thread(self._plan, mode, image_ids, open_run["run_id"] if open_run else None)
        self.total = len(todo) + self.skipped
        whole_dataset = image_ids is None and mode == "caption"
        if whole_dataset:
            store.set_meta("open_run", {"run_id": snapshot_id, "mode": mode})
        self._emit_state()

        client = self.client_factory(self.app)
        captioner = Captioner(self.project, client, self.app, snapshot, load_wordlists(), self.image_base_url)
        queue: asyncio.Queue = asyncio.Queue()
        for image in todo:
            queue.put_nowait(image)
        self._queue = queue
        parallel = max(1, min(8, int(self.project.store.settings().get("parallel", 4))))

        async def worker() -> None:
            while True:
                await self._go.wait()
                if self._stop:
                    return
                try:
                    image = queue.get_nowait()
                except asyncio.QueueEmpty:
                    return
                await self._one(captioner, image, mode, extra, snapshot_id, queue)
                if self.state == "pausing" and not self.in_flight:
                    self.state, self.message = "paused", self.message.replace("Pausing after the images in flight finish.", "Paused.")
                    self._emit_state()

        try:
            while True:
                await asyncio.gather(*(worker() for _ in range(parallel)))
                # Workers leave when the queue is empty; a self-pause can refill it.
                if self._stop or queue.empty():
                    break
        finally:
            await client.close()
            self.in_flight.clear()
            await asyncio.to_thread(refresh_template, self.project, self.app)
            if self._stop:
                self.state = "stopped"
                self.message = f"Stopped: {self.done} captions wrapped, {queue.qsize()} still in the jar."
            else:
                self.state = "done"
                if whole_dataset:
                    store.set_meta("open_run", None)
                misshapes = f", {self.failed} misshape{'s' if self.failed != 1 else ''} to sort" if self.failed else ""
                self.message = f"Batch done: {self.done} caption{'s' if self.done != 1 else ''} wrapped{misshapes}."
            self._emit_state()

    async def _one(self, captioner: Captioner, image: dict[str, Any], mode: str, extra: str | None,
                   snapshot_id: int, queue: asyncio.Queue) -> None:
        image_id = image["id"]
        card = {"id": image_id, "rel": image["rel"], "stage": "look", "text": ""}
        self.in_flight[image_id] = card
        self._emit("image_start", id=image_id, rel=image["rel"])
        last_sent = 0.0

        async def on_delta(text: str) -> None:
            nonlocal last_sent
            card["text"] = text
            now = time.monotonic()
            if now - last_sent > 0.12:
                last_sent = now
                self._emit("delta", id=image_id, text=text)

        async def on_stage(stage: str) -> None:
            card["stage"] = stage
            self._emit("stage", id=image_id, stage=stage, text=card["text"])

        try:
            if mode == "compose":
                started = time.monotonic()
                card["stage"] = "compose"
                caption = await captioner.recompose(image, reason="recomposed with a new prefill",
                                                    snapshot_id=snapshot_id, run_id=snapshot_id)
                fresh = self.project.store.image(image_id)
                result = {"caption": caption, "score": fresh["score"], "signals": fresh["signals"],
                          "seconds": time.monotonic() - started, "check_seconds": 0.0, "tokens": 0, "look_seconds": 0.0}
            else:
                result = await captioner.caption_image(image, run_id=snapshot_id, snapshot_id=snapshot_id,
                                                       on_delta=on_delta, on_stage=on_stage, extra=extra)
        except ModelServerError as exc:
            self.in_flight.pop(image_id, None)
            if exc.kind == "connect":
                # Put the image back and pause; the server may only be reloading the model.
                queue.put_nowait(image)
                if self.state == "running":
                    self._go.clear()
                    self.state = "pausing" if self.in_flight else "paused"
                    self.message = f"Paused: {exc} Start the model server, then resume."
                self._emit_state()
                return
            self._fail(image, str(exc))
            return
        except (CaptionError, OSError, ValueError) as exc:
            self.in_flight.pop(image_id, None)
            self._fail(image, str(exc))
            return
        except Exception as exc:  # noqa: BLE001  one image must never take the whole run down
            logger.exception("Unexpected error on %s", image["rel"])
            self.in_flight.pop(image_id, None)
            self._fail(image, f"Unexpected error ({type(exc).__name__}: {exc}). The full details are in the CandyCaption window.")
            return

        self.in_flight.pop(image_id, None)
        self.done += 1
        self._seconds += result["seconds"]
        self._check_seconds += result["check_seconds"]
        self._tokens += result["tokens"]
        self._look_seconds += result["look_seconds"]
        # The first image is left out of the estimate: the model is still warming up.
        if self._first_done:
            self._finishes.append(time.monotonic())
        self._first_done = True
        reasons = combine(result["signals"], self.app.get("score_weights") or {})[1]
        entry = {"id": image_id, "rel": image["rel"], "caption": result["caption"], "score": result["score"],
                 "flags": sorted({r["signal"] for r in reasons})}
        self.recent.appendleft(entry)
        self._emit("image_done", **entry)
        self._emit_state()

    def _fail(self, image: dict[str, Any], error: str) -> None:
        self.failed += 1
        if self.project.has_caption(image) and (self.project.store.image(image["id"]) or {}).get("status") == "done":
            # Its caption was written before the fault, so keep the caption and note the fault.
            self.project.store.update_image(image["id"], error=error)
        else:
            self.project.store.update_image(image["id"], status="failed", error=error)
        self._emit("image_failed", id=image["id"], rel=image["rel"], error=error)
        self._emit_state()
