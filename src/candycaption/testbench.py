"""Test bench: prove a prompt on a handful of pinned images before a full run.

Bench runs go one image at a time with the fixed seed, write nothing into the
dataset folder, and score each caption against the reference facts written
for its image.
"""

from __future__ import annotations

import copy
from typing import Any, Callable

from .client import ModelClient, ModelServerError, redact
from .dataset import Project
from .pipeline import CaptionError, Captioner, build_snapshot
from .prefills import compose_messages, load_prefills, save_prefills
from .signals import combine
from .wordlists import load_wordlists


async def run_bench(project: Project, app_settings: dict[str, Any], client: ModelClient, label: str,
                    publish: Callable[[dict[str, Any]], None] | None = None) -> int:
    publish = publish or (lambda event: None)
    snapshot = build_snapshot(project, app_settings)
    captioner = Captioner(project, client, app_settings, snapshot, load_wordlists())
    pins = project.store.testset()
    results = []
    from .store import natural_key

    ordered = sorted(((project.store.image(i), facts) for i, facts in pins.items()),
                     key=lambda pair: natural_key(pair[0]["rel"]) if pair[0] else [])
    for number, (image, facts) in enumerate(ordered, 1):
        if not image or image["missing"]:
            continue
        image_id = image["id"]
        publish({"type": "test_progress", "label": label, "done": number - 1, "total": len(pins), "rel": image["rel"]})
        entry: dict[str, Any] = {"image_id": image_id, "rel": image["rel"], "facts": facts}
        try:
            result = await captioner.caption_image(image, write=False)
            verdicts = await captioner.score_facts(result["caption"], facts)
            entry.update(
                fields=result["fields"], caption=result["caption"], seconds=round(result["seconds"], 2),
                signals=result["signals"], score=result["score"],
                reasons=combine(result["signals"], app_settings.get("score_weights") or {})[1],
                verdicts=verdicts, error=None,
            )
        except ModelServerError as exc:
            if exc.kind == "connect":
                raise
            entry.update(error=str(exc), caption="", verdicts=[], reasons=[])
        except CaptionError as exc:
            entry.update(error=str(exc), caption="", verdicts=[], reasons=[])
        results.append(entry)
    publish({"type": "test_progress", "label": label, "done": len(pins), "total": len(pins), "rel": ""})
    return project.store.add_test_run(label, snapshot, results)


def summarize(run: dict[str, Any]) -> dict[str, Any]:
    verdicts = [v for r in run["results"] for v in r.get("verdicts") or []]
    return {
        "id": run["id"], "label": run["label"], "created": run["created"],
        "prefill": run["snapshot"]["prefill"]["name"], "preset": run["snapshot"].get("preset"),
        "images": len(run["results"]), "failed": sum(1 for r in run["results"] if r.get("error")),
        "facts": len(verdicts), "captured": verdicts.count("captured"), "wrong": verdicts.count("wrong"),
        "flagged": sum(1 for r in run["results"] if r.get("reasons")),
    }


async def request_preview(project: Project, app_settings: dict[str, Any], image: dict[str, Any]) -> dict[str, Any]:
    """The exact payloads for one image, with the image itself shown as a marker."""
    snapshot = build_snapshot(project, app_settings)
    client = ModelClient(app_settings)
    captioner = Captioner(project, client, app_settings, snapshot, load_wordlists())
    try:
        route, body = captioner.look_request(image, captioner.image_url(image))
    finally:
        await client.close()
    out = {"look": {"endpoint": "/v1/responses" if route == "responses" else "/v1/chat/completions", "body": redact(body)}}
    fields = image.get("fields") or {f["key"]: f"<{f['name']}>" for f in snapshot["prefill"]["fields"]}
    _, compose_body = client.build(snapshot["model"], compose_messages(snapshot, fields), sampling=snapshot["sampling"],
                                   max_tokens=snapshot["sampling"].get("max_tokens_compose", 200))
    out["compose"] = {"endpoint": "/v1/chat/completions", "body": compose_body,
                      "note": None if image.get("fields") else "Field values are placeholders until this image has been captioned."}
    return out


def promote(project: Project, run: dict[str, Any]) -> dict[str, Any]:
    """Make a tested version the dataset's run settings."""
    snapshot = run["snapshot"]
    tested = copy.deepcopy(snapshot["prefill"])
    prefills = load_prefills()
    current = next((p for p in prefills if p["id"] == tested["id"]), None)
    prefill_id = tested["id"]
    if current is None or current != tested:
        # The prefill was edited since the test; keep the tested text as its own variant.
        prefill_id = f"{tested['id']}-run{run['id']}"
        if not any(p["id"] == prefill_id for p in prefills):
            tested.update(id=prefill_id, name=f"{tested['name']} (tested, run {run['id']})", builtin=False)
            prefills.append(tested)
            save_prefills(prefills)
    settings = project.store.settings()
    settings.update(prefill_id=prefill_id, extra_instructions=snapshot.get("extra_instructions", ""),
                    trigger=snapshot.get("trigger", ""), preset=snapshot.get("preset", settings["preset"]),
                    sampling=dict(snapshot["sampling"]), field_checks=snapshot.get("field_checks", settings["field_checks"]))
    project.store.save_settings(settings)
    return settings
