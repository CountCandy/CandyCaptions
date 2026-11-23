"""Command-line mode. It shares the captioning core with the web GUI.

    candycaption caption <folder> --model gemma-4-31b --prefill style
    candycaption health <folder>
    candycaption review <folder>
    candycaption serve
    candycaption mock
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from typing import Any

from .backends import BACKENDS, profile
from .config import load_settings
from .dataset import HEALTH_CHECKS, Project
from .prefills import get_prefill, load_prefills, load_presets
from .signals import combine


def _connection(args: argparse.Namespace) -> dict[str, Any]:
    """Saved settings from the GUI, overridden by anything given on the command line."""
    settings = load_settings()
    if args.backend:
        if args.backend != settings["backend"]:
            settings["caps"] = {}
        settings["backend"] = args.backend
        if not args.base_url:
            settings["base_url"] = profile(args.backend)["default_url"]
    if args.base_url:
        if args.base_url != settings["base_url"]:
            settings["caps"] = {}
        settings["base_url"] = args.base_url
    for key in ("api_key", "model", "verifier_model"):
        if getattr(args, key, None):
            settings[key] = getattr(args, key)
    return settings


def _duration(seconds: float | None) -> str:
    if seconds is None:
        return "?"
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    return f"{seconds // 60}m {seconds % 60:02d}s" if seconds < 3600 else f"{seconds // 3600}h {seconds % 3600 // 60:02d}m"


async def _caption(args: argparse.Namespace) -> int:
    from .runner import Runner

    settings = _connection(args)
    if not settings.get("model"):
        print("No model is chosen. Pass --model, or pick one in the web GUI first.", file=sys.stderr)
        return 2
    project = Project(args.folder)
    ds = project.store.settings()
    if args.prefill:
        prefill = next((p for p in load_prefills() if args.prefill.lower() in (p["id"].lower(), p["name"].lower())), None)
        if prefill is None:
            print(f"No prefill called {args.prefill}. Choose from: {', '.join(p['id'] for p in load_prefills())}", file=sys.stderr)
            return 2
        ds["prefill_id"] = prefill["id"]
    if args.preset:
        preset = next((p for p in load_presets() if p["name"].lower() == args.preset.lower()), None)
        if preset is None:
            print(f"No preset called {args.preset}.", file=sys.stderr)
            return 2
        ds["preset"] = preset["name"]
        ds["sampling"].update(preset["sampling"])
    for key in ("trigger", "parallel", "field_checks"):
        if getattr(args, key) is not None:
            ds[key] = getattr(args, key)
    if args.extra is not None:
        ds["extra_instructions"] = args.extra
    if args.overwrite:
        ds["overwrite"] = True
    if args.seed is not None:
        ds["sampling"]["seed"] = args.seed
    project.store.save_settings(ds)

    def show(event: dict[str, Any]) -> None:
        if event["type"] == "image_done":
            flags = f"  [{', '.join(event['flags'])}]" if event["flags"] else ""
            print(f"  wrapped   {event['rel']}{flags}")
        elif event["type"] == "image_failed":
            print(f"  misshape  {event['rel']}: {event['error']}")
        elif event["type"] == "state" and event["state"] in ("paused", "done", "stopped") and event["message"]:
            print(event["message"])
        elif event["type"] == "state" and event["done"] and event["done"] % 10 == 0 and not event["in_flight"]:
            print(f"  {event['done'] + event['failed']}/{event['total'] - event['skipped']} done, "
                  f"{event['seconds_per_image']}s per image, about {_duration(event['eta_seconds'])} left")

    runner = Runner(project, settings, publish=show)
    prefill = get_prefill(ds["prefill_id"])
    print(f"Captioning {project.folder} with {settings['model']} ({prefill['name']} prefill, {ds['parallel']} at a time)")
    mode = "compose" if args.compose_only else "retry" if args.retry else "caption"
    runner.start(mode=mode)
    try:
        while runner._task and not runner._task.done():
            await asyncio.sleep(0.2)
            if runner.state == "paused":
                # No one to press resume here, so a server that stops answering ends the run.
                runner.stop()
        await runner.wait()
    except (KeyboardInterrupt, asyncio.CancelledError):
        runner.stop()
        await runner.wait()
    if runner.skipped:
        print(f"Skipped {runner.skipped} already captioned (use --overwrite to redo them).")
    return 0 if runner.state == "done" and not runner.failed else 1


def _health(args: argparse.Namespace) -> int:
    project = Project(args.folder)
    settings = load_settings()
    images = project.scan()
    problems = 0
    for image in images:
        for badge in project.health(image, settings):
            problems += 1
            plain, candy, _ = HEALTH_CHECKS[badge["check"]]
            print(f"{image['rel']}: {plain} ({candy}): {badge['detail']}")
    print(f"{len(images)} images checked, {problems} problem{'s' if problems != 1 else ''} found.")
    return 0


def _review(args: argparse.Namespace) -> int:
    project = Project(args.folder)
    weights = load_settings().get("score_weights") or {}
    rows = []
    for image in project.store.images():
        score, reasons = combine(image["signals"] or {}, weights)
        if image["status"] == "failed":
            rows.append((2.0, image["rel"], [f"failed: {image['error']}"]))
        elif reasons:
            rows.append((score, image["rel"], [r["message"] for r in reasons]))
    for score, rel, messages in sorted(rows, reverse=True)[: args.limit]:
        print(f"{rel}  (score {score:.2f})" if score <= 1 else rel)
        for message in messages:
            print(f"    {message}")
    print(f"{len(rows)} image{'s' if len(rows) != 1 else ''} to look at.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="candycaption", description="Caption a LoRA dataset and review the doubtful captions.")
    sub = parser.add_subparsers(dest="command", required=True)

    cap = sub.add_parser("caption", help="caption every image in a folder")
    cap.add_argument("folder")
    cap.add_argument("--backend", choices=list(BACKENDS))
    cap.add_argument("--base-url")
    cap.add_argument("--api-key")
    cap.add_argument("--model")
    cap.add_argument("--verifier-model")
    cap.add_argument("--prefill", help="style, character, concept, or one of your variants")
    cap.add_argument("--preset", help="Faithful, Balanced, Varied, or one of your own")
    cap.add_argument("--trigger")
    cap.add_argument("--extra", help="extra instructions for this dataset")
    cap.add_argument("--parallel", type=int, help="1 to 8 images in flight at once")
    cap.add_argument("--field-checks", dest="field_checks", choices=["off", "flagged", "all"])
    cap.add_argument("--seed", type=int)
    cap.add_argument("--overwrite", action="store_true", help="redo images that already have a description")
    cap.add_argument("--retry", action="store_true", help="only redo the images on the retry list")
    cap.add_argument("--compose-only", action="store_true", help="rewrite captions from stored fields, without looking at images again")

    health = sub.add_parser("health", help="run the dataset health checks")
    health.add_argument("folder")

    review = sub.add_parser("review", help="list the captions most likely to be wrong")
    review.add_argument("folder")
    review.add_argument("--limit", type=int, default=50)

    serve = sub.add_parser("serve", help="start the web GUI")
    serve.add_argument("--port", type=int, default=8200)

    mock = sub.add_parser("mock", help="start the fake model server, for trying the app without a GPU")
    mock.add_argument("--port", type=int, default=1234)
    mock.add_argument("--delay", type=float, default=1.5, help="seconds each reply takes")
    mock.add_argument("--lmstudio", action="store_true", help="behave like LM Studio: token probabilities only on /v1/responses")

    args = parser.parse_args(argv)
    try:
        if args.command == "caption":
            return asyncio.run(_caption(args))
        if args.command == "health":
            return _health(args)
        if args.command == "review":
            return _review(args)
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 2
    import uvicorn

    if args.command == "serve":
        uvicorn.run("candycaption.server:app", host="127.0.0.1", port=args.port)
    else:
        from .mock_server import create_app

        uvicorn.run(create_app(delay=args.delay, chat_logprobs=not args.lmstudio), host="127.0.0.1", port=args.port)
    return 0


if __name__ == "__main__":
    sys.exit(main())
