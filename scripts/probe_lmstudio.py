"""Probe an LM Studio server for the design doc's open questions.

Run it on the PC that runs LM Studio, with the captioning model loaded:

    uv run python scripts/probe_lmstudio.py
    uv run python scripts/probe_lmstudio.py --model gemma-4-31b --parallel 4

It checks:
  1. /v1/responses with a base64 image (and by local URL if base64 is refused)
  2. whether sampling settings and a JSON schema work on /v1/responses
  3. token probabilities on /v1/chat/completions and on /v1/responses
  4. how context is split across parallel requests, and what one image costs

It prints a report and saves it as probe_report.json. Nothing here changes
LM Studio's settings; it only sends requests.
"""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
import http.server
import io
import json
import threading
import time
from typing import Any

import httpx
from PIL import Image, ImageDraw

from candycaption.imaging import tiny_test_image

SCHEMA = {"type": "object", "properties": {"colors": {"type": "string"}, "count": {"type": "string"}},
          "required": ["colors", "count"], "additionalProperties": False}
report: dict[str, Any] = {}


def say(mark: str, text: str) -> None:
    print(f"  {mark} {text}")


class Server:
    def __init__(self, base_url: str, api_key: str, model: str):
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self.http = httpx.Client(base_url=base_url.rstrip("/"), headers=headers, timeout=600)
        self.root = base_url.rstrip("/").removesuffix("/v1")
        self.model = model

    def post(self, path: str, body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        response = self.http.post(path, json={"model": self.model, **body})
        try:
            data = response.json()
        except ValueError:
            data = {"error": response.text[:300]}
        return response.status_code, data

    def responses(self, content: Any, **extra: Any) -> tuple[int, dict[str, Any], str, list | None]:
        """Returns (status, raw, text, logprobs)."""
        status, data = self.post("/responses", {"input": [{"role": "user", "content": content}], "stream": False, **extra})
        text, logprobs = "", None
        for item in data.get("output") or []:
            for part in item.get("content") or [] if item.get("type") == "message" else []:
                if part.get("type") == "output_text":
                    text += part.get("text", "")
                    logprobs = part.get("logprobs") or logprobs
        return status, data, text.strip(), logprobs

    def chat(self, content: Any, **extra: Any) -> tuple[int, dict[str, Any], str]:
        status, data = self.post("/chat/completions", {"messages": [{"role": "user", "content": content}], "stream": False, **extra})
        text = ""
        if status == 200 and data.get("choices"):
            text = (data["choices"][0].get("message") or {}).get("content") or ""
        return status, data, text.strip()


def error_of(data: dict[str, Any]) -> str:
    err = data.get("error")
    return (err.get("message") if isinstance(err, dict) else str(err))[:200] if err else ""


def serve_image(jpeg: bytes) -> str:
    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.end_headers()
            self.wfile.write(jpeg)

        def log_message(self, *args):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return f"http://127.0.0.1:{server.server_address[1]}/probe.jpg"


def big_image(side: int) -> str:
    img = Image.new("RGB", (side, side), (240, 235, 245))
    draw = ImageDraw.Draw(img)
    for n in range(40):
        x, y = (n * 197) % side, (n * 331) % side
        draw.ellipse([x, y, x + side // 6, y + side // 6], fill=((n * 53) % 255, (n * 97) % 255, (n * 29) % 255))
    buffer = io.BytesIO()
    img.save(buffer, "JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()


# -- 1. /v1/responses with an image ----------------------------------------

def check_responses_image(server: Server) -> None:
    print("\n1. /v1/responses with an image")
    image = tiny_test_image()
    question = {"type": "input_text", "text": "Name the two colors of the squares in a few words."}
    status, data, text, _ = server.responses([question, {"type": "input_image", "image_url": image}], max_output_tokens=24)
    ok = status == 200 and bool(text)
    report["responses_base64_image"] = {"works": ok, "status": status, "answer": text, "error": error_of(data)}
    say("✓" if ok else "✗", f"base64 image: {'answered ' + repr(text) if ok else f'refused ({status}) {error_of(data)}'}")
    if ok:
        return
    url = serve_image(base64.b64decode(image.split(",", 1)[1]))
    status, data, text, _ = server.responses([question, {"type": "input_image", "image_url": url}], max_output_tokens=24)
    ok = status == 200 and bool(text)
    report["responses_url_image"] = {"works": ok, "status": status, "answer": text, "error": error_of(data)}
    say("✓" if ok else "✗", f"image by local URL: {'answered ' + repr(text) if ok else f'refused ({status}) {error_of(data)}'}")


# -- 2. sampling settings and JSON schema on /v1/responses ------------------

def check_responses_settings(server: Server) -> None:
    print("\n2. Sampling settings and JSON schema on /v1/responses")
    prompt = "Write one sentence about a candy shop."
    found: dict[str, Any] = {}
    for name, value in (("temperature", 0.3), ("top_p", 0.95), ("top_k", 40), ("repeat_penalty", 1.05),
                        ("seed", 42), ("min_p", 0.05), ("max_output_tokens", 40)):
        status, data, _, _ = server.responses(prompt, **{"max_output_tokens": 40, name: value})
        found[name] = {"accepted": status == 200, "error": error_of(data)}
        say("✓" if status == 200 else "✗", f"{name}: {'accepted' if status == 200 else f'refused ({status}) {error_of(data)}'}")

    # Accepted is not the same as honored, so measure the ones that can be measured.
    def sample(**extra: Any) -> str:
        return server.responses(prompt, max_output_tokens=40, temperature=1.2, **extra)[2]

    if found["seed"]["accepted"]:
        a, b, c = sample(seed=7), sample(seed=7), sample(seed=8)
        found["seed"]["same_seed_same_text"] = a == b
        found["seed"]["other_seed_other_text"] = a != c
        say("✓" if a == b and a != c else "–", f"seed: same seed gave {'the same' if a == b else 'different'} text; "
                                    f"another seed gave {'different' if a != c else 'the same'} text")
    if found["top_k"]["accepted"]:
        a, b = sample(top_k=1, seed=1), sample(top_k=1, seed=2)
        found["top_k"]["honored"] = a == b
        say("✓" if a == b else "–", f"top_k: top_k=1 with two seeds gave {'the same' if a == b else 'different'} text")
    status, data, text, _ = server.responses("Count from 1 to 200, separated by spaces.", max_output_tokens=8)
    used = (data.get("usage") or {}).get("output_tokens")
    found["max_output_tokens"]["honored"] = used is not None and used <= 10
    say("✓" if found["max_output_tokens"]["honored"] else "–", f"max_output_tokens=8 produced {used} tokens")
    report["responses_sampling"] = found

    fmt = {"format": {"type": "json_schema", "name": "probe", "schema": SCHEMA, "strict": True}}
    status, data, text, _ = server.responses(
        [{"type": "input_text", "text": "Describe the squares."}, {"type": "input_image", "image_url": tiny_test_image()}],
        max_output_tokens=60, text=fmt)
    try:
        parsed = json.loads(text)
        ok = status == 200 and set(parsed) == {"colors", "count"}
    except ValueError:
        ok = False
    report["responses_json_schema"] = {"works": ok, "status": status, "reply": text[:200], "error": error_of(data)}
    say("✓" if ok else "✗", f"JSON schema: {'reply matches the schema: ' + text if ok else f'did not hold ({status}) {error_of(data) or text[:120]}'}")


# -- 3. token probabilities -------------------------------------------------

def check_probabilities(server: Server) -> None:
    print("\n3. Token probabilities")
    status, data, text = server.chat("Is the sky blue? Answer yes or no.", max_tokens=4, logprobs=True, top_logprobs=5)
    content = ((data.get("choices") or [{}])[0].get("logprobs") or {}).get("content") if status == 200 else None
    report["chat_logprobs"] = {"works": bool(content), "status": status, "error": error_of(data), "sample": (content or [])[:1]}
    say("✓" if content else "✗", "/v1/chat/completions: " + (f"returned, with {len(content[0].get('top_logprobs') or [])} alternatives per token"
                                                              if content else f"none ({status}) {error_of(data)}"))

    status, data, text, logprobs = server.responses("Is the sky blue? Answer yes or no.", max_output_tokens=4,
                                                    top_logprobs=5, include=["message.output_text.logprobs"])
    report["responses_logprobs"] = {"works": bool(logprobs), "status": status, "error": error_of(data), "sample": (logprobs or [])[:1]}
    say("✓" if logprobs else "✗", "/v1/responses: " + (f"returned, with {len(logprobs[0].get('top_logprobs') or [])} alternatives per token"
                                                       if logprobs else f"none ({status}) {error_of(data)}"))

    fmt = {"format": {"type": "json_schema", "name": "probe", "schema": SCHEMA, "strict": True}}
    status, data, text, logprobs = server.responses(
        [{"type": "input_text", "text": "Describe the squares."}, {"type": "input_image", "image_url": tiny_test_image()}],
        max_output_tokens=60, text=fmt, top_logprobs=5, include=["message.output_text.logprobs"])
    both = status == 200 and bool(logprobs) and text.startswith("{")
    report["responses_image_schema_logprobs_together"] = {"works": both, "status": status, "error": error_of(data)}
    say("✓" if both else "✗", "image + JSON schema + probabilities in one /v1/responses request: "
        + ("works, so Mixed Flavors (uncertain words) can run on LM Studio" if both else f"does not work ({status}) {error_of(data)}"))


# -- 4. context across parallel requests -----------------------------------

def loaded_context(server: Server) -> int | None:
    try:
        data = httpx.get(f"{server.root}/api/v0/models", timeout=10).json()
        for item in data.get("data") or []:
            if item.get("id") == server.model:
                return item.get("loaded_context_length") or item.get("max_context_length")
    except (httpx.HTTPError, ValueError):
        pass
    return None


def check_context(server: Server, parallel: int, long_side: int, context_override: int | None = None) -> None:
    print(f"\n4. Context across {parallel} parallel requests")
    status, data, _ = server.chat([{"type": "text", "text": "Describe this image in one word."},
                                   {"type": "image_url", "image_url": {"url": big_image(long_side)}}], max_tokens=4)
    image_tokens = (data.get("usage") or {}).get("prompt_tokens")
    report["image_prompt_tokens"] = {"long_side": long_side, "prompt_tokens": image_tokens, "error": error_of(data)}
    say("✓" if image_tokens else "–", f"one {long_side} px image plus a short question costs {image_tokens} prompt tokens")

    context = context_override or loaded_context(server)
    report["loaded_context_length"] = context
    if not context:
        say("–", "could not read the loaded context length from /api/v0/models. Run again with --context N, "
                 "where N is the context length the model is loaded with in LM Studio")
        return
    say("✓", f"the model is loaded with a context of {context} tokens")

    # A prompt that fits the whole context but not a 1/N share of it. If the server splits its
    # context across slots, these overflow or get truncated when sent together.
    _, data, _ = server.chat("x " * 200, max_tokens=1)
    per_pair = ((data.get("usage") or {}).get("prompt_tokens") or 210) / 200
    target = int(context * (1 / parallel + 0.15))
    if target > 60000:
        say("–", f"the split test would need {target}-token prompts; lower the context length in LM Studio to 16k-32k and run again")
        report["context_split"] = {"skipped": "context too large to test quickly"}
        return
    filler = "x " * int(target / per_pair)
    prompt = f"The first word is PRALINE. {filler}\nThe last word is NOUGAT. Reply with the first word and the last word."

    def one(_: int) -> dict[str, Any]:
        status, data, text = server.chat(prompt, max_tokens=12, temperature=0)
        return {"status": status, "prompt_tokens": (data.get("usage") or {}).get("prompt_tokens"),
                "kept_start": "PRALINE" in text.upper(), "kept_end": "NOUGAT" in text.upper(), "error": error_of(data)}

    alone = one(0)
    started = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(parallel) as pool:
        together = list(pool.map(one, range(parallel)))
    seconds = time.monotonic() - started
    fine = lambda r: r["status"] == 200 and r["kept_start"] and r["kept_end"]  # noqa: E731
    report["context_split"] = {"prompt_tokens_target": target, "alone": alone, "together": together, "seconds": round(seconds, 1)}
    say("✓" if fine(alone) else "✗", f"alone, a {alone['prompt_tokens']}-token prompt: "
        + ("answered with both words" if fine(alone) else f"status {alone['status']}, start kept {alone['kept_start']}, end kept {alone['kept_end']} {alone['error']}"))
    good = sum(fine(r) for r in together)
    if not fine(alone):
        verdict = "inconclusive: the prompt did not survive even on its own"
    elif good == parallel:
        verdict = f"NOT split: every parallel request kept the whole {target}-token prompt, so each has the full context"
    else:
        verdict = (f"SPLIT (or limited): only {good} of {parallel} parallel requests kept the whole prompt. "
                   f"Plan for about {context // parallel} tokens per request, and raise the context length to parallel × what one image needs")
    report["context_split"]["verdict"] = verdict
    say("✓" if good == parallel else "!", verdict)
    if image_tokens:
        need = image_tokens + 1200
        say("•", f"one image with its fields needs roughly {need} tokens (image + instructions + 800 for the reply), "
                 f"so {parallel} in parallel want a context of at least {need * parallel} if it is split")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", default="http://localhost:1234/v1")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--model", default="", help="model identifier; defaults to the first one the server lists")
    parser.add_argument("--parallel", type=int, default=4, help="set this to the model's Max Concurrent Predictions")
    parser.add_argument("--long-side", type=int, default=1536)
    parser.add_argument("--context", type=int, help="the model's loaded context length, if it cannot be read from the server")
    parser.add_argument("--skip-context", action="store_true")
    args = parser.parse_args()

    try:
        listed = httpx.get(args.base_url.rstrip("/") + "/models", timeout=10,
                           headers={"Authorization": f"Bearer {args.api_key}"} if args.api_key else {}).json()
    except (httpx.HTTPError, ValueError) as exc:
        raise SystemExit(f"Could not reach {args.base_url}: {exc}\nStart LM Studio's server (Developer tab) and try again.")
    models = [m["id"] for m in listed.get("data", [])]
    model = args.model or (models[0] if models else "")
    if not model:
        raise SystemExit("The server lists no models. Load one in LM Studio first.")
    print(f"Probing {args.base_url} with model {model}")
    report.update(base_url=args.base_url, model=model, models=models)
    server = Server(args.base_url, args.api_key, model)

    for check in (check_responses_image, check_responses_settings, check_probabilities):
        try:
            check(server)
        except httpx.HTTPError as exc:
            say("✗", f"stopped: {exc}")
    if not args.skip_context:
        try:
            check_context(server, max(2, args.parallel), args.long_side, args.context)
        except httpx.HTTPError as exc:
            say("✗", f"stopped: {exc}")

    with open("probe_report.json", "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    print("\nSaved probe_report.json. Paste it back and the LM Studio profile can be set from real answers.")


if __name__ == "__main__":
    main()
