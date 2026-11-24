"""A small fake OpenAI-style model server, so tests and demos need no GPU.

It cannot see images. It builds plausible fields from the tags in the request
and a hash of the image, and it misbehaves on request: put one of these words
in an image's tag file to trigger a failure for that image.

    mock_bad_json    the field reply is broken JSON, every time
    mock_flaky       broken JSON the first time, fine on the retry
    mock_empty       an empty reply
    mock_banned      the first caption uses banned words; the retry is clean
    mock_error       HTTP 500
    mock_unsure      the model wavers between red and orange clothing
    mock_contradict  the hair color disagrees with the tagger
    mock_doubt       the yes/no check will not confirm the Clothing field

Run it with ``candycaption mock`` and point the Candy Shop tab at
http://localhost:1234/v1.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import re
import time
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

COLOURS = ["red", "blue", "green", "black", "white", "pink", "purple", "brown", "blonde", "silver"]
OUTFITS = ["school uniform with a pleated skirt", "hooded jacket over a shirt", "long summer dress", "knitted sweater and jeans", "armored coat"]
PLACES = ["a classroom with desks by the window", "a city street at a crossing", "a forest clearing with tall grass", "a plain white background", "a kitchen counter with shelves"]
POSES = ["standing with one hand raised", "sitting on a bench holding a cup", "walking while looking over the shoulder", "leaning on a railing", "kneeling and reaching forward"]
LIGHTS = ["soft daylight from the left", "warm sunset light", "cool overcast light", "bright indoor lighting", "night, lit by street lamps"]
FACES = ["a small smile, looking at the viewer", "a neutral expression, looking away", "an open-mouthed laugh, eyes closed", "a frown, looking down"]
CAMERAS = ["upper body, seen from the front", "full body, seen from slightly below", "close-up on the face", "full body, seen from behind"]
TOKEN_RE = re.compile(r"\s*(?:[A-Za-z0-9]+|[^A-Za-z0-9\s])|\s+$")
MARKERS = ["mock_bad_json", "mock_flaky", "mock_empty", "mock_banned", "mock_error", "mock_unsure", "mock_contradict", "mock_doubt"]


def _pick(pool: list[str], seed: int, salt: int) -> str:
    return pool[(seed >> salt) % len(pool)]


def _texts_and_image(messages: list[dict[str, Any]]) -> tuple[str, str | None]:
    """All text in the request, and the first image URL if there is one."""
    texts, image = [], None
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, str):
            texts.append(content)
            continue
        for part in content or []:
            if part.get("type") in ("text", "input_text", "output_text"):
                texts.append(part.get("text", ""))
            elif part.get("type") == "image_url":
                image = image or part["image_url"]["url"]
            elif part.get("type") == "input_image":
                image = image or part.get("image_url")
    return "\n".join(texts), image


def make_fields(keys: list[str], tags: list[str], seed: int) -> tuple[dict[str, str], dict[str, tuple[str, float]]]:
    """Fields for an image, plus {word: (alternative, probability)} the model 'wavered' on."""
    def tag_color(suffix: str, salt: int) -> str:
        for tag in tags:
            if tag.endswith("_" + suffix) and tag[: -len(suffix) - 1] in COLOURS:
                return tag[: -len(suffix) - 1]
        return _pick(COLOURS, seed, salt)

    hair, eyes = tag_color("hair", 3), tag_color("eyes", 5)
    if "mock_contradict" in tags:
        hair = next(c for c in COLOURS if c != hair and c not in ("blonde",))
    count = "two girls" if "2girls" in tags else "one boy" if "1boy" in tags else "one girl"
    cloth_color = "red" if "mock_unsure" in tags else _pick(COLOURS, seed, 7)
    values = {
        "quality": "finished piece" + (", small signature in the corner" if seed % 4 == 0 else ", no visible defects"),
        "style": "digital illustration with clean line work and flat shading",
        "camera": _pick(CAMERAS, seed, 2),
        "subjects": count,
        "appearance": f"long {hair} hair, {eyes} eyes",
        "clothing": f"a {cloth_color} scarf and a {_pick(OUTFITS, seed, 9)}",
        "expression": _pick(FACES, seed, 11),
        "action": _pick(POSES, seed, 13),
        "setting": _pick(PLACES, seed, 15) + (" MOCK_BANNED" if "mock_banned" in tags else ""),
        "lighting": _pick(LIGHTS, seed, 17),
        "text": "none",
        "rating": "safe",
    }
    fields = {key: values.get(key, "none") for key in keys}
    unsure = {"red": ("orange", 0.42)} if "mock_unsure" in tags else {}
    return fields, unsure


def tokenize(text: str, unsure: dict[str, tuple[str, float]] | None = None) -> list[dict[str, Any]]:
    out, pending = [], dict(unsure or {})
    for piece in TOKEN_RE.findall(text):
        word = piece.strip()
        entry = {"token": piece, "logprob": -0.02, "top_logprobs": [{"token": piece, "logprob": -0.02}]}
        if word in pending:
            alt, p = pending.pop(word)
            lead = piece[: len(piece) - len(piece.lstrip())]
            entry["logprob"] = math.log(1 - p - 0.03)
            entry["top_logprobs"] = [{"token": piece, "logprob": entry["logprob"]}, {"token": lead + alt, "logprob": math.log(p)}]
        out.append(entry)
    return out


def create_app(*, chat_logprobs: bool = True, responses: bool = True, responses_base64: bool = True,
               api_key: str | None = None, delay: float = 0.0, strict_stream: bool = True) -> FastAPI:
    app = FastAPI(title="CandyCaption mock model server")
    state: dict[str, Any] = {"flaky_seen": set(), "doubt": set(), "requests": []}
    app.state.mock = state

    def answer(body: dict[str, Any], messages: list[dict[str, Any]], schema_name: str | None,
               schema: dict[str, Any] | None) -> tuple[int, str, dict[str, tuple[str, float]]]:
        text, image = _texts_and_image(messages)
        lower = text.lower()
        seed = int(hashlib.sha1((image or text).encode("utf-8")).hexdigest()[:12], 16)
        image_key = hashlib.sha1(image.encode("utf-8")).hexdigest() if image else None

        def is_gray(url: str | None) -> bool:
            """The one thing the mock can see: whether an image is black and white."""
            if not url or not url.startswith("data:"):
                return False
            import base64
            import io

            from PIL import Image, ImageStat

            with Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1]))) as img:
                return ImageStat.Stat(img.convert("HSV")).mean[1] < 6

        if schema_name == "style_note":
            note = "monochrome pencil sketch with rough hatching" if is_gray(image) else "digital illustration with clean line work and flat shading"
            return 200, json.dumps({"style": note}), {}

        if lower.startswith("you are given style notes"):
            return 200, "Digital illustration with clean line work and flat shading. Colors are saturated and evenly lit.", {}

        if "made in that same rendering style" in lower:
            odd = is_gray(image)
            return 200, "no" if odd else "yes", {"__yes__": ("", 0.06 if odd else 0.95)}

        if "how does the first image differ" in lower:
            return 200, "The first image is a monochrome pencil sketch, while the others are flat-colored digital illustrations.", {}

        if schema_name == "caption_fields":
            tag_line = re.search(r"Tags from the tagger[^:]*: (.*)", text)
            tags = [t.strip().lower().replace(" ", "_") for t in tag_line.group(1).split(",")] if tag_line else []
            if "mock_error" in tags:
                return 500, "mock server error", {}
            if "mock_empty" in tags:
                return 200, "", {}
            if "mock_bad_json" in tags or ("mock_flaky" in tags and image_key not in state["flaky_seen"]):
                state["flaky_seen"].add(image_key)
                return 200, '{"quality": "finished piece", "style": "digital', {}
            if "mock_doubt" in tags:
                state["doubt"].add(image_key)
            fields, unsure = make_fields(list(schema["properties"]), tags, seed)
            return 200, json.dumps(fields, ensure_ascii=False), unsure

        if schema_name == "chat_fields":
            current = json.loads(re.search(r"Current fields:\n(\{.*?\n\})", text, re.S).group(1))
            last = next(m["content"] for m in reversed(messages) if m["role"] == "user" and isinstance(m["content"], str))
            for new, old in re.findall(r"(\w+),? not (?:an? |the )?(\w+)", last):
                current = {k: re.sub(rf"\b{re.escape(old)}\b", new, v) for k, v in current.items()}
            for old, new in re.findall(r"(\w+) (?:is|are|should be) (?:an? |the )?(\w+)\s*(?:[.,]|$)", last):
                current = {k: re.sub(rf"\b\w+(?= {re.escape(old)}\b)", new, v) for k, v in current.items()}
            return 200, json.dumps(current, ensure_ascii=False), {}

        if schema_name == "fact_scores":
            caption = re.search(r"Caption: (.*)", text).group(1).lower()
            verdicts = []
            for fact in re.findall(r"^\d+\. (.*)$", text, re.M):
                words = [w for w in re.findall(r"[a-z0-9]+", fact.lower()) if len(w) > 2]
                if all(w in caption for w in words):
                    verdicts.append("captured")
                elif any(w in caption for w in words):
                    verdicts.append("wrong")
                else:
                    verdicts.append("missing")
            return 200, json.dumps({"verdicts": verdicts}), {}

        if schema_name:
            return 200, json.dumps({"answer": "ok"}), {}

        if "answer yes or no" in lower:
            doubted = image_key in state["doubt"] and "field: clothing" in lower
            return 200, "no" if doubted else "yes", {"__yes__": ("", 0.2 if doubted else 0.97)}

        if lower.rstrip().endswith("say ok."):
            return 200, "ok", {}

        if "notes:" in lower:
            notes = text.split("Notes:", 1)[1]
            # Like a model that follows the note, leave out a "defects only" note that reports none.
            lines = [line for line in notes.strip().splitlines()
                     if ": " in line and not ("defects only" in line and "no visible defects" in line)]
            values = [line.split(": ", 1)[1].strip().rstrip(".") for line in lines]
            banned = "MOCK_BANNED" in notes and "Rewrite without" not in text
            sentence = ". ".join(v[0].upper() + v[1:] for v in values if v) + "."
            sentence = sentence.replace(" MOCK_BANNED", "")
            if banned:
                sentence = "This image shows a tasteful masterpiece. " + sentence
            return 200, sentence, {}

        if image:
            return 200, "Red and blue squares on white.", {}
        return 200, "ok", {}

    def with_logprobs(text: str, unsure: dict[str, tuple[str, float]]) -> list[dict[str, Any]]:
        if "__yes__" in unsure:
            p = unsure["__yes__"][1]
            pair = [{"token": "yes", "logprob": math.log(p)}, {"token": "no", "logprob": math.log(1 - p)}]
            pair.sort(key=lambda t: -t["logprob"])
            return [{"token": text, "logprob": pair[0]["logprob"], "top_logprobs": pair}]
        return tokenize(text, unsure)

    def unauthorized(request: Request) -> JSONResponse | None:
        if api_key and request.headers.get("authorization") != f"Bearer {api_key}":
            return JSONResponse({"error": {"message": "Invalid API key."}}, status_code=401)
        return None

    @app.get("/v1/models")
    async def models(request: Request):
        return unauthorized(request) or {"object": "list", "data": [{"id": "mock-captioner"}, {"id": "mock-verifier"}]}

    @app.post("/v1/chat/completions")
    async def chat(request: Request):
        if (denied := unauthorized(request)):
            return denied
        body = await request.json()
        state["requests"].append({"path": "chat", "body": body})
        if strict_stream and "stream" not in body:
            return JSONResponse({"error": {"message": "stream must be set explicitly"}}, status_code=400)
        fmt = (body.get("response_format") or {}).get("json_schema") or {}
        status, text, unsure = answer(body, body["messages"], fmt.get("name"), fmt.get("schema"))
        if status != 200:
            return JSONResponse({"error": {"message": text}}, status_code=status)
        if delay:
            await asyncio.sleep(delay)
        tokens = with_logprobs(text, unsure) if text else []
        want = chat_logprobs and body.get("logprobs")
        usage = {"prompt_tokens": 300, "completion_tokens": len(tokens), "total_tokens": 300 + len(tokens)}
        base = {"id": "mock", "object": "chat.completion", "created": int(time.time()), "model": body.get("model")}

        if not body.get("stream"):
            choice: dict[str, Any] = {"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}
            if want:
                choice["logprobs"] = {"content": tokens}
            return {**base, "choices": [choice], "usage": usage}

        async def stream():
            for tok in tokens:
                choice = {"index": 0, "delta": {"content": tok["token"]}, "finish_reason": None}
                if want:
                    choice["logprobs"] = {"content": [tok]}
                yield f"data: {json.dumps({**base, 'object': 'chat.completion.chunk', 'choices': [choice]})}\n\n"
                if delay:
                    await asyncio.sleep(delay / max(1, len(tokens)))
            yield f"data: {json.dumps({**base, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]})}\n\n"
            yield f"data: {json.dumps({**base, 'choices': [], 'usage': usage})}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(stream(), media_type="text/event-stream")

    @app.post("/v1/responses")
    async def respond(request: Request):
        if (denied := unauthorized(request)):
            return denied
        if not responses:
            return JSONResponse({"error": {"message": "Unknown endpoint"}}, status_code=404)
        body = await request.json()
        state["requests"].append({"path": "responses", "body": body})
        messages = body["input"] if isinstance(body["input"], list) else [{"role": "user", "content": body["input"]}]
        _, image = _texts_and_image(messages)
        if image and image.startswith("data:") and not responses_base64:
            return JSONResponse({"error": {"message": "image_url must be an http(s) URL"}}, status_code=400)
        fmt = (body.get("text") or {}).get("format") or {}
        status, text, unsure = answer(body, messages, fmt.get("name"), fmt.get("schema"))
        if status != 200:
            return JSONResponse({"error": {"message": text}}, status_code=status)
        part: dict[str, Any] = {"type": "output_text", "text": text}
        tokens = with_logprobs(text, unsure) if text else []
        if "message.output_text.logprobs" in (body.get("include") or []):
            part["logprobs"] = tokens
        return {"id": "mock", "object": "response", "status": "completed", "model": body.get("model"),
                "output": [{"type": "message", "role": "assistant", "content": [part]}],
                "usage": {"input_tokens": 300, "output_tokens": len(tokens)}}

    return app


app = create_app()
