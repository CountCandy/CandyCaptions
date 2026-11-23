"""Test connection: send one tiny image and see what the server really supports.

The results are stored as the connection's capabilities, so the pipeline only
asks for things this server was seen to do. This also answers, at run time,
the design doc's open questions about token probabilities.
"""

from __future__ import annotations

import json
from typing import Any

from .backends import default_caps
from .client import ModelClient, ModelServerError, image_part, text_part
from .imaging import tiny_test_image

TEST_SCHEMA = {"type": "object", "properties": {"answer": {"type": "string"}}, "required": ["answer"], "additionalProperties": False}


async def test_connection(settings: dict[str, Any], probe_image_url: str | None = None, transport=None) -> dict[str, Any]:
    """Returns {ok, models, caps, lines}; lines are plain-language findings for the UI."""
    caps = default_caps(settings.get("backend", "other"))
    lines: list[dict[str, Any]] = []

    def note(ok: bool | None, text: str) -> None:
        lines.append({"ok": ok, "text": text})

    probe = dict(settings, caps={**caps, "chat_logprobs": True, "json_schema": True})
    client = ModelClient(probe, transport=transport)
    sampling = {"temperature": 0.0, "seed": 1}
    try:
        try:
            models = await client.list_models()
        except ModelServerError as exc:
            note(False, str(exc))
            return {"ok": False, "models": [], "caps": caps, "lines": lines}
        note(True, f"Connected. {len(models)} model{'s' if len(models) != 1 else ''} available.")
        model = settings.get("model") or (models[0] if models else "")
        if not model:
            note(False, "The server lists no models. Load one, then test again.")
            return {"ok": False, "models": models, "caps": caps, "lines": lines}

        image = tiny_test_image()
        messages = [{"role": "user", "content": [text_part("Name the two colours of the squares in a few words."), image_part(image, probe_image_url)]}]

        # 1. An image on chat completions, with token probabilities if the server gives them.
        try:
            reply = await client.complete(model, messages, sampling=sampling, max_tokens=24, want_logprobs=True)
            caps["chat_logprobs"] = bool(reply.tokens)
        except ModelServerError as exc:
            if exc.kind == "connect":
                raise
            client.caps["chat_logprobs"] = False
            caps["chat_logprobs"] = False
            reply = await client.complete(model, messages, sampling=sampling, max_tokens=24)
        note(True, f"{model} read the test image and answered: \"{reply.text[:80]}\"")
        note(caps["chat_logprobs"], "Token probabilities on chat completions." if caps["chat_logprobs"]
             else "No token probabilities on chat completions.")

        # 2. JSON schema output.
        try:
            reply = await client.complete(model, [{"role": "user", "content": "Reply with JSON: the answer is the word ok."}],
                                          sampling=sampling, max_tokens=24, schema=("connection_test", TEST_SCHEMA))
            json.loads(reply.text)
            caps["json_schema"] = True
            note(True, "JSON schema output works.")
        except (ModelServerError, json.JSONDecodeError) as exc:
            if isinstance(exc, ModelServerError) and exc.kind == "connect":
                raise
            caps["json_schema"] = False
            note(False, "JSON schema output is not enforced here. Replies are still validated, so a bad one is retried.")

        # 3. Only when chat has no probabilities: look for them on /v1/responses.
        if not caps["chat_logprobs"]:
            await _probe_responses(client, model, messages, sampling, caps, note)

        if not caps["chat_logprobs"] and not caps["responses_logprobs"]:
            note(None, "Without token probabilities, Mixed Flavours (uncertain words) is off and Taste Check "
                       "(field checks) uses plain yes or no.")
        caps["tested"] = True
        return {"ok": True, "models": models, "caps": caps, "lines": lines}
    except ModelServerError as exc:
        note(False, str(exc))
        return {"ok": False, "models": [], "caps": caps, "lines": lines}
    finally:
        await client.close()


async def _probe_responses(client: ModelClient, model: str, messages: list, sampling: dict, caps: dict, note) -> None:
    def setcaps(**values: Any) -> None:
        caps.update(values)
        client.caps.update(values)

    setcaps(chat_logprobs=False, responses_logprobs=True, responses_schema=True)
    reply = None
    for mode in ("base64", "url"):
        if mode == "url" and not messages[0]["content"][1].get("_http_url"):
            break
        setcaps(responses_image=mode)
        try:
            reply = await client.complete(model, messages, sampling=sampling, max_tokens=24, want_logprobs=True)
            if reply.route == "responses" and reply.text:
                break
            reply = None
        except ModelServerError as exc:
            if exc.kind == "connect":
                raise
            reply = None
    if reply is None:
        setcaps(responses=False, responses_image=None, responses_logprobs=False, responses_schema=False)
        note(False, "/v1/responses did not accept the test image, so it is not used.")
        return
    setcaps(responses=True, responses_logprobs=bool(reply.tokens))
    how = "as base64" if caps["responses_image"] == "base64" else "by local address"
    note(True, f"/v1/responses accepts images {how}.")
    note(caps["responses_logprobs"], "Token probabilities on /v1/responses." if caps["responses_logprobs"]
         else "No token probabilities on /v1/responses either.")
    if not caps["responses_logprobs"]:
        setcaps(responses_schema=False)
        return
    try:
        text_only = [{"role": "user", "content": "Reply with JSON: the answer is the word ok."}]
        reply = await client.complete(model, text_only, sampling=sampling, max_tokens=24, want_logprobs=True,
                                      schema=("connection_test", TEST_SCHEMA))
        json.loads(reply.text)
        note(True, "JSON schema output works on /v1/responses.")
    except (ModelServerError, json.JSONDecodeError) as exc:
        if isinstance(exc, ModelServerError) and exc.kind == "connect":
            raise
        setcaps(responses_schema=False)
        note(False, "JSON schema output does not work on /v1/responses, so field replies stay on chat completions "
                    "and Mixed Flavours (uncertain words) is off.")
    try:
        setcaps(responses_extra_sampling=True)
        await client.complete(model, [{"role": "user", "content": "Say ok."}],
                              sampling={"temperature": 0.0, "top_k": 40, "repeat_penalty": 1.05, "seed": 1},
                              max_tokens=8, want_logprobs=True)
        note(True, "/v1/responses accepted top_k, repeat_penalty and seed.")
    except ModelServerError as exc:
        if exc.kind == "connect":
            raise
        setcaps(responses_extra_sampling=False)
        note(None, "/v1/responses refused top_k, repeat_penalty and seed, so only temperature and top_p are sent there.")
