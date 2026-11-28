"""Test connection: send one tiny image and see what the server really supports.

The results are stored as the connection's capabilities, so the pipeline only
asks for things this server was seen to do. This also answers, at run time,
the design doc's open questions about token probabilities.
"""

from __future__ import annotations

import json
from typing import Any

from .backends import THINKING_CONTROLS, default_caps, profile
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

        # 0. Thinking first: the short test requests below fail on a model that thinks unless room is made for it.
        await _probe_thinking(client, model, settings.get("backend", "other"), caps, note)

        image = tiny_test_image()
        messages = [{"role": "user", "content": [text_part("Name the two colors of the squares in a few words."), image_part(image, probe_image_url)]}]

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
        if reply.text:
            note(True, f"{model} read the test image and answered: \"{reply.text[:80]}\"")
        else:
            note(False, f"{model} gave no answer about the test image. " + (reply.why_empty() or "Check that it accepts images."))
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
            note(None, "Without token probabilities, Mixed Flavors (uncertain words) is off and Taste Check "
                       "(field checks) uses plain yes or no.")
        caps["tested"] = True
        return {"ok": True, "models": models, "caps": caps, "lines": lines}
    except ModelServerError as exc:
        note(False, str(exc))
        return {"ok": False, "models": [], "caps": caps, "lines": lines}
    finally:
        await client.close()


async def _probe_thinking(client: ModelClient, model: str, backend: str, caps: dict, note) -> None:
    """Find out whether this model thinks, and which per-request switch, if any, the server obeys."""
    ask = [{"role": "user", "content": "Reply with the single word ok."}]

    def setcaps(**values: Any) -> None:
        caps.update(values)
        client.caps.update(values)

    async def run(control: str | None, on: bool):
        setcaps(thinking_control=control)
        return await client.complete(model, ask, sampling={"temperature": 0.0, "seed": 1, "thinking": on},
                                     max_tokens=16, think=True)

    setcaps(thinking_seen=False)
    unasked = (await run(None, False)).thought            # nothing sent: what the server does when left alone
    first = profile(backend)["thinking"]
    order = ([first] if first else []) + [c for c in THINKING_CONTROLS if c != first]
    for control in order:
        try:
            off = await run(control, False)
            if off.thought:
                continue                                    # told not to think, and it did: this switch is ignored
            on = await run(control, True)
        except ModelServerError as exc:
            if exc.kind == "connect":
                raise
            continue                                        # the server refused the field
        if on.thought or unasked:
            setcaps(thinking_control=control, thinking_seen=False)
            if on.thought:
                note(True, f"Thinking can be switched per request here (through {control}). CandyCaption turns it on only "
                           "where the recipe asks for it, and off for every other request.")
            else:
                note(None, f"Thinking can be switched off per request here (through {control}), but asking for it "
                           "did not make this model think.")
            return
    setcaps(thinking_control=None, thinking_seen=unasked)
    if unasked:
        note(None, "This model thinks before it answers, and the server takes no per-request switch for it. Every "
                   "request is given extra room so the answer still fits, which makes captioning slower. To stop it, "
                   "turn thinking off in the model's settings on the server.")
    else:
        note(None, "This model does not think, or the server takes no per-request switch for it. To use thinking, "
                   "turn it on in the model's settings on the server, then test again.")


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
                    "and Mixed Flavors (uncertain words) is off.")
    if caps.get("thinking_control") == "reasoning_effort":
        # The switch was proven on chat completions. See whether this endpoint takes it as well.
        try:
            setcaps(responses_reasoning=True)
            reply = await client.complete(model, [{"role": "user", "content": "Say ok."}], sampling=dict(sampling, thinking=False),
                                          max_tokens=16, want_logprobs=True, think=True)
            if reply.route != "responses" or reply.thought or not reply.text:
                raise ModelServerError("reply", "the switch was not obeyed")
        except ModelServerError as exc:
            if exc.kind == "connect":
                raise
            setcaps(responses_reasoning=False)
            note(None, "/v1/responses does not take the thinking switch. Requests that think go through chat completions "
                       "instead, so Mixed Flavors (uncertain words) is off for an image while Thinking is ticked.")
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
