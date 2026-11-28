"""Client for OpenAI-compatible model servers.

Every request is self-contained: the server keeps no memory between requests
and this client never resends earlier messages. ``stream`` is always set
explicitly, because Unsloth's endpoint streams by default when it is left out.
"""

from __future__ import annotations

import copy
import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

import httpx

from .backends import DEFAULT_THINKING_ROOM, effective_caps, sampling_body, thinking_plan

THINK_RE = re.compile(r"^\s*<think>.*?</think>\s*", re.S)


class ModelServerError(Exception):
    """kind is 'connect' (server not answering), 'http' (it refused) or 'reply' (unusable answer)."""

    def __init__(self, kind: str, message: str):
        super().__init__(message)
        self.kind = kind


@dataclass
class Reply:
    text: str
    raw: str
    tokens: list[dict[str, Any]] | None
    completion_tokens: int
    seconds: float
    route: str
    payload: dict[str, Any] = field(default_factory=dict)
    # True when the model produced reasoning ("thinking") as well as, or instead of, an answer.
    thought: bool = False
    cut_off: bool = False

    def why_empty(self) -> str:
        """Plain-words reason for a reply with no usable answer in it."""
        if self.thought:
            return ("The model spent its whole reply on thinking and wrote no answer, even with extra room. Raise "
                    "Thinking room in the recipe's sampling settings, or turn thinking off for this model.")
        if self.cut_off:
            return "The reply was cut off at the token limit before it finished. Raise the max tokens setting."
        return ""


def text_part(text: str) -> dict[str, Any]:
    return {"type": "text", "text": text}


def image_part(data_url: str, http_url: str | None = None) -> dict[str, Any]:
    part: dict[str, Any] = {"type": "image_url", "image_url": {"url": data_url}}
    if http_url:
        part["_http_url"] = http_url
    return part


def redact(payload: Any) -> Any:
    """Copy of a request payload with base64 images replaced by a short marker."""
    if isinstance(payload, dict):
        return {k: redact(v) for k, v in payload.items()}
    if isinstance(payload, list):
        return [redact(v) for v in payload]
    if isinstance(payload, str) and payload.startswith("data:image/"):
        return f"<image: base64, {len(payload) * 3 // 4 // 1024} KB>"
    return payload


def _to_responses_input(messages: list[dict[str, Any]], image_mode: str | None) -> list[dict[str, Any]]:
    out = []
    for msg in messages:
        content = msg["content"]
        if isinstance(content, str):
            out.append({"role": msg["role"], "content": content})
            continue
        parts = []
        for part in content:
            if part["type"] == "text":
                kind = "output_text" if msg["role"] == "assistant" else "input_text"
                parts.append({"type": kind, "text": part["text"]})
            else:
                url = part["image_url"]["url"]
                if image_mode == "url" and part.get("_http_url"):
                    url = part["_http_url"]
                parts.append({"type": "input_image", "image_url": url})
        out.append({"role": msg["role"], "content": parts})
    return out


def _clean_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cleaned = copy.deepcopy(messages)
    for msg in cleaned:
        if isinstance(msg["content"], list):
            for part in msg["content"]:
                part.pop("_http_url", None)
    return cleaned


def _has_image(messages: list[dict[str, Any]]) -> bool:
    return any(
        isinstance(m["content"], list) and any(p["type"] == "image_url" for p in m["content"])
        for m in messages
    )


def normalize_base_url(url: str) -> str:
    """Accept a bare server address: http://127.0.0.1:1234 means http://127.0.0.1:1234/v1."""
    url = url.strip().rstrip("/")
    if url and "://" not in url:
        url = "http://" + url
    if url and not httpx.URL(url).path.strip("/"):
        url += "/v1"
    return url


class ModelClient:
    def __init__(self, settings: dict[str, Any], transport: httpx.AsyncBaseTransport | None = None):
        self.settings = settings
        self.backend = settings.get("backend", "other")
        self.base_url = normalize_base_url(settings.get("base_url") or "")
        self.caps = effective_caps(settings)
        headers = {}
        if settings.get("api_key"):
            headers["Authorization"] = f"Bearer {settings['api_key']}"
        timeout = httpx.Timeout(settings.get("request_timeout", 300), connect=10)
        self.http = httpx.AsyncClient(headers=headers, timeout=timeout, transport=transport)

    async def close(self) -> None:
        await self.http.aclose()

    # -- building -----------------------------------------------------------

    def build(
        self,
        model: str,
        messages: list[dict[str, Any]],
        *,
        sampling: dict[str, Any],
        max_tokens: int,
        schema: tuple[str, dict[str, Any]] | None = None,
        want_logprobs: bool = False,
        top_logprobs: int = 5,
        stream: bool = False,
        think: bool = False,
    ) -> tuple[str, dict[str, Any]]:
        """Return (route, body). The body is exactly what gets sent."""
        caps = self.caps
        switch, room = thinking_plan(caps, sampling, think)
        max_tokens = int(max_tokens) + room     # thinking is paid for out of the same limit as the answer
        route = "chat"
        if want_logprobs and not caps["chat_logprobs"] and caps["responses_logprobs"]:
            image_ok = not _has_image(messages) or caps["responses_image"]
            schema_ok = schema is None or caps["responses_schema"]
            if image_ok and schema_ok:
                route = "responses"

        if route == "responses":
            body: dict[str, Any] = {
                "model": model,
                "input": _to_responses_input(messages, caps["responses_image"]),
                "stream": False,
                "max_output_tokens": int(max_tokens),
                "top_logprobs": top_logprobs,
                "include": ["message.output_text.logprobs"],
            }
            for key in ("temperature", "top_p"):
                if sampling.get(key) is not None:
                    body[key] = sampling[key]
            if caps.get("responses_extra_sampling"):
                extra = sampling_body(self.backend, sampling)
                for key in ("top_k", "repeat_penalty", "seed"):
                    if key in extra:
                        body[key] = extra[key]
            if schema:
                body["text"] = {"format": {"type": "json_schema", "name": schema[0], "schema": schema[1], "strict": True}}
            if "reasoning_effort" in switch:
                body["reasoning"] = {"effort": switch["reasoning_effort"]}
            return route, body

        body = {"model": model, "messages": _clean_messages(messages), "stream": bool(stream)}
        body.update(sampling_body(self.backend, sampling, max_tokens))
        if "max_tokens" not in body:
            body["max_tokens"] = int(max_tokens)
        body.update(switch)
        if stream:
            body["stream_options"] = {"include_usage": True}
        if schema and caps["json_schema"]:
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": schema[0], "strict": True, "schema": schema[1]},
            }
        if want_logprobs and caps["chat_logprobs"]:
            body["logprobs"] = True
            body["top_logprobs"] = top_logprobs
        return route, body

    # -- sending ------------------------------------------------------------

    async def list_models(self) -> list[str]:
        data = await self._post_or_get("GET", "/models")
        if not isinstance(data.get("data"), list):
            raise ModelServerError("reply", f"{self.base_url}/models did not return a model list. "
                                            "Check the base URL; it normally ends in /v1.")
        return sorted(item["id"] for item in data["data"] if "id" in item)

    async def complete(
        self,
        model: str,
        messages: list[dict[str, Any]],
        *,
        sampling: dict[str, Any],
        max_tokens: int,
        schema: tuple[str, dict[str, Any]] | None = None,
        want_logprobs: bool = False,
        top_logprobs: int = 5,
        on_delta: Callable[[str], Awaitable[None]] | None = None,
        think: bool = False,
    ) -> Reply:
        route, body = self.build(
            model, messages, sampling=sampling, max_tokens=max_tokens, schema=schema,
            want_logprobs=want_logprobs, top_logprobs=top_logprobs, stream=on_delta is not None, think=think,
        )
        return await self.send(route, body, on_delta)

    async def send(self, route: str, body: dict[str, Any], on_delta=None) -> Reply:
        """Send one request. If thinking used up the whole reply, ask once more with room for it."""
        reply = await self._send_once(route, body, on_delta)
        if reply.text or not (reply.thought and reply.cut_off):
            return reply
        key = "max_output_tokens" if route == "responses" else "max_tokens"
        roomier = dict(body)
        roomier[key] = int(body.get(key) or 0) + max(DEFAULT_THINKING_ROOM, int(body.get(key) or 0))
        return await self._send_once(route, roomier, on_delta)

    async def _send_once(self, route: str, body: dict[str, Any], on_delta=None) -> Reply:
        started = time.monotonic()
        thought = cut_off = False
        if route == "responses":
            data = await self._post_or_get("POST", "/responses", body)
            raw, tokens = _parse_responses(data)
            used = (data.get("usage") or {}).get("output_tokens", 0)
            thought = any(item.get("type") == "reasoning" for item in data.get("output") or [])
            cut_off = data.get("status") == "incomplete"
        elif body.get("stream"):
            raw, tokens, used, thought, cut_off = await self._stream_chat(body, on_delta)
        else:
            data = await self._post_or_get("POST", "/chat/completions", body)
            try:
                choice = data["choices"][0]
            except (KeyError, IndexError, TypeError):
                raise ModelServerError("reply", "The server's answer had no choices in it.")
            message = choice.get("message") or {}
            raw = message.get("content") or ""
            thought = bool(message.get("reasoning_content") or message.get("reasoning"))
            cut_off = choice.get("finish_reason") == "length"
            tokens = _chat_tokens(choice.get("logprobs"))
            used = (data.get("usage") or {}).get("completion_tokens", 0)
        seconds = time.monotonic() - started
        text = THINK_RE.sub("", raw).strip()
        thought = thought or text != raw.strip() or raw.lstrip().startswith("<think>")
        if raw.lstrip().startswith("<think>") and "</think>" not in raw:
            text = ""   # the reply ended while still thinking
        return Reply(text=text, raw=raw, tokens=tokens, completion_tokens=used or len(tokens or []),
                     seconds=seconds, route=route, payload=body, thought=thought, cut_off=cut_off)

    async def _post_or_get(self, method: str, path: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
        try:
            response = await self.http.request(method, self.base_url + path, json=body)
        except (httpx.ConnectError, httpx.TimeoutException, httpx.RemoteProtocolError, httpx.ReadError) as exc:
            raise ModelServerError("connect", f"The model server at {self.base_url} is not answering ({type(exc).__name__}).")
        return _json_or_raise(response.status_code, response.text)

    async def _stream_chat(self, body: dict[str, Any], on_delta) -> tuple[str, list | None, int, bool, bool]:
        raw, tokens, used, saw_logprobs = "", [], 0, False
        thought = cut_off = False
        try:
            async with self.http.stream("POST", self.base_url + "/chat/completions", json=body) as response:
                if response.status_code >= 400:
                    _json_or_raise(response.status_code, (await response.aread()).decode("utf-8", "replace"))
                async for line in response.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    chunk = line[5:].strip()
                    if chunk == "[DONE]":
                        break
                    try:
                        data = json.loads(chunk)
                    except json.JSONDecodeError:
                        continue
                    if data.get("usage"):
                        used = data["usage"].get("completion_tokens", used)
                    for choice in data.get("choices") or []:
                        delta = choice.get("delta") or {}
                        thought = thought or bool(delta.get("reasoning_content") or delta.get("reasoning"))
                        cut_off = cut_off or choice.get("finish_reason") == "length"
                        piece = delta.get("content")
                        if piece:
                            raw += piece
                            if on_delta:
                                await on_delta(raw)
                        got = _chat_tokens(choice.get("logprobs"))
                        if got:
                            saw_logprobs = True
                            tokens.extend(got)
        except (httpx.ConnectError, httpx.TimeoutException, httpx.RemoteProtocolError, httpx.ReadError) as exc:
            raise ModelServerError("connect", f"The model server at {self.base_url} is not answering ({type(exc).__name__}).")
        return raw, (tokens if saw_logprobs else None), used, thought, cut_off


def _json_or_raise(status: int, text: str) -> dict[str, Any]:
    if status in (502, 503, 504):
        raise ModelServerError("connect", f"The model server answered {status}; it may be loading or stopped.")
    if status >= 400:
        detail = text.strip()[:400]
        try:
            err = json.loads(text).get("error")
            detail = err.get("message", detail) if isinstance(err, dict) else (err or detail)
        except (json.JSONDecodeError, AttributeError):
            pass
        raise ModelServerError("http", f"The model server refused the request ({status}): {detail}")
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        raise ModelServerError("reply", "The model server sent something that is not JSON.")
    if not isinstance(data, dict):
        raise ModelServerError("reply", "The model server sent an unexpected answer.")
    return data


def _chat_tokens(logprobs: dict[str, Any] | None) -> list[dict[str, Any]] | None:
    if not logprobs or not logprobs.get("content"):
        return None
    return [
        {
            "token": item.get("token", ""),
            "logprob": item.get("logprob", 0.0),
            "top": [{"token": t.get("token", ""), "logprob": t.get("logprob", -99.0)} for t in item.get("top_logprobs") or []],
        }
        for item in logprobs["content"]
    ]


def _parse_responses(data: dict[str, Any]) -> tuple[str, list | None]:
    text, tokens, saw = "", [], False
    for item in data.get("output") or []:
        if item.get("type") != "message":
            continue
        for part in item.get("content") or []:
            if part.get("type") != "output_text":
                continue
            text += part.get("text", "")
            if part.get("logprobs"):
                saw = True
                tokens.extend(_chat_tokens({"content": part["logprobs"]}) or [])
    return text, (tokens if saw else None)
