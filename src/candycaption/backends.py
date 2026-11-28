"""Backend profiles: what each kind of model server honors.

The UI grays out a setting the connected backend ignores, and the request
builder here never sends it, so nothing is dropped silently.
"""

from __future__ import annotations

from typing import Any

ALL_SAMPLING = ["temperature", "top_p", "top_k", "min_p", "repeat_penalty", "seed", "max_tokens", "thinking"]

BACKENDS: dict[str, dict[str, Any]] = {
    "lmstudio": {
        "name": "LM Studio",
        "default_url": "http://localhost:1234/v1",
        "api_key": "none",
        "sampling": ["temperature", "top_p", "top_k", "repeat_penalty", "seed", "max_tokens"],
        "repeat_penalty_name": "repeat_penalty",
        # LM Studio's per-request thinking switch has changed between versions, so nothing is sent
        # until Test connection has seen one work.
        "thinking": None,
        # Token probabilities only come through /v1/responses (v0.3.39+).
        "chat_logprobs": False,
        "notes": "No min_p. Token probabilities only on /v1/responses; Test connection checks "
                 "whether that endpoint works here, and whether thinking can be switched per request.",
    },
    "unsloth": {
        "name": "Unsloth Studio",
        "default_url": "http://localhost:8888/v1",
        "api_key": "required",
        "sampling": ALL_SAMPLING,
        "repeat_penalty_name": "repeat_penalty",
        "thinking": "chat_template_kwargs",
        "chat_logprobs": True,
        "notes": "Needs an API key (sk-unsloth-…). Streams by default, so every request sets stream itself.",
    },
    "llamacpp": {
        "name": "llama.cpp server",
        "default_url": "http://localhost:8080/v1",
        "api_key": "optional",
        "sampling": ALL_SAMPLING,
        "repeat_penalty_name": "repeat_penalty",
        "thinking": "chat_template_kwargs",
        "chat_logprobs": True,
        "notes": "",
    },
    "vllm": {
        "name": "vLLM",
        "default_url": "http://localhost:8000/v1",
        "api_key": "optional",
        "sampling": ALL_SAMPLING,
        "repeat_penalty_name": "repetition_penalty",
        "thinking": "chat_template_kwargs",
        "chat_logprobs": True,
        "notes": "No official Windows build; runs through WSL2 or the vllm-windows fork.",
    },
    "other": {
        "name": "Other OpenAI-compatible",
        "default_url": "http://localhost:8000/v1",
        "api_key": "optional",
        "sampling": ["temperature", "top_p", "seed", "max_tokens"],
        "repeat_penalty_name": "repeat_penalty",
        "thinking": None,
        "chat_logprobs": True,
        "notes": "Only the standard OpenAI settings are sent.",
    },
}


def profile(backend: str) -> dict[str, Any]:
    return BACKENDS.get(backend, BACKENDS["other"])


def default_caps(backend: str) -> dict[str, Any]:
    """What to assume before Test connection has looked at the real server."""
    prof = profile(backend)
    return {
        "tested": False,
        # How thinking is switched per request: "reasoning_effort", "chat_template_kwargs" or None (not at all).
        "thinking_control": prof["thinking"],
        # The model was seen thinking without being asked to, so every request needs room for it.
        "thinking_seen": False,
        # /v1/responses was seen to accept the thinking switch too. Until then it is not sent there.
        "responses_reasoning": False,
        "json_schema": True,
        "chat_logprobs": bool(prof["chat_logprobs"]),
        "responses": False,
        "responses_image": None,
        "responses_schema": False,
        "responses_logprobs": False,
    }


def effective_caps(settings: dict[str, Any]) -> dict[str, Any]:
    caps = default_caps(settings.get("backend", "other"))
    caps.update(settings.get("caps") or {})
    return caps


def sampling_body(backend: str, sampling: dict[str, Any], max_tokens: int | None = None) -> dict[str, Any]:
    """Translate the app's sampling settings into this backend's request fields."""
    prof = profile(backend)
    allowed = set(prof["sampling"])
    body: dict[str, Any] = {}
    for key in ("temperature", "top_p", "seed"):
        if key in allowed and sampling.get(key) is not None:
            body[key] = sampling[key]
    if "top_k" in allowed and sampling.get("top_k") is not None:
        body["top_k"] = int(sampling["top_k"])
    if "min_p" in allowed and sampling.get("min_p") is not None:
        body["min_p"] = sampling["min_p"]
    if "repeat_penalty" in allowed and sampling.get("repeat_penalty") is not None:
        body[prof["repeat_penalty_name"]] = sampling["repeat_penalty"]
    if "max_tokens" in allowed and max_tokens:
        body["max_tokens"] = int(max_tokens)
    return body


THINKING_CONTROLS = ("reasoning_effort", "chat_template_kwargs")
DEFAULT_THINKING_ROOM = 2048


def thinking_plan(caps: dict[str, Any], sampling: dict[str, Any], think: bool) -> tuple[dict[str, Any], int]:
    """(fields that switch thinking for this request, extra tokens to allow for it).

    `think` says whether this kind of request is one worth thinking about (looking at the image is;
    a yes or no check is not). Where the server takes a switch, thinking is on only for those. Where it
    takes none, the model thinks or not as the server is set, so every request gets the room.
    """
    control = caps.get("thinking_control")
    wanted = bool(sampling.get("thinking")) and think
    room = int(sampling.get("max_tokens_thinking") or DEFAULT_THINKING_ROOM)
    if control == "reasoning_effort":
        return {"reasoning_effort": "medium" if wanted else "none"}, room if wanted else 0
    if control == "chat_template_kwargs":
        return {"chat_template_kwargs": {"enable_thinking": wanted}}, room if wanted else 0
    happens = bool(sampling.get("thinking")) or bool(caps.get("thinking_seen"))
    return {}, room if happens else 0
