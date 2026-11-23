"""Backend profiles: what each kind of model server honours.

The UI greys out a setting the connected backend ignores, and the request
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
        "thinking": None,
        # Token probabilities only come through /v1/responses (v0.3.39+).
        "chat_logprobs": False,
        "notes": "No min_p. Token probabilities only on /v1/responses; Test connection checks "
                 "whether that endpoint works here. Turn thinking off in LM Studio's model settings.",
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
    if prof["thinking"] == "chat_template_kwargs" and "thinking" in allowed:
        body["chat_template_kwargs"] = {"enable_thinking": bool(sampling.get("thinking", False))}
    return body
