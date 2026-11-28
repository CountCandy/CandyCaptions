"""The captioning pipeline for one image: look, validate, compose, check, write, score.

Two short requests per image. "Look" sends the image and gets the fields back
as JSON. "Compose" is text-only and turns the fields the prefill keeps into one
paragraph, so switching prefill later only reruns compose, never the image step.
"""

from __future__ import annotations

import asyncio
import copy
import json
import logging
import re
import time
from typing import Any, Awaitable, Callable

from . import signals as sig
from .client import ModelClient, ModelServerError, Reply, image_part, text_part
from .dataset import Project
from .imaging import prepare_data_url
from .prefills import (caption_fields, compose_messages, field_schema, find_banned, get_prefill,
                       look_messages, parse_fields)

CHECK_SYSTEM = "You verify claims about an image. Answer with one word: yes or no."
CHAT_SYSTEM = (
    "You are correcting the notes for one image. The user tells you what is wrong. Reply with the full set of "
    "fields as one JSON object, changing only what the user's message requires and leaving every other field as it is."
)
FACTS_SYSTEM = (
    "You compare a caption with reference facts about the same image. For each fact, answer \"captured\" if the "
    "caption states it, \"wrong\" if the caption contradicts it, or \"missing\" if the caption does not mention it."
)
FACTS_SCHEMA = {
    "type": "object",
    "properties": {"verdicts": {"type": "array", "items": {"type": "string", "enum": ["captured", "wrong", "missing"]}}},
    "required": ["verdicts"], "additionalProperties": False,
}


logger = logging.getLogger(__name__)


class CaptionError(Exception):
    pass


def build_snapshot(project: Project, app_settings: dict[str, Any]) -> dict[str, Any]:
    """Everything that shaped a run: the exact prefill, instructions and settings."""
    ds = project.store.settings()
    return {
        "prefill": copy.deepcopy(get_prefill(ds["prefill_id"])),
        "extra_instructions": ds["extra_instructions"],
        "trigger": ds["trigger"],
        "preset": ds["preset"],
        "sampling": dict(ds["sampling"]),
        "field_checks": ds["field_checks"],
        "background": ds["background"],
        "backend": app_settings.get("backend"),
        "model": app_settings.get("model"),
        "verifier_model": app_settings.get("verifier_model") or app_settings.get("model"),
    }


DANGLING = {"a", "an", "and", "are", "as", "at", "behind", "by", "for", "from", "in", "into", "is", "its", "near", "of",
            "on", "or", "over", "that", "the", "their", "to", "under", "was", "while", "with"}


# The recipes ask for American English; some models slip anyway, so the common cases are fixed here.
AMERICAN = {"grey": "gray", "greys": "grays", "greyish": "grayish", "colour": "color", "colours": "colors",
            "coloured": "colored", "colourful": "colorful", "colouring": "coloring", "centre": "center",
            "centres": "centers", "centred": "centered", "armour": "armor", "armoured": "armored",
            "jewellery": "jewelry", "moustache": "mustache", "theatre": "theater", "fibre": "fiber",
            "harbour": "harbor", "neighbouring": "neighboring", "favourite": "favorite", "vapour": "vapor",
            "travelling": "traveling", "kneeling": "kneeling", "plough": "plow", "pyjamas": "pajamas",
            "symbolise": "symbolize", "emphasise": "emphasize", "stylised": "stylized", "towards": "toward"}
_AMERICAN_RE = re.compile(r"\b(" + "|".join(AMERICAN) + r")\b", re.I)


def american(text: str) -> str:
    def swap(match: re.Match) -> str:
        word = match.group(0)
        new = AMERICAN[word.lower()]
        return new.capitalize() if word[0].isupper() else new
    return _AMERICAN_RE.sub(swap, text)


def clean_caption(text: str, cut_off: bool = False) -> str:
    text = american(" ".join(text.split()))
    # Some models break the last sentence off to stay under the word limit ("...illuminated by.").
    last = text.rstrip(".!?\"' ").rsplit(" ", 1)[-1].lower()
    if last in DANGLING and ". " in text:
        text = text[:text.rstrip(".!? ").rfind(". ") + 1]
    if cut_off and text and text[-1] not in ".!?\"'":
        # The reply hit the token limit mid-sentence: keep only the complete sentences.
        end = max(text.rfind(". "), text.rfind("! "), text.rfind("? "))
        text = text[:end + 1] if end > 0 else text
    if len(text) > 1 and text[0] == text[-1] and text[0] in "\"'":
        text = text[1:-1].strip()
    return text


def fields_limit(snapshot: dict[str, Any]) -> int:
    """Token limit for the fields reply: the dataset's setting, but never too small for the recipe.

    Each field may hold up to three sentences, so a recipe with many fields needs more than a limit
    that was saved when there were fewer. A reply cut off mid-JSON is useless, so err on the roomy side.
    """
    return max(int(snapshot["sampling"].get("max_tokens_fields", 1800)), 110 * len(snapshot["prefill"]["fields"]))


def compose_limit(snapshot: dict[str, Any]) -> int:
    """Token limit for the caption: room for the longest one the recipe asks for (about 1.4 tokens a word)."""
    return max(int(snapshot["sampling"].get("max_tokens_compose", 330)), int(snapshot["prefill"].get("max_words", 160) * 1.8) + 40)


def refresh_tag_signals(project: Project, app_settings: dict[str, Any], wordlists: dict[str, Any]) -> int:
    """Work out Flavor Clash again for every captioned image, after tag files changed. Returns how many moved."""
    captioner = Captioner(project, None, app_settings, build_snapshot(project, app_settings), wordlists)  # type: ignore[arg-type]
    weights = app_settings.get("score_weights") or {}
    moved = 0
    for image in project.store.images():
        if not image.get("fields"):
            continue
        signals = captioner.signals_for(image, image["fields"], None, None, image.get("signals"))
        if (signals.get("tags") or {}) != ((image.get("signals") or {}).get("tags") or {}):
            project.store.update_image(image["id"], signals=signals, score=sig.combine(signals, weights)[0])
            moved += 1
    return moved


class Captioner:
    def __init__(self, project: Project, client: ModelClient, app_settings: dict[str, Any],
                 snapshot: dict[str, Any], wordlists: dict[str, Any], image_base_url: str | None = None):
        self.project = project
        self.client = client
        self.app = app_settings
        self.snapshot = snapshot
        self.wordlists = wordlists
        self.image_base_url = image_base_url
        self.prefill = snapshot["prefill"]
        self.names = {f["key"]: f["name"] for f in self.prefill["fields"]}

    # -- requests -----------------------------------------------------------

    def image_url(self, image: dict[str, Any]) -> str:
        return prepare_data_url(self.project.path(image), int(self.snapshot["sampling"].get("long_side", 1536)),
                                self.snapshot.get("background", "#ffffff"))

    def _http_url(self, image: dict[str, Any]) -> str | None:
        if not self.image_base_url:
            return None
        return f"{self.image_base_url}/api/image/{image['id']}?size=prepared"

    def look_request(self, image: dict[str, Any], image_url: str, snapshot: dict[str, Any] | None = None):
        """(route, body) of the look request, exactly as it will be sent."""
        snapshot = snapshot or self.snapshot
        messages = look_messages(snapshot, image_url, self.project.read_tags(image), image["rel"],
                                 self._http_url(image), image.get("note") or "")
        return self.client.build(
            snapshot["model"], messages, sampling=snapshot["sampling"],
            max_tokens=fields_limit(snapshot),
            schema=("caption_fields", field_schema(snapshot["prefill"])), want_logprobs=True, stream=True, think=True,
        )

    async def look(self, image: dict[str, Any], image_url: str, on_delta=None,
                   extra: str | None = None) -> tuple[dict[str, str], Reply]:
        snapshot = self.snapshot
        if extra is not None:
            # What the user says about this one image is kept with the image and used every time.
            image = dict(image, note=extra)
        for attempt in range(2):
            route, body = self.look_request(image, image_url, snapshot)
            if route != "chat" or on_delta is None:
                body["stream"] = False
                body.pop("stream_options", None)
            reply = await self.client.send(route, body, on_delta if body.get("stream") else None)
            fields = parse_fields(reply.text, self.prefill)
            if fields is not None:
                return fields, reply
        raise CaptionError(reply.why_empty() or "The model's reply was malformed or empty twice in a row.")

    async def compose(self, fields: dict[str, str], note: str = "") -> tuple[str, float]:
        """Fields to one paragraph. Reruns once if banned words slip in and keeps the cleaner version."""
        if not caption_fields(self.prefill, fields):
            raise CaptionError("No field has anything to put in a caption.")
        sampling = self.snapshot["sampling"]
        limit = compose_limit(self.snapshot)
        started = time.monotonic()
        reply = await self.client.complete(self.snapshot["model"], compose_messages(self.snapshot, fields, None, note),
                                           sampling=sampling, max_tokens=limit)
        caption = clean_caption(reply.text, reply.cut_off)
        hits = find_banned(caption, self.prefill.get("banned_words") or [])
        if hits or not caption:
            again = await self.client.complete(self.snapshot["model"], compose_messages(self.snapshot, fields, hits or None, note),
                                               sampling=sampling, max_tokens=limit)
            second = clean_caption(again.text, again.cut_off)
            if second and (not caption or len(find_banned(second, self.prefill.get("banned_words") or [])) < len(hits)):
                caption = second
            reply = again if not caption else reply
        if not caption:
            raise CaptionError(reply.why_empty() or "The model returned an empty caption.")
        return caption, time.monotonic() - started

    def checkable(self, fields: dict[str, str]) -> list[str]:
        keys = []
        for item in self.prefill["fields"]:
            value = (fields.get(item["key"]) or "").strip().lower().strip(". ")
            if value and value not in ("none", "n/a", "nothing") and not item.get("choices"):
                keys.append(item["key"])
        return keys

    def check_messages(self, image_url: str, item: dict[str, Any], value: str) -> list[dict[str, Any]]:
        """Image first, then the question, so the image is processed once for all of its checks."""
        question = (f"Field: {item['name']} ({item['holds']}).\nClaim: {value}\n"
                    "Is this claim an accurate description of this image? Answer yes or no.")
        return [{"role": "system", "content": CHECK_SYSTEM},
                {"role": "user", "content": [image_part(image_url), text_part(question)]}]

    async def check(self, image_url: str, fields: dict[str, str], keys: list[str] | None = None) -> dict[str, Any]:
        """Ask the verifier yes or no about each field value, one fresh request per field."""
        started = time.monotonic()
        values, measured = {}, True
        by_key = {f["key"]: f for f in self.prefill["fields"]}
        sampling = dict(self.snapshot["sampling"], temperature=0.0)
        for key in keys if keys is not None else self.checkable(fields):
            messages = self.check_messages(image_url, by_key[key], fields[key])
            reply = await self.client.complete(self.snapshot["verifier_model"], messages, sampling=sampling,
                                               max_tokens=4, want_logprobs=True, top_logprobs=10)
            values[key], exact = sig.yes_probability(reply.text, reply.tokens)
            measured = measured and exact
        return {"values": values, "measured": measured, "seconds": time.monotonic() - started}

    # -- scoring ------------------------------------------------------------

    def checker_name(self) -> str:
        same = self.snapshot["verifier_model"] == self.snapshot["model"]
        return "the model" if same else "the verifier"

    def signals_for(self, image: dict[str, Any], fields: dict[str, str], reply: Reply | None,
                    checks: dict[str, Any] | None, previous: dict[str, Any] | None = None) -> dict[str, Any]:
        tags = sig.parse_tags(self.project.read_tags(image))
        out = dict(previous or {})
        if reply is not None:
            out["uncertain"] = {
                "available": reply.tokens is not None,
                "items": sig.uncertain_words(reply.raw, reply.tokens, self.names, self.wordlists,
                                             float(self.app.get("uncertain_min_alt", 0.15))),
            }
        else:
            # After an edit, a highlight only stands while its word is still in the field.
            kept = [i for i in (out.get("uncertain") or {}).get("items", []) if i["word"] in (fields.get(i["field"]) or "").lower()]
            out["uncertain"] = dict(out.get("uncertain") or {"available": False}, items=kept)
        out["tags"] = {"available": bool(tags), "items": sig.tag_contradictions(fields, tags, self.names, self.wordlists)}
        if checks is not None:
            out["checks"] = {
                "available": True, "measured": checks["measured"], "values": checks["values"],
                "items": sig.check_items(checks["values"], self.names, float(self.app.get("check_threshold", 0.6)), self.checker_name()),
            }
        else:
            out.setdefault("checks", {"available": False, "items": [], "values": {}})
        out.setdefault("template", {"items": []})
        return out

    def safe_signals(self, image: dict[str, Any], fields: dict[str, str], reply: Reply | None,
                     checks: dict[str, Any] | None) -> dict[str, Any]:
        """Scoring must never cost a finished caption: a fault here is logged and leaves the image unflagged."""
        try:
            return self.signals_for(image, fields, reply, checks)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Scoring %s failed", image.get("rel"))
            empty = {"available": False, "items": []}
            return {"uncertain": dict(empty), "tags": dict(empty), "checks": {**empty, "values": {}},
                    "template": {"items": []}, "error": f"{type(exc).__name__}: {exc}"}

    def score(self, signals: dict[str, Any]) -> float:
        return sig.combine(signals, self.app.get("score_weights") or {})[0]

    # -- one image ----------------------------------------------------------

    async def caption_image(self, image: dict[str, Any], *, run_id: int | None = None, snapshot_id: int | None = None,
                            on_delta: Callable[[str], Awaitable[None]] | None = None,
                            on_stage: Callable[[str], Awaitable[None]] | None = None,
                            extra: str | None = None, write: bool = True) -> dict[str, Any]:
        """Run the whole pipeline for one image. The caption is written the moment it finishes."""
        async def stage(name: str) -> None:
            if on_stage:
                await on_stage(name)

        started = time.monotonic()
        image_url = await asyncio.to_thread(self.image_url, image)
        await stage("look")
        fields, reply = await self.look(image, image_url, on_delta, extra)
        await stage("compose")
        note = extra if extra is not None else (image.get("note") or "")
        caption, _ = await self.compose(fields, note)
        seconds = time.monotonic() - started
        signals = self.safe_signals(image, fields, reply, None)
        result = {
            "fields": fields, "caption": caption, "signals": signals, "score": self.score(signals),
            "seconds": seconds, "check_seconds": 0.0, "tokens": reply.completion_tokens,
            "look_seconds": reply.seconds, "raw": reply.raw,
        }
        if write:
            # The caption is done the moment it is written. Whatever happens in the checks
            # below can add flags to it but can never turn it back into a failure.
            self.project.write_caption(image, caption, "overwritten by a run")
            self.project.store.update_image(
                image["id"], status="done", error=None, fields=fields, caption=caption, signals=signals,
                score=result["score"], reviewed=0, seconds=seconds, check_seconds=0.0,
                raw=reply.raw, snapshot_id=snapshot_id, run_id=run_id, note=note,
            )

        mode = self.snapshot.get("field_checks", "off")
        flagged = bool(signals["uncertain"]["items"] or signals["tags"]["items"])
        if mode == "all" or (mode == "flagged" and flagged):
            await stage("check")
            try:
                checks = await self.check(image_url, fields)
            except ModelServerError as exc:
                signals["checks"] = {"available": False, "items": [], "values": {}, "error": str(exc)}
            else:
                signals = self.safe_signals(image, fields, reply, checks)
                result.update(signals=signals, score=self.score(signals), check_seconds=checks["seconds"])
            if write:
                self.project.store.update_image(image["id"], signals=signals, score=result["score"],
                                                check_seconds=result["check_seconds"])
        return result

    async def recompose(self, image: dict[str, Any], fields: dict[str, str] | None = None, *,
                        reason: str = "recomposed", snapshot_id: int | None = None, run_id: int | None = None) -> str:
        """Rewrite the caption from stored (or edited) fields without looking at the image again."""
        fields = fields or image.get("fields")
        if not fields:
            raise CaptionError("This image has no stored fields to compose from.")
        caption, _ = await self.compose(fields, image.get("note") or "")
        self.project.write_caption(image, caption, reason)
        signals = self.signals_for(image, fields, None, None, image.get("signals"))
        values = {"fields": fields, "caption": caption, "signals": signals, "score": self.score(signals), "status": "done", "error": None}
        if snapshot_id:
            values.update(snapshot_id=snapshot_id, run_id=run_id)
        self.project.store.update_image(image["id"], **values)
        return caption

    # -- review chat and test bench -----------------------------------------

    async def chat(self, image: dict[str, Any], fields: dict[str, str], caption: str,
                   conversation: list[dict[str, str]]) -> dict[str, Any]:
        """One turn of review chat. The conversation holds only this image and the user's messages."""
        form = "\n".join(f"- {f['key']} ({f['name']}): {f['holds']}" for f in self.prefill["fields"])
        opening = f"Current fields:\n{json.dumps(fields, ensure_ascii=False, indent=1)}\n\nCurrent caption: {caption}"
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": CHAT_SYSTEM + "\n\nFields:\n" + form},
            {"role": "user", "content": [image_part(await asyncio.to_thread(self.image_url, image)), text_part(opening)]},
            {"role": "assistant", "content": "Understood. Tell me what to fix."},
        ]
        for turn in conversation:
            messages.append({"role": turn["role"], "content": turn["content"]})
        reply = await self.client.complete(self.snapshot["model"], messages, sampling=self.snapshot["sampling"],
                                           max_tokens=fields_limit(self.snapshot),
                                           schema=("chat_fields", field_schema(self.prefill)), think=True)
        new_fields = parse_fields(reply.text, self.prefill)
        if new_fields is None:
            raise CaptionError("The model's reply was not a usable set of fields. Try rephrasing.")
        new_caption, _ = await self.compose(new_fields, image.get("note") or "")
        return {"fields": new_fields, "caption": new_caption, "assistant": reply.text}

    async def score_facts(self, caption: str, facts: list[str]) -> list[str]:
        if not facts:
            return []
        listing = "\n".join(f"{n}. {fact}" for n, fact in enumerate(facts, 1))
        messages = [{"role": "system", "content": FACTS_SYSTEM},
                    {"role": "user", "content": f"Caption: {caption}\n\nFacts:\n{listing}\n\nReply as JSON with one verdict per fact, in order."}]
        sampling = dict(self.snapshot["sampling"], temperature=0.0)
        reply = await self.client.complete(self.snapshot["verifier_model"], messages, sampling=sampling,
                                           max_tokens=30 + 8 * len(facts), schema=("fact_scores", FACTS_SCHEMA))
        try:
            verdicts = json.loads(reply.text)["verdicts"]
        except (json.JSONDecodeError, KeyError, TypeError):
            verdicts = []
        verdicts = [v if v in ("captured", "wrong", "missing") else "missing" for v in verdicts]
        return (verdicts + ["missing"] * len(facts))[:len(facts)]


def refresh_template(project: Project, app_settings: dict[str, Any]) -> None:
    """Recompute template repetition across the dataset and each image's combined score."""
    images = [i for i in project.store.images() if i["status"] == "done" and i["signals"] is not None]
    captions = {i["id"]: project.read_caption(i) for i in images}
    repeated = sig.template_repetition(captions)
    weights = app_settings.get("score_weights") or {}
    for image in images:
        signals = image["signals"]
        items = [repeated[image["id"]]] if image["id"] in repeated else []
        if (signals.get("template") or {}).get("items") != items:
            signals["template"] = {"items": items}
            project.store.update_image(image["id"], signals=signals, score=sig.combine(signals, weights)[0])


__all__ = ["Captioner", "CaptionError", "ModelServerError", "build_snapshot", "refresh_template"]
