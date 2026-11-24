"""Prefills, caption fields, sampling presets and the requests built from them.

Fields are data, not code: each prefill stores its own field list and the JSON
schema is built from it, so any field can be renamed, removed or added.
Nudity handling lives only in prefill text; nothing here is specific to it.
"""

from __future__ import annotations

import copy
import json
import re
from typing import Any

from .client import image_part, text_part
from .config import load_json, save_json

# key, name, what it holds
BASE_FIELDS = [
    ("quality", "Quality", "finish (rough sketch or finished piece) and any defect that is plain to see, "
                           "such as a watermark, signature, heavy compression blocks or blur; say \"no visible defects\" when there are none"),
    ("style", "Style and medium", "medium (digital, pencil, paint, 3D, photo), line work, colour, shading"),
    ("camera", "Camera and framing", "close-up, upper body or full body; angle from above, below or behind; depth of field"),
    ("subjects", "Subjects", "how many and what: solo, two girls and a cat"),
    ("appearance", "Appearance", "hair, eyes, skin, body, ears, tail, wings"),
    ("clothing", "Clothing and accessories", "outfit and colours"),
    ("expression", "Expression and gaze", "facial expression, where they are looking"),
    ("action", "Action and pose", "what they are doing, pose, held objects"),
    ("setting", "Setting and background", "place, props, simple or detailed background"),
    ("lighting", "Lighting and time", "light sources, time of day, weather"),
    ("text", "Visible text", "exact transcription of any visible text, in quotes, line breaks kept"),
    ("rating", "Content rating", "one of: safe, suggestive, explicit"),
]


def _fields(overrides: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for key, name, holds in BASE_FIELDS:
        item = {"key": key, "name": name, "holds": holds, "in_caption": True, "caption_note": ""}
        if key == "rating":
            item["in_caption"] = False
            item["choices"] = ["safe", "suggestive", "explicit"]
        item.update(overrides.get(key, {}))
        out.append(item)
    return out


LOOK_PROMPT = (
    "You are describing one image for an image-model training set. Fill in every field of the form "
    "with short, plain, factual notes about what is visible. Do not write a caption yet."
)

COMMON_RULES = [
    "Describe only what is visible. Never guess names, series, artists or intent.",
    "The tags come from an automatic tagger and can be wrong. When a tag and the image disagree, the image wins.",
    "If a field has nothing to report, write \"none\".",
    "Quality records finish and defects as facts. Never write praise such as \"masterpiece\" or \"high quality\".",
    "Report a defect only when you can point to where it is in the image. Soft focus, fog, depth of field, film grain "
    "and dithered shading are part of the picture, not defects.",
    "Transcribe visible text exactly, in quotes, keeping line breaks.",
    "Describe nudity plainly and factually, with no euphemism, judgement or qualifiers such as \"artistic\" "
    "or \"tasteful\", and never explain it by art style, period or medium.",
]

COMPOSE_PROMPT = (
    "Turn the notes into one paragraph that describes the image, {min_words} to {max_words} words long. "
    "Follow the order of the notes. Use plain declarative sentences. Use only what the notes state, and keep "
    "their concrete details (colours, counts, objects, expressions, what is where); when space is short, cut "
    "filler words before you cut a fact. "
    "skip any note that says \"none\". Do not mention notes, fields or tags. Reply with the paragraph only."
)

COMMON_TOGGLES = [
    {"id": "use_trigger", "label": "Refer to the subject as the trigger word", "line": "Refer to the character as {trigger}.", "stage": "compose", "on": False},
    {"id": "lighting", "label": "Include lighting", "line": "Include the lighting.", "stage": "compose", "on": True},
    {"id": "camera", "label": "Include the camera angle", "line": "Include the camera angle and framing.", "stage": "compose", "on": True},
    {"id": "watermarks", "label": "Mention watermarks", "line": "Mention any watermark or signature.", "stage": "compose", "on": True},
    {"id": "no_preamble", "label": "Avoid phrases like \"This image shows\"", "line": "Never open with phrases like \"This image shows\" or \"The image depicts\".", "stage": "compose", "on": True},
    {"id": "no_mood", "label": "Leave out mood", "line": "Leave out mood and atmosphere words.", "stage": "compose", "on": False},
]

COMMON_BANNED = ["artistic", "tasteful", "tastefully", "masterpiece", "best quality", "high quality", "stunning", "breathtaking", "this image shows", "the image depicts"]


def _toggles(on: set[str], line_overrides: dict[str, str] | None = None) -> list[dict[str, Any]]:
    toggles = copy.deepcopy(COMMON_TOGGLES)
    for toggle in toggles:
        if toggle["id"] in on:
            toggle["on"] = True
        if line_overrides and toggle["id"] in line_overrides:
            toggle["line"] = line_overrides[toggle["id"]]
    return toggles


DEFAULT_PREFILLS: list[dict[str, Any]] = [
    {
        "id": "style", "name": "Style", "builtin": True,
        "summary": "Describes everything the image shows and never names the style, so the LoRA absorbs it.",
        "look_prompt": LOOK_PROMPT,
        "compose_prompt": COMPOSE_PROMPT,
        "rules": list(COMMON_RULES),
        "caption_rules": ["Never mention the art style, medium, line work, colouring or rendering."],
        "fields": _fields({
            "quality": {"caption_note": "mention defects only; skip this note if it reports none"},
            "style": {"in_caption": False},
        }),
        "toggles": _toggles(set(), {"use_trigger": "Begin the caption with {trigger}."}),
        "banned_words": COMMON_BANNED + ["art style", "in the style of", "anime style", "stylized", "stylised"],
        "preset": "Balanced", "min_words": 40, "max_words": 70,
    },
    {
        "id": "character", "name": "Character", "builtin": True,
        "summary": "Describes everything including style, but leaves the character's fixed traits to the trigger word.",
        "look_prompt": LOOK_PROMPT,
        "compose_prompt": COMPOSE_PROMPT,
        "rules": list(COMMON_RULES),
        "caption_rules": ["Do not describe the character's fixed traits (face, hair, eye colour, body). "
                          "Only mention appearance that differs from their usual look."],
        "fields": _fields({
            "appearance": {"caption_note": "only changes from the character's usual look; skip the fixed traits"},
        }),
        "toggles": _toggles({"use_trigger"}),
        "banned_words": list(COMMON_BANNED),
        "preset": "Balanced", "min_words": 40, "max_words": 70,
    },
    {
        "id": "concept", "name": "Concept", "builtin": True,
        "summary": "Describes everything including style, but leaves the concept itself to the trigger word. "
                   "Untick the field that holds your concept, such as Action and pose for a pose.",
        "look_prompt": LOOK_PROMPT,
        "compose_prompt": COMPOSE_PROMPT,
        "rules": list(COMMON_RULES),
        "caption_rules": ["Name the concept only by its trigger word and do not describe it further."],
        "fields": _fields({}),
        "toggles": _toggles({"use_trigger"}, {"use_trigger": "Name the concept as {trigger}."}),
        "banned_words": list(COMMON_BANNED),
        "preset": "Balanced", "min_words": 40, "max_words": 70,
    },
]

# Starting points to tune on the test bench, not measured values.
DEFAULT_SAMPLING = {
    "temperature": 0.3, "top_p": 0.95, "top_k": 40, "min_p": 0.05, "repeat_penalty": 1.05,
    "max_tokens_fields": 800, "max_tokens_compose": 200, "seed": 42, "thinking": False, "long_side": 1536,
}

DEFAULT_PRESETS: list[dict[str, Any]] = [
    {"name": "Faithful", "builtin": True, "sampling": {"temperature": 0.15, "top_p": 0.9, "top_k": 40, "min_p": 0.05, "repeat_penalty": 1.05}},
    {"name": "Balanced", "builtin": True, "sampling": {"temperature": 0.3, "top_p": 0.95, "top_k": 40, "min_p": 0.05, "repeat_penalty": 1.05}},
    {"name": "Varied", "builtin": True, "sampling": {"temperature": 0.7, "top_p": 0.95, "top_k": 60, "min_p": 0.03, "repeat_penalty": 1.05}},
]


# -- storage ---------------------------------------------------------------

def load_prefills() -> list[dict[str, Any]]:
    saved = load_json("prefills.json", None)
    if not saved:
        return copy.deepcopy(DEFAULT_PREFILLS)
    ids = {p["id"] for p in saved}
    return saved + [copy.deepcopy(p) for p in DEFAULT_PREFILLS if p["id"] not in ids]


def save_prefills(prefills: list[dict[str, Any]]) -> None:
    save_json("prefills.json", prefills)


def get_prefill(prefill_id: str, prefills: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    prefills = prefills if prefills is not None else load_prefills()
    for prefill in prefills:
        if prefill["id"] == prefill_id:
            return prefill
    return prefills[0]


def default_prefill(prefill_id: str) -> dict[str, Any] | None:
    for prefill in DEFAULT_PREFILLS:
        if prefill["id"] == prefill_id:
            return copy.deepcopy(prefill)
    return None


def load_presets() -> list[dict[str, Any]]:
    saved = load_json("presets.json", None)
    custom = [p for p in (saved or []) if not p.get("builtin")]
    return copy.deepcopy(DEFAULT_PRESETS) + custom


def save_presets(presets: list[dict[str, Any]]) -> None:
    save_json("presets.json", [p for p in presets if not p.get("builtin")])


# -- building requests -----------------------------------------------------

def fill(text: str, values: dict[str, Any]) -> str:
    """Replace {placeholders}; unknown ones are left alone."""
    def swap(match: re.Match) -> str:
        key = match.group(1)
        return str(values[key]) if key in values else match.group(0)
    return re.sub(r"\{(\w+)\}", swap, text)


def _lines(snapshot: dict[str, Any], stage: str, values: dict[str, Any]) -> list[str]:
    """Rules, switched-on toggles and extra instructions, in layer order."""
    prefill = snapshot["prefill"]
    lines = list(prefill.get("rules") or [])
    if stage == "compose":
        lines += list(prefill.get("caption_rules") or [])
    for toggle in prefill.get("toggles") or []:
        if toggle.get("on") and toggle.get("stage", "compose") in (stage, "both"):
            lines.append(toggle["line"])
    extra = (snapshot.get("extra_instructions") or "").strip()
    if extra:
        lines.extend(line.strip() for line in extra.splitlines() if line.strip())
    out = []
    for line in lines:
        # A line that needs a trigger word says nothing useful without one.
        if "{trigger}" in line and not values.get("trigger"):
            continue
        out.append(fill(line, values))
    return out


def field_schema(prefill: dict[str, Any]) -> dict[str, Any]:
    props = {}
    for item in prefill["fields"]:
        prop: dict[str, Any] = {"type": "string", "description": item.get("holds", "")}
        if item.get("choices"):
            prop["enum"] = list(item["choices"])
        props[item["key"]] = prop
    return {"type": "object", "properties": props, "required": list(props), "additionalProperties": False}


def placeholder_values(snapshot: dict[str, Any], tags: str = "", filename: str = "") -> dict[str, Any]:
    prefill = snapshot["prefill"]
    return {
        "tags": tags, "trigger": (snapshot.get("trigger") or "").strip(), "filename": filename,
        "min_words": prefill.get("min_words", 40), "max_words": prefill.get("max_words", 70),
    }


def look_messages(snapshot: dict[str, Any], image_url: str, tags: str, filename: str,
                  http_url: str | None = None) -> list[dict[str, Any]]:
    """Fixed text first and the image last, so the server can reuse the shared opening."""
    prefill = snapshot["prefill"]
    values = placeholder_values(snapshot, tags, filename)
    form = "\n".join(f"- {f['key']} ({f['name']}): {f['holds']}" for f in prefill["fields"])
    system = "\n\n".join([
        fill(prefill["look_prompt"], values),
        "Rules:\n" + "\n".join(f"- {line}" for line in _lines(snapshot, "look", values)),
        "Fields:\n" + form,
        "Reply with one JSON object that has exactly these keys, each a string.",
    ])
    tag_text = f"Tags from the tagger (hints, may be wrong): {tags}" if tags else "No tags are available for this image."
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": [text_part(tag_text), image_part(image_url, http_url)]},
    ]


def caption_fields(prefill: dict[str, Any], fields: dict[str, str]) -> list[tuple[dict[str, Any], str]]:
    """The fields this prefill lets into the caption, in its order, skipping empty ones."""
    out = []
    for item in prefill["fields"]:
        value = str(fields.get(item["key"], "") or "").strip()
        if item.get("in_caption") and value and value.lower().strip(". ") not in ("none", "n/a", "nothing"):
            out.append((item, value))
    return out


def compose_messages(snapshot: dict[str, Any], fields: dict[str, str], avoid: list[str] | None = None) -> list[dict[str, Any]]:
    prefill = snapshot["prefill"]
    values = placeholder_values(snapshot)
    banned = [w for w in prefill.get("banned_words") or [] if w.strip()]
    parts = [
        fill(prefill["compose_prompt"], values),
        "Rules:\n" + "\n".join(f"- {line}" for line in _lines(snapshot, "compose", values)),
    ]
    if banned:
        parts.append("Never use these words or phrases: " + ", ".join(banned) + ".")
    if avoid:
        parts.append("Your last attempt used banned wording. Rewrite without: " + ", ".join(avoid) + ".")
    notes = []
    for item, value in caption_fields(prefill, fields):
        label = item["name"] + (f" ({item['caption_note']})" if item.get("caption_note") else "")
        notes.append(f"{label}: {value}")
    return [
        {"role": "system", "content": "\n\n".join(parts)},
        {"role": "user", "content": "Notes:\n" + "\n".join(notes)},
    ]


def find_banned(text: str, banned: list[str]) -> list[str]:
    hits = []
    for word in banned or []:
        word = word.strip()
        if word and re.search(r"(?<!\w)" + re.escape(word) + r"(?!\w)", text, re.I):
            hits.append(word)
    return hits


def parse_fields(text: str, prefill: dict[str, Any]) -> dict[str, str] | None:
    """Parse the model's field reply. None means malformed or empty."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```\w*\s*|\s*```$", "", text)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            data = json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            return None
    if not isinstance(data, dict):
        return None
    out, filled = {}, 0
    for item in prefill["fields"]:
        value = data.get(item["key"], data.get(item["name"]))
        if isinstance(value, (list, tuple)):
            value = ", ".join(str(v) for v in value)
        value = "" if value is None else str(value).strip()
        filled += bool(value)
        out[item["key"]] = value or "none"
    # Fewer than half the fields filled in counts as an empty reply.
    return out if filled * 2 >= len(prefill["fields"]) else None
