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
    ("style", "Style and medium", "medium (digital, pencil, paint, 3D, photo), line work, color, shading"),
    ("camera", "Camera and framing", "close-up, upper body or full body; angle from above, below or behind; depth of field"),
    ("subjects", "Subjects", "how many and what: solo, two girls and a cat"),
    ("appearance", "Appearance", "hair, eyes, skin, body, ears, tail, wings"),
    ("clothing", "Clothing and accessories", "outfit and colors"),
    ("expression", "Expression and gaze", "facial expression, where they are looking"),
    ("action", "Action and pose", "what they are doing, pose, held objects"),
    ("setting", "Setting and background", "place, props, simple or detailed background"),
    ("layout", "Layout", "where things are in the frame: what is on the left, right, center, top and bottom, "
                         "what is in front of or behind what, and how the subjects stand in relation to each other"),
    ("details", "Small details", "things easy to miss: jewelry, buttons, patterns, textures and materials, "
                                 "small objects, marks, anything held or worn that was not already listed"),
    ("lighting", "Lighting and time", "light sources, time of day, weather"),
    ("text", "Visible text", "exact transcription of any visible text, in quotes, line breaks kept"),
    ("other", "Anything else", "anything visible that no field above covers; \"none\" if they cover everything"),
    ("rating", "Content rating", "one of: safe, suggestive, explicit"),
]
# Fields added after the first release. A recipe saved before then gets them once, in these places.
ADDED_FIELDS = {"layout": "setting", "details": "layout", "other": "text"}       # new key: the key it follows


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


OLD_LOOK_PROMPT = (
    "You are describing one image for an image-model training set. Fill in every field of the form "
    "with short, plain, factual notes about what is visible. Do not write a caption yet."
)
LOOK_PROMPT = (
    "You are describing one image for an image-model training set. Fill in every field of the form "
    "with plain, factual notes about what is visible. Be thorough: write one to three full sentences for a "
    "field that has something to report, and name concrete things such as colors, counts, materials, "
    "positions and shapes. Look at the whole image, including the edges and the background, before you "
    "answer. Do not repeat the same fact in two fields. Do not write a caption yet."
)

COMMON_RULES = [
    "Write in American English.",
    "Describe only what is visible. Never guess names, series, artists or intent.",
    "The tags come from an automatic tagger and can be wrong. When a tag and the image disagree, the image wins.",
    "If a field has nothing to report, write \"none\".",
    "Quality records finish and defects as facts. Never write praise such as \"masterpiece\" or \"high quality\".",
    "Report a defect only when you can point to where it is in the image. Soft focus, fog, depth of field, film grain "
    "and dithered shading are part of the picture, not defects.",
    "Transcribe visible text exactly, in quotes, keeping line breaks.",
    "Describe nudity plainly and factually, with no euphemism, judgment or qualifiers such as \"artistic\" "
    "or \"tasteful\", and never explain it by art style, period or medium.",
]

OLD_COMPOSE_PROMPT = (
    "Turn the notes into one paragraph that describes the image, {min_words} to {max_words} words long. "
    "Follow the order of the notes. Use plain declarative sentences. Use only what the notes state, and keep "
    "their concrete details (colors, counts, objects, expressions, what is where); when space is short, cut "
    "filler words before you cut a fact. "
    "skip any note that says \"none\". Do not mention notes, fields or tags. End on a complete sentence: if you are "
    "running long, stop one sentence early instead of breaking one off. Reply with the paragraph only."
)
COMPOSE_PROMPT = (
    "Turn the notes into one paragraph that describes the image, {min_words} to {max_words} words long. "
    "Follow the order of the notes. Use plain declarative sentences. Use only what the notes state, and keep "
    "their concrete details (colors, counts, objects, expressions, what is where); when space is short, cut "
    "filler words before you cut a fact. Never pad: if the notes do not fill the length, stop when they are "
    "used up instead of adding general remarks. "
    "Skip any note that says \"none\". Do not mention notes, fields or tags. End on a complete sentence: if you are "
    "running long, stop one sentence early instead of breaking one off. Reply with the paragraph only."
)

# MOSS-Image's own long captions are detailed descriptions; this is that length, not the old medium one.
LONG_WORDS = (110, 160)
OLD_WORDS = (40, 70)
WORDING = 3      # raised whenever the built-in wording changes in a way saved recipes should pick up

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
        "caption_rules": ["Never mention the art style, medium, line work, coloring or rendering."],
        "fields": _fields({
            "quality": {"caption_note": "mention defects only; skip this note if it reports none"},
            "style": {"in_caption": False},
        }),
        "toggles": _toggles(set(), {"use_trigger": "Begin the caption with {trigger}."}),
        # The last four keep a "nothing wrong with it" Quality note from turning into a sentence of the caption.
        "banned_words": COMMON_BANNED + ["art style", "in the style of", "anime style", "stylized", "stylised",
                                         "no visible defects", "no watermark", "no visible watermark", "the piece is finished"],
        "preset": "Balanced", "min_words": LONG_WORDS[0], "max_words": LONG_WORDS[1], "wording": WORDING,
    },
    {
        "id": "character", "name": "Character", "builtin": True,
        "summary": "Describes everything including style, but leaves the character's fixed traits to the trigger word.",
        "look_prompt": LOOK_PROMPT,
        "compose_prompt": COMPOSE_PROMPT,
        "rules": list(COMMON_RULES),
        "caption_rules": ["Do not describe the character's fixed traits (face, hair, eye color, body). "
                          "Only mention appearance that differs from their usual look."],
        "fields": _fields({
            "appearance": {"caption_note": "only changes from the character's usual look; skip the fixed traits"},
        }),
        "toggles": _toggles({"use_trigger"}),
        "banned_words": list(COMMON_BANNED),
        "preset": "Balanced", "min_words": LONG_WORDS[0], "max_words": LONG_WORDS[1], "wording": WORDING,
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
        "preset": "Balanced", "min_words": LONG_WORDS[0], "max_words": LONG_WORDS[1], "wording": WORDING,
    },
]

# Starting points to tune on the test bench, not measured values.
DEFAULT_SAMPLING = {
    "temperature": 0.3, "top_p": 0.95, "top_k": 40, "min_p": 0.05, "repeat_penalty": 1.05,
    "max_tokens_fields": 1800, "max_tokens_compose": 330, "seed": 42, "thinking": False,
    "max_tokens_thinking": 2048, "long_side": 1536,
}

DEFAULT_PRESETS: list[dict[str, Any]] = [
    {"name": "Faithful", "builtin": True, "sampling": {"temperature": 0.15, "top_p": 0.9, "top_k": 40, "min_p": 0.05, "repeat_penalty": 1.05}},
    {"name": "Balanced", "builtin": True, "sampling": {"temperature": 0.3, "top_p": 0.95, "top_k": 40, "min_p": 0.05, "repeat_penalty": 1.05}},
    {"name": "Varied", "builtin": True, "sampling": {"temperature": 0.7, "top_p": 0.95, "top_k": 60, "min_p": 0.03, "repeat_penalty": 1.05}},
]


# -- storage ---------------------------------------------------------------

_SPELLING = [("colour", "color"), ("Colour", "Color"), ("judgement", "judgment"), ("flavour", "flavor")]


def _american(value: Any) -> Any:
    """Recipes saved before the app switched to American spelling get the same change."""
    if isinstance(value, str):
        for old, new in _SPELLING:
            value = value.replace(old, new)
        return value
    if isinstance(value, list):
        return [_american(v) for v in value]
    if isinstance(value, dict):
        return {k: _american(v) for k, v in value.items()}
    return value

def load_prefills() -> list[dict[str, Any]]:
    saved = load_json("prefills.json", None)
    if not saved:
        return copy.deepcopy(DEFAULT_PREFILLS)
    saved = _american(saved)
    for prefill in saved:
        default = default_prefill(prefill.get("id", ""))
        if default is None:
            continue
        upgrade_prefill(prefill, default)
    ids = {p["id"] for p in saved}
    return saved + [copy.deepcopy(p) for p in DEFAULT_PREFILLS if p["id"] not in ids]


def upgrade_prefill(prefill: dict[str, Any], default: dict[str, Any]) -> None:
    """Bring a built-in recipe saved by an older version up to date. Nothing of the user's is removed,
    and anything they changed from the old default is left as they set it."""
    wording = prefill.get("wording", 1)
    if wording < 2:
        # The American English rule, and (Style) the phrases that keep a clean Quality note out of the caption.
        rule = "Write in American English."
        if rule not in prefill.get("rules", []):
            prefill["rules"] = [rule] + list(prefill.get("rules", []))
        have = {w.lower() for w in prefill.get("banned_words", [])}
        prefill["banned_words"] = list(prefill.get("banned_words", [])) + [
            w for w in default["banned_words"] if w.lower() not in have and w.startswith(("no ", "the piece"))]
    if wording < 3:
        # Longer captions: three more fields, fuller notes, and the long length MOSS-Image was trained on.
        fields = prefill.setdefault("fields", [])
        for key, after in ADDED_FIELDS.items():
            if any(f["key"] == key for f in fields):
                continue
            new = next(copy.deepcopy(f) for f in default["fields"] if f["key"] == key)
            at = next((n + 1 for n, f in enumerate(fields) if f["key"] == after), len(fields))
            fields.insert(at, new)
        if prefill.get("look_prompt") == OLD_LOOK_PROMPT:
            prefill["look_prompt"] = LOOK_PROMPT
        if prefill.get("compose_prompt") == OLD_COMPOSE_PROMPT:
            prefill["compose_prompt"] = COMPOSE_PROMPT
        if (prefill.get("min_words"), prefill.get("max_words")) == OLD_WORDS:
            prefill["min_words"], prefill["max_words"] = LONG_WORDS
    prefill["wording"] = WORDING


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


KNOWN_FACTS = (
    "Known facts about this image, from the person who made the dataset: {note}\n"
    "Treat these as true even where the image is hard to read, and use their words for what things and "
    "characters are called. A name given here is not a guess: use it directly, as in \"Link stands\", "
    "not \"a figure known as Link\"."
)


def look_messages(snapshot: dict[str, Any], image_url: str, tags: str, filename: str,
                  http_url: str | None = None, note: str = "") -> list[dict[str, Any]]:
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
    if note.strip():
        tag_text += "\n\n" + KNOWN_FACTS.format(note=note.strip())
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": [text_part(tag_text), image_part(image_url, http_url)]},
    ]


_CLEAN_START = re.compile(r"^(no|not|without|free of|free from|lacks?|none|nothing|n/a)\b")
_CLEAN_WHOLE = re.compile(r"^( the| this| a)?( piece| image| picture| render| digital render| illustration| artwork| work)?"
                          r"( is| looks)?( a)?( finished| complete| completed| clean| polished)*"
                          r"( piece| render| work| image| illustration)?$")
_DEFECT_NOUNS = {"defect", "defects", "watermark", "watermarks", "signature", "signatures", "blur", "artifacts",
                 "jpeg artifacts", "compression artifacts", "compression blocks", "noise", "text", "visible defects"}


def reports_no_defects(value: str) -> bool:
    """True when a Quality note only says the image is finished and clean.

    "finished piece, no visible watermark, signature, or blur" is clean;
    "finished piece, small signature in the corner" is not.
    """
    after_no = False
    for clause in re.split(r"[,;.]|\band\b|\bor\b|\bwith\b", value.lower()):
        clause = clause.strip()
        if not clause:
            continue
        if _CLEAN_START.match(clause):
            after_no = True
        elif after_no and clause in _DEFECT_NOUNS:
            continue                      # still inside the "no watermark, signature, or blur" list
        elif _CLEAN_WHOLE.match(" " + clause):
            after_no = False
        else:
            return False
    return True


def caption_fields(prefill: dict[str, Any], fields: dict[str, str]) -> list[tuple[dict[str, Any], str]]:
    """The fields this prefill lets into the caption, in its order, skipping empty ones."""
    out = []
    for item in prefill["fields"]:
        value = str(fields.get(item["key"], "") or "").strip()
        if not item.get("in_caption") or not value or value.lower().strip(". ") in ("none", "n/a", "nothing"):
            continue
        # A "defects only" field with no defects to report stays out, so the model cannot restate
        # "no watermark, signature, or blur" as a sentence of the caption.
        if "defects only" in (item.get("caption_note") or "").lower() and reports_no_defects(value):
            continue
        out.append((item, value))
    return out


def compose_messages(snapshot: dict[str, Any], fields: dict[str, str], avoid: list[str] | None = None,
                     note: str = "") -> list[dict[str, Any]]:
    prefill = snapshot["prefill"]
    values = placeholder_values(snapshot)
    banned = [w for w in prefill.get("banned_words") or [] if w.strip()]
    parts = [
        fill(prefill["compose_prompt"], values),
        "Rules:\n" + "\n".join(f"- {line}" for line in _lines(snapshot, "compose", values)),
    ]
    if banned:
        parts.append("Never use these words or phrases: " + ", ".join(banned) + ".")
    if note.strip():
        parts.append(KNOWN_FACTS.format(note=note.strip()))
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
