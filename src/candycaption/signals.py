"""Review signals: measured reasons to doubt a caption.

The model is never asked to rate its own confidence. Each signal names the
field in doubt and says why in plain words.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from json.decoder import scanstring
from typing import Any

from .wordlists import word_index

SIGNALS = ["uncertain", "checks", "tags", "template"]
WORD_RE = re.compile(r"[a-z0-9]+(?:[-'][a-z0-9]+)*")
NUMBER_WORDS = {"a": 1, "an": 1, "one": 1, "single": 1, "lone": 1, "solo": 1, "two": 2, "three": 3, "four": 4,
                "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}


# -- uncertain words -------------------------------------------------------

def value_spans(raw: str) -> dict[str, tuple[int, int]]:
    """Where each top-level string value sits inside the raw JSON text."""
    spans: dict[str, tuple[int, int]] = {}
    i = raw.find("{")
    if i < 0:
        return spans
    i += 1
    decoder = json.JSONDecoder()
    try:
        while i < len(raw):
            while i < len(raw) and raw[i] in " \t\r\n,":
                i += 1
            if i >= len(raw) or raw[i] != '"':
                break
            key, i = scanstring(raw, i + 1)
            while i < len(raw) and raw[i] in " \t\r\n:":
                i += 1
            if i < len(raw) and raw[i] == '"':
                _, end = scanstring(raw, i + 1)
                spans[key] = (i + 1, end - 1)
                i = end
            else:
                _, i = decoder.raw_decode(raw, i)
    except (ValueError, IndexError):
        pass
    return spans


def uncertain_words(raw: str, tokens: list[dict[str, Any]] | None, names: dict[str, str],
                    wordlists: dict[str, Any], min_alt: float = 0.15) -> list[dict[str, Any]]:
    """Words where the model weighed an alternative that means something different."""
    if not tokens:
        return []
    index = word_index(wordlists)
    spans = value_spans(raw)
    items, pos = [], 0
    for tok in tokens:
        text = tok.get("token") or ""
        at = raw.find(text, pos) if text else -1
        if at < 0 or at > pos + 16:
            continue
        pos = at + len(text)
        start = at + len(text) - len(text.lstrip())
        key = next((k for k, (a, b) in spans.items() if a <= start < b), None)
        if key is None or key not in names:
            continue
        if start > 0 and raw[start - 1].isalnum():
            continue  # not the first piece of a word
        match = WORD_RE.match(raw.lower(), start)
        if not match:
            continue
        word = match.group(0)
        if word not in index:
            continue
        category, group = index[word]
        a, b = spans[key]
        rest = set(WORD_RE.findall(raw.lower(), match.end(), b))
        alts = []
        for alt in tok.get("top") or []:
            alt_word = alt["token"].strip().lower().strip(".,;:\"'")
            p = math.exp(alt["logprob"])
            if alt_word == word or alt_word not in index or p < min_alt:
                continue
            alt_category, alt_group = index[alt_word]
            if alt_category != category or alt_group == group:
                continue
            # When the alternative turns up later in the same field ("green tunic ... brown belt",
            # "left figure ... right figure"), the model was choosing what to mention first,
            # not wavering about a fact.
            if any(index.get(later) == (alt_category, alt_group) for later in rest):
                continue
            alts.append({"word": alt_word, "p": round(p, 3)})
        if alts:
            p_word = round(math.exp(tok.get("logprob", 0.0)), 3)
            best = max(alts, key=lambda a: a["p"])
            items.append({
                "field": key, "word": word, "p": p_word, "alts": alts, "category": category,
                # Which occurrence of the word in the field this is, so the page marks the right one.
                "nth": len(re.findall(r"(?<![\w-])" + re.escape(word) + r"(?![\w-])", raw.lower()[a:start])),
                "score": min(1.0, 2 * sum(a["p"] for a in alts)),
                "message": f"{names[key]}: model split between {word} ({p_word:.0%}) and {best['word']} ({best['p']:.0%})",
            })
    return items


# -- tag contradictions ----------------------------------------------------

def parse_tags(text: str) -> list[str]:
    tags = []
    for piece in re.split(r"[,\n]", text or ""):
        piece = piece.strip().lower()
        if piece:
            tags.append(re.sub(r"\s+", "_", piece))
    return tags


def _color_group(word: str, wordlists: dict[str, Any], index: dict[str, tuple[str, int]]) -> int | None:
    aliases = wordlists["tag_map"].get("color_aliases") or {}
    word = aliases.get(word, word)
    hit = index.get(word)
    return hit[1] if hit and hit[0] == "colors" else None


def _colors_near(text: str, nouns: list[str], index: dict[str, tuple[str, int]]) -> set[int] | None:
    """Color groups described for any of these nouns; None when no noun is mentioned."""
    found: set[int] = set()
    seen = False
    for segment in re.split(r"[.;,]", text.lower()):
        words = WORD_RE.findall(segment)
        for i, word in enumerate(words):
            if word not in nouns:
                continue
            seen = True
            window = words[max(0, i - 4):i]
            # "hair is red", "eyes are a deep blue"
            if i + 1 < len(words) and words[i + 1] in ("is", "are", "in"):
                window = window + words[i + 2:i + 5]
            for near in window:
                hit = index.get(near)
                if hit and hit[0] == "colors":
                    found.add(hit[1])
    return found if seen else None


def tag_contradictions(fields: dict[str, str], tags: list[str], names: dict[str, str],
                       wordlists: dict[str, Any]) -> list[dict[str, Any]]:
    """A field that disagrees with the tagger. A tag the fields leave out is not flagged."""
    index = word_index(wordlists)
    tag_map = wordlists.get("tag_map") or {}
    attributes = tag_map.get("attributes") or {}
    colors = wordlists["categories"].get("colors") or []
    items = []

    # Several color tags can share one attribute ("red_hair" and "blue_hair" with 2girls).
    wanted: dict[str, dict[int, str]] = {}
    for tag in tags:
        for attr in attributes:
            if tag.endswith("_" + attr):
                group = _color_group(tag[: -len(attr) - 1], wordlists, index)
                if group is not None:
                    wanted.setdefault(attr, {})[group] = tag
    for attr, groups in wanted.items():
        spec = attributes[attr]
        field = spec["field"]
        if field not in names:
            continue
        found = _colors_near(fields.get(field, ""), [n.lower() for n in spec["nouns"]], index)
        if not found or found & set(groups):
            continue
        said = ", ".join(colors[g][0] for g in sorted(found))
        tag = next(iter(groups.values()))
        items.append({"field": field, "tag": tag, "score": 1.0,
                      "message": f"{names[field]}: says {said} {attr}, but the tagger says {tag}"})

    count_field = tag_map.get("count_field", "subjects")
    if count_field in names:
        text = fields.get(count_field, "").lower()
        words = WORD_RE.findall(text)
        for tag in tags:
            spec = (tag_map.get("counts") or {}).get(tag)
            if not spec:
                continue
            nouns = set((tag_map.get("count_nouns") or {}).get(spec["noun"], []))
            said = set()
            for i, word in enumerate(words):
                if word in nouns:
                    for near in reversed(words[max(0, i - 3):i]):
                        number = NUMBER_WORDS.get(near) or (int(near) if near.isdigit() else None)
                        if number:
                            said.add(number)
                            break
            if spec["noun"] == "people":
                wrong = bool(said) and min(said) > spec["count"]
            else:
                wrong = bool(said) and spec["count"] not in said
            if wrong:
                items.append({"field": count_field, "tag": tag, "score": 1.0,
                              "message": f"{names[count_field]}: counts {', '.join(map(str, sorted(said)))}, but the tagger says {tag}"})

        # The tagger sees nobody, and the field counts people.
        people_tag = tag_map.get("people_tag")
        if people_tag and people_tag in tags:
            nouns = {n for group in (tag_map.get("count_nouns") or {}).values() for n in group}
            for i, word in enumerate(words):
                if word in nouns and any(NUMBER_WORDS.get(w) or w.isdigit() for w in words[max(0, i - 3):i]):
                    items.append({"field": count_field, "tag": people_tag, "score": 1.0,
                                  "message": f"{names[count_field]}: counts people, but the tagger says {people_tag}"})
                    break

    several = any(((tag_map.get("counts") or {}).get(t) or {}).get("count", 1) > 1 for t in tags) or any(
        t.startswith("multiple_") for t in tags)
    tagset = set(tags)
    for spec in tag_map.get("opposites") or []:
        field = spec.get("field")
        if field not in names or (spec.get("per_person") and several):
            continue
        text = fields.get(field, "").lower()
        said = [n for n, choice in enumerate(spec["choices"]) if any(_has_phrase(text, p) for p in choice["says"])]
        tagged = [n for n, choice in enumerate(spec["choices"]) if tagset & set(choice["tags"])]
        # A clash needs the tagger to say one thing and the field to say only another.
        if len(tagged) != 1 or not said or tagged[0] in said:
            continue
        tag = next(t for t in spec["choices"][tagged[0]]["tags"] if t in tagset)
        phrase = next(p for p in spec["choices"][said[0]]["says"] if _has_phrase(text, p))
        items.append({"field": field, "tag": tag, "score": 0.8,
                      "message": f"{names[field]}: says \"{phrase}\", but the tagger says {tag}"})

    skip = set(tag_map.get("absence_skip") or []) | set(tag_map.get("counts") or {})
    flagged = {(i["field"], i["tag"]) for i in items}
    for tag in tags:
        if tag in skip or tag.startswith("no_") or len(tag) < 3 or not re.fullmatch(r"[a-z][a-z0-9_ -]*", tag):
            continue
        words_of_tag = r"[\s-]+".join(re.escape(w) for w in tag.split("_"))
        absent = re.compile(r"\b(?:no|without(?:\s+an?|\s+any)?|not\s+wearing(?:\s+an?|\s+any)?|lacks?(?:\s+an?|\s+any)?)\s+"
                            + words_of_tag + r"(?:e?s)?\b")
        for field, name in names.items():
            hit = absent.search(fields.get(field, "").lower())
            if hit and (field, tag) not in flagged:
                items.append({"field": field, "tag": tag, "score": 0.8,
                              "message": f"{name}: says \"{hit.group(0)}\", but the tagger says {tag}"})
                break
    return items


def _has_phrase(text: str, phrase: str) -> bool:
    """The phrase as whole words, not as part of a longer word ("night" is not in "knight")."""
    return re.search(r"(?<![a-z0-9])" + re.escape(phrase) + r"(?![a-z0-9])", text) is not None


# -- template repetition ---------------------------------------------------

def template_repetition(captions: dict[int, str], n: int = 5, share: float = 0.2,
                        flag_at: float = 0.3) -> dict[int, dict[str, Any]]:
    """Captions that repeat long phrases found in many others."""
    grams: dict[int, list[str]] = {}
    spread: Counter[str] = Counter()
    for key, caption in captions.items():
        words = WORD_RE.findall((caption or "").lower())
        mine = [" ".join(words[i:i + n]) for i in range(len(words) - n + 1)]
        grams[key] = mine
        spread.update(set(mine))
    need = max(3, math.ceil(len(captions) * share))
    out = {}
    for key, mine in grams.items():
        if not mine:
            continue
        common = [g for g in mine if spread[g] >= need]
        fraction = len(common) / len(mine)
        if fraction >= flag_at:
            phrase = max(common, key=lambda g: spread[g])
            out[key] = {
                "field": None, "phrase": phrase, "count": spread[phrase], "score": min(1.0, fraction * 1.5),
                "message": f"Repeats phrasing found in {spread[phrase]} captions: \"{phrase}…\"",
            }
    return out


# -- field checks ----------------------------------------------------------

def check_items(values: dict[str, float], names: dict[str, str], threshold: float, by: str) -> list[dict[str, Any]]:
    items = []
    for key, p_yes in values.items():
        if p_yes < threshold and key in names:
            items.append({"field": key, "p_yes": round(p_yes, 3), "score": 1.0 - p_yes,
                          "message": f"{names[key]}: {by} would not confirm it ({p_yes:.0%} yes)"})
    return items


def yes_probability(text: str, tokens: list[dict[str, Any]] | None) -> tuple[float, bool]:
    """(probability of yes, measured). Falls back to a plain yes/no reading."""
    for tok in tokens or []:
        if not tok.get("token", "").strip():
            continue
        yes = no = 0.0
        for alt in tok.get("top") or [tok]:
            word = alt["token"].strip().lower().strip(".,!\"'")
            p = math.exp(alt["logprob"])
            if word in ("yes", "y", "true"):
                yes += p
            elif word in ("no", "n", "false"):
                no += p
        if yes + no > 0:
            return yes / (yes + no), True
        break
    return (1.0 if text.strip().lower().startswith(("yes", "true")) else 0.0), False


# -- combining -------------------------------------------------------------

def combine(signals: dict[str, Any], weights: dict[str, float]) -> tuple[float, list[dict[str, Any]]]:
    """Simple weighted sum, worst item per signal. Returns (score, reasons)."""
    score, reasons = 0.0, []
    for name in SIGNALS:
        items = (signals.get(name) or {}).get("items") or []
        if not items:
            continue
        worst = max(item["score"] for item in items)
        score += weights.get(name, 0.0) * worst
        for item in items:
            reasons.append({"signal": name, "field": item.get("field"), "message": item["message"], "score": item["score"]})
    reasons.sort(key=lambda r: -r["score"])
    return round(score, 4), reasons
