"""Editable word lists and the tag-to-field map used by the review signals.

``categories`` decide when two alternatives the model weighed differ in
meaning: words in the same inner list are wordings of one fact ("red",
"crimson"); words in different lists of one category are different facts
("red", "blue"). Only these categories are compared.
"""

from __future__ import annotations

import copy
from typing import Any

from .config import load_json, save_json

DEFAULT_WORDLISTS: dict[str, Any] = {
    "categories": {
        "colours": [
            ["red", "crimson", "scarlet", "ruby"],
            ["orange", "amber"],
            ["yellow", "golden", "gold", "blonde", "blond"],
            ["green", "emerald", "lime", "olive"],
            ["blue", "navy", "azure", "cobalt"],
            ["aqua", "teal", "turquoise", "cyan"],
            ["purple", "violet", "lavender", "lilac"],
            ["pink", "magenta", "rose"],
            ["brown", "brunette", "chestnut", "auburn", "tan"],
            ["black"],
            ["white"],
            ["grey", "gray", "silver"],
        ],
        "counts": [
            ["one", "1", "single", "solo", "alone"],
            ["two", "2", "pair", "couple", "both"],
            ["three", "3", "trio"],
            ["four", "4"],
            ["five", "5"],
            ["six", "6"],
            ["several", "many", "multiple", "group", "crowd"],
        ],
        "sides": [
            ["left"], ["right"],
            ["above", "over"], ["below", "under", "beneath"],
            ["front"], ["behind", "back"],
        ],
        "objects": [
            ["cup", "mug", "teacup"], ["glass", "bottle"], ["phone", "smartphone", "cellphone"],
            ["book", "notebook"], ["sword", "katana", "blade"], ["gun", "pistol", "rifle"],
            ["umbrella", "parasol"], ["bag", "backpack", "purse"], ["staff", "wand"],
            ["cat", "kitten"], ["dog", "puppy"], ["fox"], ["rabbit", "bunny"], ["bird"], ["horse"],
            ["dress", "gown"], ["skirt"], ["shirt", "blouse"], ["jacket", "coat", "blazer"],
            ["hat", "cap"], ["scarf"], ["gloves"], ["boots"], ["shoes", "sneakers"],
            ["day", "daytime", "daylight"], ["night", "nighttime"], ["sunset", "dusk"], ["sunrise", "dawn"],
            ["indoors", "inside"], ["outdoors", "outside"],
            ["standing"], ["sitting", "seated"], ["lying", "reclining"], ["kneeling"], ["running"], ["walking"],
            ["smiling", "smile", "grinning"], ["frowning", "frown"], ["crying", "tears"],
        ],
    },
    "tag_map": {
        # <colour>_<attribute> tags are checked against the words near these nouns.
        "attributes": {
            "hair": {"field": "appearance", "nouns": ["hair", "ponytail", "twintails", "braid", "braids", "bangs"]},
            "eyes": {"field": "appearance", "nouns": ["eyes", "eye"]},
            "skin": {"field": "appearance", "nouns": ["skin"]},
            "shirt": {"field": "clothing", "nouns": ["shirt", "t-shirt", "blouse"]},
            "dress": {"field": "clothing", "nouns": ["dress", "gown"]},
            "skirt": {"field": "clothing", "nouns": ["skirt"]},
            "jacket": {"field": "clothing", "nouns": ["jacket", "coat", "blazer"]},
            "coat": {"field": "clothing", "nouns": ["coat", "jacket"]},
            "sweater": {"field": "clothing", "nouns": ["sweater", "jumper", "cardigan"]},
            "hoodie": {"field": "clothing", "nouns": ["hoodie"]},
            "pants": {"field": "clothing", "nouns": ["pants", "trousers", "jeans"]},
            "shorts": {"field": "clothing", "nouns": ["shorts"]},
            "thighhighs": {"field": "clothing", "nouns": ["thighhighs", "thigh-highs", "stockings", "socks"]},
            "pantyhose": {"field": "clothing", "nouns": ["pantyhose", "tights"]},
            "gloves": {"field": "clothing", "nouns": ["gloves", "glove"]},
            "footwear": {"field": "clothing", "nouns": ["shoes", "boots", "sneakers", "heels", "sandals"]},
            "headwear": {"field": "clothing", "nouns": ["hat", "cap", "beret", "hood"]},
            "scarf": {"field": "clothing", "nouns": ["scarf"]},
            "ribbon": {"field": "clothing", "nouns": ["ribbon"]},
            "bow": {"field": "clothing", "nouns": ["bow", "bowtie"]},
            "necktie": {"field": "clothing", "nouns": ["tie", "necktie"]},
            "bikini": {"field": "clothing", "nouns": ["bikini"]},
            "swimsuit": {"field": "clothing", "nouns": ["swimsuit"]},
            "kimono": {"field": "clothing", "nouns": ["kimono", "yukata"]},
        },
        # Booru colour words that are not in the colours list under the same spelling.
        "colour_aliases": {"blonde": "yellow", "gray": "grey", "silver": "grey", "aqua": "aqua", "light_brown": "brown",
                           "dark_blue": "blue", "light_blue": "blue", "dark_green": "green", "light_purple": "purple"},
        # Count tags are checked against number words in the Subjects field.
        "counts": {
            "solo": {"noun": "people", "count": 1},
            "1girl": {"noun": "girl", "count": 1}, "2girls": {"noun": "girl", "count": 2},
            "3girls": {"noun": "girl", "count": 3}, "4girls": {"noun": "girl", "count": 4},
            "1boy": {"noun": "boy", "count": 1}, "2boys": {"noun": "boy", "count": 2},
            "3boys": {"noun": "boy", "count": 3},
        },
        "count_field": "subjects",
        "count_nouns": {
            "girl": ["girl", "girls", "woman", "women", "female", "females", "lady", "ladies"],
            "boy": ["boy", "boys", "man", "men", "male", "males", "guy", "guys"],
            "people": ["people", "persons", "figures", "characters", "girls", "boys", "women", "men"],
        },
    },
}


def load_wordlists() -> dict[str, Any]:
    saved = load_json("wordlists.json", None)
    if not isinstance(saved, dict) or "categories" not in saved or "tag_map" not in saved:
        return copy.deepcopy(DEFAULT_WORDLISTS)
    return saved


def save_wordlists(wordlists: dict[str, Any]) -> None:
    save_json("wordlists.json", wordlists)


def word_index(wordlists: dict[str, Any]) -> dict[str, tuple[str, int]]:
    """word -> (category, group number)."""
    index: dict[str, tuple[str, int]] = {}
    for category, groups in (wordlists.get("categories") or {}).items():
        for number, group in enumerate(groups):
            for word in group:
                index.setdefault(str(word).lower(), (category, number))
    return index
