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
        "colors": [
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
            ["gray", "grey", "silver"],
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
        # <color>_<attribute> tags are checked against the words near these nouns.
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
            "legwear": {"field": "clothing", "nouns": ["stockings", "socks", "tights", "leggings", "thighhighs", "thigh-highs"]},
            "socks": {"field": "clothing", "nouns": ["socks", "sock"]},
            "cape": {"field": "clothing", "nouns": ["cape", "cloak"]},
            "vest": {"field": "clothing", "nouns": ["vest", "waistcoat"]},
            "apron": {"field": "clothing", "nouns": ["apron"]},
            "leotard": {"field": "clothing", "nouns": ["leotard"]},
            "bodysuit": {"field": "clothing", "nouns": ["bodysuit", "jumpsuit"]},
            "hairband": {"field": "clothing", "nouns": ["hairband", "headband"]},
            "nails": {"field": "appearance", "nouns": ["nails", "fingernails"]},
            "lips": {"field": "appearance", "nouns": ["lips", "lipstick"]},
            "fur": {"field": "appearance", "nouns": ["fur"]},
            "wings": {"field": "appearance", "nouns": ["wings", "wing"]},
            "tail": {"field": "appearance", "nouns": ["tail"]},
            "horns": {"field": "appearance", "nouns": ["horns", "horn"]},
            "background": {"field": "setting", "nouns": ["background", "backdrop"]},
        },
        # Things that cannot both be true. Each set is a list of choices; a choice has the tags that mean it
        # and the wordings a field would use for it. A clash is a tag from one choice while the field uses
        # only another choice's wording. "per_person" sets are skipped when the tags show several people,
        # since one may be sitting while another stands.
        "opposites": [
            {"name": "hair length", "field": "appearance", "per_person": True, "choices": [
                {"tags": ["short_hair", "very_short_hair"], "says": ["short hair", "short-haired", "cropped hair", "bob cut", "pixie cut"]},
                {"tags": ["long_hair", "very_long_hair", "absurdly_long_hair"], "says": ["long hair", "long-haired", "waist-length hair", "hip-length hair", "floor-length hair"]}]},
            {"name": "pose", "field": "action", "per_person": True, "choices": [
                {"tags": ["sitting", "seiza", "wariza", "indian_style"], "says": ["sitting", "sits", "seated", "sat"]},
                {"tags": ["standing"], "says": ["standing", "stands"]},
                {"tags": ["lying", "on_back", "on_stomach", "on_side"], "says": ["lying", "lies", "reclining", "reclines"]},
                {"tags": ["kneeling"], "says": ["kneeling", "kneels"]}]},
            {"name": "eyes", "field": "expression", "per_person": True, "choices": [
                {"tags": ["closed_eyes"], "says": ["eyes closed", "closed eyes", "eyes are closed", "eyes shut"]},
                {"tags": ["looking_at_viewer"], "says": ["looking at the viewer", "looks at the viewer", "looking at the camera",
                                                         "looks at the camera", "looking directly", "eye contact"]}]},
            {"name": "mouth", "field": "expression", "per_person": True, "choices": [
                {"tags": ["closed_mouth"], "says": ["mouth closed", "closed mouth", "lips closed", "lips pressed"]},
                {"tags": ["open_mouth"], "says": ["mouth open", "open mouth", "mouth is open", "mouth wide open"]}]},
            {"name": "framing", "field": "camera", "choices": [
                {"tags": ["full_body"], "says": ["full body", "full-body", "head to toe"]},
                {"tags": ["upper_body"], "says": ["upper body", "upper-body", "waist up", "from the waist up", "bust shot"]},
                {"tags": ["cowboy_shot"], "says": ["cowboy shot", "thighs up", "from the thighs up", "knees up"]},
                {"tags": ["portrait", "close-up"], "says": ["close-up", "closeup", "close up", "headshot", "head shot"]}]},
            {"name": "height of view", "field": "camera", "choices": [
                {"tags": ["from_above"], "says": ["from above", "high angle", "bird's-eye", "birds-eye", "overhead"]},
                {"tags": ["from_below"], "says": ["from below", "low angle", "worm's-eye"]}]},
            {"name": "side of view", "field": "camera", "choices": [
                {"tags": ["from_behind"], "says": ["from behind", "back view", "rear view", "seen from the back"]},
                {"tags": ["straight-on"], "says": ["from the front", "front view", "straight on", "straight-on", "head-on"]}]},
            {"name": "place", "field": "setting", "choices": [
                {"tags": ["indoors"], "says": ["indoors", "indoor", "interior"]},
                {"tags": ["outdoors"], "says": ["outdoors", "outdoor", "outside"]}]},
            {"name": "time of day", "field": "lighting", "choices": [
                {"tags": ["day", "blue_sky"], "says": ["daylight", "daytime", "midday", "noon", "sunlit", "sunny"]},
                {"tags": ["night", "night_sky", "starry_sky"], "says": ["night", "nighttime", "moonlight", "moonlit", "starlight"]}]},
            {"name": "color", "field": "style", "choices": [
                {"tags": ["monochrome", "greyscale"], "says": ["monochrome", "grayscale", "black and white", "black-and-white"]},
                {"tags": [], "says": ["full color", "full-color", "colorful", "vivid colors", "brightly colored"]}]},
        ],
        # A tag for a thing, while a field says in so many words that the thing is not there ("no hat").
        # Tags that are themselves about absence or are too general to check this way are left out.
        "absence_skip": ["solo", "no_humans", "simple_background", "white_background", "transparent_background",
                         "text", "signature", "background"],
        "people_tag": "no_humans",
        "attributes_version": 2,
        # Booru color words that are not in the colors list under the same spelling.
        "color_aliases": {"blonde": "yellow", "grey": "gray", "silver": "gray", "aqua": "aqua", "light_brown": "brown",
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
    # Lists saved before the app switched to American spelling.
    if "colours" in saved["categories"]:
        saved["categories"]["colors"] = saved["categories"].pop("colours")
    if "colour_aliases" in saved["tag_map"]:
        saved["tag_map"]["color_aliases"] = saved["tag_map"].pop("colour_aliases")
    # Lists saved before Flavor Clash was widened get the new parts. Entries already there are left as they are.
    default = DEFAULT_WORDLISTS["tag_map"]
    for key in ("opposites", "absence_skip", "people_tag"):
        if key not in saved["tag_map"]:
            saved["tag_map"][key] = copy.deepcopy(default[key])
    if saved["tag_map"].get("attributes_version", 1) < 2:
        for name, spec in default["attributes"].items():
            saved["tag_map"].setdefault("attributes", {}).setdefault(name, copy.deepcopy(spec))
        saved["tag_map"]["attributes_version"] = 2
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
