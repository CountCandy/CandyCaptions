import json
import math

import pytest

from candycaption import signals as sig
from candycaption.backends import sampling_body
from candycaption.filters import FilterError, compile_filter
from candycaption.prefills import (DEFAULT_PREFILLS, compose_messages, field_schema, get_prefill, load_prefills,
                                   look_messages, parse_fields)
from candycaption.wordlists import DEFAULT_WORDLISTS

NAMES = {"appearance": "Appearance", "clothing": "Clothing and accessories", "subjects": "Subjects"}


def token(text, p=0.98, alts=()):
    top = [{"token": text, "logprob": math.log(p)}] + [{"token": t, "logprob": math.log(q)} for t, q in alts]
    return {"token": text, "logprob": math.log(p), "top": top}


def test_uncertain_words_only_flag_a_change_of_meaning():
    raw = '{"clothing": "a red scarf", "appearance": "crimson hair"}'
    tokens = [token('{"'), token("clothing"), token('": "'), token("a"),
              token(" red", 0.5, [(" orange", 0.4), (" crimson", 0.05)]), token(" scarf"), token('", "'),
              token("appearance"), token('": "'), token("crimson", 0.5, [("red", 0.45)]), token(" hair"), token('"}')]
    items = sig.uncertain_words(raw, tokens, NAMES, DEFAULT_WORDLISTS)
    # red/orange differ in meaning; crimson/red are two wordings of one fact.
    assert [(i["field"], i["word"], i["alts"][0]["word"]) for i in items] == [("clothing", "red", "orange")]
    assert "split between red (50%) and orange (40%)" in items[0]["message"]
    assert sig.uncertain_words(raw, None, NAMES, DEFAULT_WORDLISTS) == []


def test_uncertain_words_ignore_field_names_and_quiet_alternatives():
    raw = '{"clothing": "red"}'
    tokens = [token('{"'), token("clothing"), token('": "'), token("red", 0.9, [("blue", 0.05)]), token('"}')]
    assert sig.uncertain_words(raw, tokens, NAMES, DEFAULT_WORDLISTS) == []


def test_tag_contradictions_but_not_omissions():
    tags = sig.parse_tags("1girl, red hair, blue_eyes, white_shirt")
    assert tags == ["1girl", "red_hair", "blue_eyes", "white_shirt"]
    agree = {"appearance": "long crimson hair, blue eyes", "clothing": "a coat", "subjects": "one girl"}
    assert sig.tag_contradictions(agree, tags, NAMES, DEFAULT_WORDLISTS) == []
    clash = {"appearance": "blonde hair and eyes that are green", "clothing": "a black shirt", "subjects": "two girls"}
    found = {(i["field"], i["tag"]) for i in sig.tag_contradictions(clash, tags, NAMES, DEFAULT_WORDLISTS)}
    assert found == {("appearance", "red_hair"), ("appearance", "blue_eyes"), ("clothing", "white_shirt"), ("subjects", "1girl")}


def test_two_hair_color_tags_are_satisfied_by_either():
    fields = {"appearance": "one with red hair, one with blue hair", "subjects": "two girls"}
    assert sig.tag_contradictions(fields, ["2girls", "red_hair", "blue_hair"], NAMES, DEFAULT_WORDLISTS) == []


def test_template_repetition():
    shared = "a girl stands in a field of flowers under a bright sky"
    captions = {i: f"{shared} holding item number {i}" for i in range(6)}
    captions[99] = "two foxes sleep beside a quiet river at dusk while snow falls on distant hills"
    flagged = sig.template_repetition(captions)
    assert set(flagged) == set(range(6))
    assert flagged[0]["count"] == 6


def test_yes_probability_measured_and_fallback():
    p, measured = sig.yes_probability("yes", [token("yes", 0.7, [("no", 0.2), ("maybe", 0.1)])])
    assert measured and p == pytest.approx(0.7 / 0.9)
    assert sig.yes_probability("No.", None) == (0.0, False)
    assert sig.yes_probability("Yes, it is", None) == (1.0, False)


def test_combine_is_a_weighted_sum_sorted_worst_first():
    signals = {"uncertain": {"items": [{"field": "clothing", "message": "a", "score": 0.5}]},
               "tags": {"items": [{"field": "appearance", "message": "b", "score": 1.0}]}}
    score, reasons = sig.combine(signals, {"uncertain": 0.3, "tags": 0.2})
    assert score == pytest.approx(0.35) and [r["signal"] for r in reasons] == ["tags", "uncertain"]


def test_value_spans_survive_escapes_and_truncation():
    raw = '{"a": "say \\"hi\\"", "n": 3, "b": "cut off'
    spans = sig.value_spans(raw)
    assert raw[spans["a"][0]:spans["a"][1]] == 'say \\"hi\\"'
    assert "b" not in spans


ROW = {"caption": "A girl with a red scarf", "rel": "sub/cat_01.png", "tags": "1girl, scarf", "words": 6, "tokens": 600,
       "score": 0.4, "reviewed": False, "status": "done", "fields": {"rating": "safe"}, "health": [{"check": "transparency"}],
       "reasons": [{"signal": "tags", "field": "clothing", "field_name": "Clothing and accessories"}]}


@pytest.mark.parametrize("query,expected", [
    ("", True), ("scarf", True), ("dress", False), ("signal:tags", True), ("signal:uncertain", False),
    ("signal:tags AND NOT field:clothing", False), ("field:accessories", True), ('caption:"red scarf"', True),
    ("file:cat tokens>512", True), ("words>70 OR tokens>=600", True), ("NOT (file:dog OR reviewed:yes)", True),
    ("rating:safe health:transparency", True), ("signal:any status:failed", False), ("tags:1girl score<0.5", True),
])
def test_filters(query, expected):
    assert compile_filter(query)(ROW) is expected


@pytest.mark.parametrize("query", ["signal:nonsense", "words>many", "(file:cat", "color:red", "AND"])
def test_bad_filters_explain_themselves(query):
    with pytest.raises(FilterError):
        compile_filter(query)


def snapshot(prefill_id, **extra):
    base = {"prefill": get_prefill(prefill_id, load_prefills()), "extra_instructions": "", "trigger": "", "sampling": {}}
    base.update(extra)
    return base


FIELDS = {"quality": "finished piece, watermark", "style": "watercolor", "camera": "close-up", "subjects": "one girl",
          "appearance": "silver hair", "clothing": "none", "expression": "smiling", "action": "waving",
          "setting": "a beach", "lighting": "sunset", "text": "none", "rating": "safe"}


def test_style_prefill_leaves_style_out_and_character_keeps_it():
    style = compose_messages(snapshot("style"), FIELDS)[1]["content"]
    assert "watercolor" not in style and "Content rating" not in style and "Clothing" not in style
    assert "Quality (mention defects only" in style
    character = compose_messages(snapshot("character", trigger="kyarine"), FIELDS)
    assert "Style and medium: watercolor" in character[1]["content"]
    assert "Refer to the character as kyarine." in character[0]["content"]
    assert "only changes from the character's usual look" in character[1]["content"]


def test_layers_placeholders_and_request_order():
    snap = snapshot("style", extra_instructions="The creature is a fox spirit, never a dog.\nFile: {filename}")
    system, user = look_messages(snap, "data:image/jpeg;base64,AAAA", "1girl, fox", "a/fox.png")
    assert "fox spirit" in system["content"] and "File: a/fox.png" in system["content"]
    assert "{trigger}" not in system["content"]          # trigger lines vanish when there is no trigger
    assert "never mention the art style" not in system["content"].lower()   # caption-only rules stay out of the look
    assert [part["type"] for part in user["content"]] == ["text", "image_url"]  # fixed text first, image last
    compose = compose_messages(snap, FIELDS, avoid=["tasteful"])[0]["content"]
    assert "40 to 70 words" in compose and "Rewrite without: tasteful" in compose and "masterpiece" in compose


def test_nudity_lives_only_in_prefill_text():
    import pathlib

    import candycaption

    for prefill in DEFAULT_PREFILLS:
        assert any("nudity" in rule.lower() for rule in prefill["rules"])
    for source in pathlib.Path(candycaption.__file__).parent.glob("*.py"):
        if source.name != "prefills.py":
            assert "nudity" not in source.read_text(encoding="utf-8").lower(), source.name


def test_fields_are_data_and_drive_the_schema():
    prefill = json.loads(json.dumps(get_prefill("style", load_prefills())))
    prefill["fields"] = [f for f in prefill["fields"] if f["key"] != "rating"] + [
        {"key": "mood", "name": "Mood", "holds": "overall mood", "in_caption": True, "caption_note": ""}]
    schema = field_schema(prefill)
    assert "rating" not in schema["properties"] and schema["required"][-1] == "mood"
    assert parse_fields('```json\n{"quality": "x", "style": "y", "camera": "z", "subjects": "s", "appearance": "a", "clothing": ["hat", "scarf"]}\n```', prefill)["clothing"] == "hat, scarf"
    assert parse_fields('{"quality": "only one"}', prefill) is None
    assert parse_fields("not json", prefill) is None


def test_sampling_never_sends_what_a_backend_ignores():
    sampling = {"temperature": 0.3, "top_p": 0.95, "top_k": 40, "min_p": 0.05, "repeat_penalty": 1.05, "seed": 42, "thinking": False}
    lm = sampling_body("lmstudio", sampling, 800)
    assert "min_p" not in lm and "chat_template_kwargs" not in lm and lm["top_k"] == 40 and lm["max_tokens"] == 800
    assert sampling_body("llamacpp", sampling, 800)["min_p"] == 0.05
    assert sampling_body("llamacpp", sampling, 800)["chat_template_kwargs"] == {"enable_thinking": False}
    assert "repetition_penalty" in sampling_body("vllm", sampling, 800)
    assert set(sampling_body("other", sampling, 800)) == {"temperature", "top_p", "seed", "max_tokens"}


@pytest.mark.parametrize("given,expected", [
    ("http://127.0.0.1:1234", "http://127.0.0.1:1234/v1"), ("http://localhost:1234/", "http://localhost:1234/v1"),
    ("localhost:1234", "http://localhost:1234/v1"), ("http://localhost:1234/v1/", "http://localhost:1234/v1"),
    ("https://example.com/api/v1", "https://example.com/api/v1"), ("", ""),
])
def test_a_bare_server_address_gets_v1(given, expected):
    from candycaption.client import normalize_base_url

    assert normalize_base_url(given) == expected


def test_a_choice_of_what_to_mention_first_is_not_flagged():
    # "green tunic ... brown belt": brown comes later in the same field, so green/brown was an ordering choice.
    raw = '{"clothing": "green tunic with brown belt, green boots", "appearance": "red hair, red cheeks"}'
    tokens = [token('{"'), token("clothing"), token('": "'), token("green", 0.74, [("brown", 0.18 + 0.1)]), token(" tunic"),
              token(" with"), token(" brown"), token(" belt"), token(","), token(" green"), token(" boots"), token('", "'),
              token("appearance"), token('": "'), token("red"), token(" hair"), token(","),
              token(" red", 0.49, [(" orange", 0.36)]), token(" cheeks"), token('"}')]
    items = sig.uncertain_words(raw, tokens, NAMES, DEFAULT_WORDLISTS, min_alt=0.25)
    assert [(i["field"], i["word"], i["nth"]) for i in items] == [("appearance", "red", 1)]   # the second "red", not the first


def test_long_captions_get_room_and_are_never_left_mid_sentence():
    from candycaption.pipeline import clean_caption

    cut = "Two figures stand in a clearing. The camera shows the full body of both"
    assert clean_caption(cut, cut_off=True) == "Two figures stand in a clearing."
    assert clean_caption(cut) == cut
    assert clean_caption("One sentence only that never ends", cut_off=True) == "One sentence only that never ends"


def test_old_settings_get_the_stricter_uncertain_word_default():
    from candycaption.config import load_settings, save_json

    save_json("settings.json", {"uncertain_min_alt": 0.15, "model": "m"})
    assert load_settings()["uncertain_min_alt"] == 0.25
    save_json("settings.json", {"uncertain_min_alt": 0.15, "settings_version": 2})
    assert load_settings()["uncertain_min_alt"] == 0.15      # a value chosen after the change is kept


def test_saved_recipes_and_word_lists_follow_the_spelling_change():
    from candycaption.config import save_json
    from candycaption.wordlists import load_wordlists

    old = json.loads(json.dumps(DEFAULT_PREFILLS))
    old[0]["rules"].append("Never mention colouring or judgement.")
    save_json("prefills.json", old)
    assert "Never mention coloring or judgment." in load_prefills()[0]["rules"]
    lists = json.loads(json.dumps(DEFAULT_WORDLISTS))
    lists["categories"]["colours"] = lists["categories"].pop("colors")
    lists["tag_map"]["colour_aliases"] = lists["tag_map"].pop("color_aliases")
    save_json("wordlists.json", lists)
    loaded = load_wordlists()
    assert "colors" in loaded["categories"] and "color_aliases" in loaded["tag_map"]
