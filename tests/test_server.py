import httpx
import pytest

from candycaption import server
from candycaption.config import load_settings, save_settings

from conftest import make_image


@pytest.fixture
async def api(settings, dataset):
    save_settings(settings)
    server.state.project = None
    server.state.runner = None
    transport = httpx.ASGITransport(app=server.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://app") as client:
        opened = await client.post("/api/dataset/open", json={"path": str(dataset)})
        assert opened.status_code == 200 and len(opened.json()["images"]) == 3
        yield client
    if server.state.project:
        server.state.project.store.close()


async def run_batch(api, **body):
    assert (await api.post("/api/run/start", json=body)).status_code == 200
    await server.state.runner.wait()
    return (await api.get("/api/run")).json()


async def test_connect_test_stores_what_the_server_supports(api):
    result = (await api.post("/api/connect/test")).json()
    assert result["ok"] and result["caps"]["tested"] and result["caps"]["chat_logprobs"]
    assert load_settings()["caps"]["chat_logprobs"] is True
    # Pointing at another server forgets what the old one supported.
    changed = (await api.put("/api/settings", json={"base_url": "http://localhost:1/v1"})).json()
    assert changed["caps"]["tested"] is False


async def test_run_then_review_worst_first(api, dataset):
    status = await run_batch(api)
    assert status["state"] == "done" and status["done"] == 3
    rows = (await api.get("/api/review", params={"sort": "score"})).json()["rows"]
    assert [r["rel"] for r in rows][-1] == "plain.png" and rows[0]["score"] >= rows[1]["score"] > 0
    assert [r["rel"] for r in (await api.get("/api/review")).json()["rows"]] == ["clash.png", "plain.png", "unsure.png"]
    only_tags = (await api.get("/api/review", params={"q": "signal:tags"})).json()["rows"]
    assert [r["rel"] for r in only_tags] == ["clash.png"]
    assert (await api.get("/api/review", params={"q": "signal:bogus"})).status_code == 422

    detail = (await api.get(f"/api/review/{only_tags[0]['id']}")).json()
    assert detail["reasons"][0]["field_name"] == "Appearance" and detail["snapshot"]["prefill"] == "Style"

    # Approve removes it from the queue.
    await api.post(f"/api/review/{detail['id']}/approve", json={})
    assert "clash.png" not in [r["rel"] for r in (await api.get("/api/review")).json()["rows"]]
    assert "clash.png" in [r["rel"] for r in (await api.get("/api/review", params={"q": "reviewed:yes"})).json()["rows"]]


async def test_edits_save_immediately_and_keep_history(api, dataset):
    await run_batch(api)
    row = (await api.get("/api/review", params={"q": "file:clash"})).json()["rows"][0]
    old = (dataset / "clash_nl.txt").read_text().strip()

    edited = (await api.put(f"/api/review/{row['id']}/caption", json={"text": "Two girls   sit together."})).json()
    assert (dataset / "clash_nl.txt").read_text() == "Two girls sit together.\n"
    assert edited["history"][0]["old"] == old

    # Fixing the tag clears the contradiction.
    fixed = (await api.put(f"/api/review/{row['id']}/tags", json={"text": "2girls, blue_hair"})).json()
    assert (dataset / "clash.txt").read_text().strip() == "2girls, blue_hair"
    assert not [r for r in fixed["reasons"] if r["signal"] == "tags"]

    # Editing a field recomposes the description.
    recomposed = (await api.put(f"/api/review/{row['id']}/fields", json={"fields": {"setting": "a rooftop at night"}})).json()
    assert "rooftop at night" in recomposed["caption"].lower()
    assert "rooftop" in (dataset / "clash_nl.txt").read_text().lower()


async def test_review_chat_accept_and_promote_to_rule(api, dataset):
    await run_batch(api)
    row = (await api.get("/api/review", params={"q": "file:unsure"})).json()["rows"][0]
    before = (dataset / "unsure_nl.txt").read_text()
    proposal = (await api.post(f"/api/review/{row['id']}/chat", json={"messages": [{"role": "user", "content": "The scarf is orange, not red."}]})).json()
    assert "orange scarf" in proposal["fields"]["clothing"] and "orange scarf" in proposal["caption"].lower()
    assert (dataset / "unsure_nl.txt").read_text() == before        # nothing is saved until you accept

    accepted = (await api.post(f"/api/review/{row['id']}/chat/accept", json=proposal)).json()
    assert "orange scarf" in (dataset / "unsure_nl.txt").read_text().lower()
    assert accepted["signals"]["checks"]["available"] and "clothing" in accepted["signals"]["checks"]["values"]
    assert not [r for r in accepted["reasons"] if r["signal"] == "uncertain"]   # the wavering word is gone

    rule = "Scarves in this set are orange, never red."
    promoted = (await api.post("/api/rules/promote", json={"line": rule})).json()
    assert rule in promoted["settings"]["extra_instructions"]
    matches = (await api.post("/api/rules/matches", json={"word": "scarf"})).json()["matches"]
    assert len(matches) == 3
    status = await run_batch(api, image_ids=[m["id"] for m in matches])
    assert status["done"] == 3
    assert rule in server.state.project.store.snapshot(status["run_id"])["extra_instructions"]


async def test_regenerate_one_image_with_extra_instructions(api, dataset, mock_url):
    make_image(dataset / "flaky.png", (3, 3, 3), tags="mock_bad_json")
    await api.post("/api/dataset/rescan")
    status = await run_batch(api)
    assert status["failed"] == 1
    failed = (await api.get("/api/review", params={"q": "status:failed"})).json()["rows"][0]
    assert (await api.post(f"/api/review/{failed['id']}/regenerate", json={})).status_code == 422
    (dataset / "flaky.txt").write_text("1girl")
    again = (await api.post(f"/api/review/{failed['id']}/regenerate", json={"extra": "Mention the hat."})).json()
    assert again["status"] == "done" and again["caption"]
    requests = [r["body"] for r in mock_url[1].state.mock["requests"]]
    look = [b for b in requests if "caption_fields" in str(b.get("response_format"))][-1]
    compose = [b for b in requests if "Notes:" in str(b["messages"][-1]["content"])][-1]
    # What you say about one image reaches both steps as a known fact, and is kept with the image.
    assert "Known facts about this image" in look["messages"][1]["content"][0]["text"]
    assert "Mention the hat." in look["messages"][1]["content"][0]["text"]
    assert "Mention the hat." in compose["messages"][0]["content"]
    assert again["note"] == "Mention the hat."
    mock_url[1].state.mock["requests"].clear()
    await run_batch(api, image_ids=[failed["id"]])
    assert "Mention the hat." in str(mock_url[1].state.mock["requests"][0]["body"])      # used again on a later run


async def test_test_bench_scores_facts_and_writes_nothing(api, dataset):
    images = (await api.get("/api/dataset")).json()["images"]
    target = next(i for i in images if i["rel"] == "unsure.png")
    await api.post("/api/test/pin", json={"id": target["id"], "pinned": True})
    await api.put("/api/test/facts", json={"id": target["id"], "facts": ["one girl", "red scarf", "three dragons"]})
    result = (await api.post("/api/test/run", json={"label": "first", "twice": True})).json()
    assert len(result["run_ids"]) == 2 and not list(dataset.glob("*_nl.txt"))
    run = (await api.get(f"/api/test/run/{result['run_ids'][0]}")).json()
    assert run["results"][0]["verdicts"] == ["captured", "captured", "missing"]
    assert run["summary"]["captured"] == 2 and run["summary"]["flagged"] == 1
    assert any(r["signal"] == "uncertain" for r in run["results"][0]["reasons"])     # the bench shows review flags too

    preview = (await api.post("/api/test/request", json={"id": target["id"]})).json()
    assert preview["look"]["endpoint"] == "/v1/chat/completions"
    assert "<image: base64" in str(preview["look"]["body"]) and "data:image" not in str(preview["look"]["body"])

    # Promote makes the tested version the dataset's run settings, even after the recipe has changed since.
    await api.put("/api/dataset/settings", json={"trigger": "changed", "sampling": {"temperature": 1.2}})
    promoted = (await api.post("/api/test/promote", json={"run_id": result["run_ids"][0]})).json()["settings"]
    assert promoted["trigger"] == "" and promoted["sampling"]["temperature"] == 0.3


async def test_prefill_variants_reset_and_sharing(api):
    made = (await api.post("/api/prefills/character/duplicate", json={"name": "Character, chibi"})).json()
    assert made["id"] == "character-chibi"
    variant = next(p for p in made["prefills"] if p["id"] == made["id"])
    variant["fields"][0]["key"] = "Bad Key"
    assert (await api.put(f"/api/prefills/{made['id']}", json=variant)).status_code == 422

    style = next(p for p in made["prefills"] if p["id"] == "style")
    style["banned_words"].append("sparkly")
    await api.put("/api/prefills/style", json=style)
    exported = (await api.get("/api/share/export")).json()
    assert "sparkly" in next(p for p in exported["prefills"] if p["id"] == "style")["banned_words"]
    style["max_words"] = 160
    await api.put("/api/prefills/style", json=style)
    reset = (await api.post("/api/prefills/style/reset")).json()["prefills"]
    assert "sparkly" not in next(p for p in reset if p["id"] == "style")["banned_words"]
    assert next(p for p in reset if p["id"] == "style")["max_words"] == 160      # a reset keeps the word range you chose
    imported = (await api.post("/api/share/import", json=exported)).json()["prefills"]
    assert "sparkly" in next(p for p in imported if p["id"] == "style")["banned_words"]
    assert (await api.delete("/api/prefills/style")).status_code == 409
    assert (await api.delete(f"/api/prefills/{made['id']}")).status_code == 200


async def test_plain_errors_without_a_dataset_or_model(api):
    server.state.project = None
    response = await api.get("/api/review")
    assert response.status_code == 409 and "Point me at a folder" in response.json()["detail"]
    assert (await api.post("/api/dataset/open", json={"path": "/no/such/folder"})).status_code == 404


async def test_review_trusts_a_description_on_disk_over_a_stale_failure(api, dataset):
    await run_batch(api)
    image = server.state.project.store.image_by_rel("plain.png")
    server.state.project.store.update_image(image["id"], status="failed", error="The model returned an empty caption.")
    data = (await api.get("/api/review")).json()
    assert data["totals"]["failed"] == 0 and len(data["rows"]) == 3
    assert server.state.project.store.image(image["id"])["status"] == "done"


async def test_folder_picker_opens_at_the_start_folder(api, dataset):
    assert (await api.get("/api/browse")).json()["path"] == ""
    saved = (await api.put("/api/settings", json={"start_folder": str(dataset.parent)})).json()
    assert saved["settings"]["start_folder"] == str(dataset.parent)
    opened = (await api.get("/api/browse")).json()
    assert opened["path"] == str(dataset.parent) and "dataset" in [f["name"] for f in opened["folders"]]
    assert (await api.get("/api/browse", params={"roots": "true"})).json()["path"] == ""      # the drive list is still reachable
    await api.put("/api/settings", json={"start_folder": "/no/longer/there"})
    assert (await api.get("/api/browse")).json()["path"] == ""


async def test_lists_sort_by_name_in_natural_order(api, dataset):
    for name in ("ta (2).png", "ta (16).png", "ta (10).png"):
        make_image(dataset / name, (5, 90, 200), tags="1girl")
    await api.post("/api/dataset/rescan")
    names = [i["rel"] for i in (await api.get("/api/dataset")).json()["images"]]
    assert [n for n in names if n.startswith("ta")] == ["ta (2).png", "ta (10).png", "ta (16).png"]
    await run_batch(api)
    by_name = [r["rel"] for r in (await api.get("/api/review")).json()["rows"]]
    assert by_name == ["clash.png", "plain.png", "ta (2).png", "ta (10).png", "ta (16).png", "unsure.png"]
    by_score = (await api.get("/api/review", params={"sort": "score"})).json()["rows"]
    assert [r["score"] for r in by_score] == sorted((r["score"] for r in by_score), reverse=True)


async def test_an_image_replaced_under_the_same_name_starts_fresh(api, dataset, tmp_path):
    await run_batch(api)
    store = server.state.project.store
    old = store.image_by_rel("plain.png")
    await api.post("/api/test/pin", json={"id": old["id"], "pinned": True})
    old_caption = (dataset / "plain_nl.txt").read_text()
    old_thumb = await api.get(f"/api/image/{old['id']}")
    assert (await api.get(f"/api/image/{old['id']}", headers={"If-None-Match": old_thumb.headers["etag"]})).status_code == 304

    make_image(dataset / "plain.png", (10, 200, 240), size=(700, 1100))        # a new picture, same name
    data = (await api.post("/api/dataset/rescan")).json()
    fresh = store.image_by_rel("plain.png")
    assert fresh["id"] == old["id"] and fresh["fields"] is None and fresh["status"] == "pending" and not fresh["note"]
    assert (fresh["width"], fresh["height"]) == (700, 1100) and old["id"] not in store.testset()
    badge = next(i for i in data["images"] if i["rel"] == "plain.png")["health"]
    assert "stale_caption" in [b["check"] for b in badge]
    new_thumb = await api.get(f"/api/image/{old['id']}", headers={"If-None-Match": old_thumb.headers["etag"]})
    assert new_thumb.status_code == 200 and new_thumb.content != old_thumb.content      # the browser gets the new picture

    # The next run rewrites the old description even with Overwrite off, and keeps the old one in history.
    status = await run_batch(api)
    assert status["done"] == 1 and status["skipped"] == 2
    assert (dataset / "plain_nl.txt").read_text() != old_caption
    assert store.history(old["id"])[0]["old"] == old_caption.strip()
    assert store.image(old["id"])["status"] == "done" and store.image(old["id"])["fields"]
    after = (await api.get("/api/dataset")).json()
    assert "stale_caption" not in [b["check"] for i in after["images"] if i["rel"] == "plain.png" for b in i["health"]]

    # Another folder with the same file names never shows this folder's pictures.
    other = tmp_path / "other"
    other.mkdir()
    make_image(other / "clash.png", (250, 250, 0))
    make_image(other / "plain.png", (0, 0, 0))
    await api.post("/api/dataset/open", json={"path": str(other)})
    there = await api.get("/api/image/1", headers={"If-None-Match": new_thumb.headers["etag"]})
    assert there.status_code == 200 and there.headers["etag"] != new_thumb.headers["etag"]


async def test_recent_folders_keep_the_last_five(api, tmp_path):
    for n in range(7):
        folder = tmp_path / f"set{n}"
        folder.mkdir()
        make_image(folder / "a.png")
        recent = (await api.post("/api/dataset/open", json={"path": str(folder)})).json()["recent"]
    assert len(recent) == 5 and recent[0].endswith("set6")


async def test_picture_addresses_differ_between_datasets_and_file_versions(api, dataset, tmp_path):
    first = (await api.get("/api/dataset")).json()
    other = tmp_path / "other"
    other.mkdir()
    make_image(other / "plain.png", (0, 0, 0))
    second = (await api.post("/api/dataset/open", json={"path": str(other)})).json()
    assert first["id"] != second["id"] and first["images"][0]["id"] == second["images"][0]["id"] == 1
    before = second["images"][0]["v"]
    make_image(other / "plain.png", (9, 9, 9), size=(640, 640))
    assert (await api.post("/api/dataset/rescan")).json()["images"][0]["v"] != before


async def test_the_answer_key_never_reaches_the_model_but_the_note_does(api, dataset, mock_url):
    images = (await api.get("/api/dataset")).json()["images"]
    target = next(i for i in images if i["rel"] == "plain.png")
    await api.post("/api/test/pin", json={"id": target["id"], "pinned": True})
    await api.put("/api/test/facts", json={"id": target["id"], "facts": ["SECRET-ANSWER-KEY sits on a stump"]})
    saved = (await api.put(f"/api/image/{target['id']}/note", json={"text": "An elf girl sits on a  tree stump."})).json()
    assert saved["note"] == "An elf girl sits on a tree stump."
    assert (await api.get("/api/test")).json()["pins"][0]["note"] == saved["note"]

    requests = mock_url[1].state.mock["requests"]
    requests.clear()
    await api.post("/api/test/run", json={"label": "notes"})
    bodies = [str(r["body"]) for r in requests]
    look = next(b for b in bodies if "caption_fields" in b)
    compose = next(b for b in bodies if "Notes:" in b and "fact_scores" not in b)
    scoring = next(b for b in bodies if "fact_scores" in b)
    # The note is told to both captioning steps as fact.
    assert "An elf girl sits on a tree stump." in look and "Known facts about this image" in look
    assert "An elf girl sits on a tree stump." in compose
    # The answer key is only ever used to score the finished caption.
    assert "SECRET-ANSWER-KEY" not in look and "SECRET-ANSWER-KEY" not in compose and "SECRET-ANSWER-KEY" in scoring

    preview = (await api.post("/api/test/request", json={"id": target["id"]})).json()
    assert "An elf girl" in str(preview["look"]["body"]) and "An elf girl" in str(preview["compose"]["body"])


async def test_the_page_always_loads_its_current_scripts(api):
    page = await api.get("/")
    version = (await api.get("/health")).json()["page_version"]
    assert page.headers["cache-control"] == "no-store"
    assert f'/static/core.js?v={version}"' in page.text and f'/static/style.css?v={version}"' in page.text
    assert page.text.count(f"?v={version}") == 11          # ten scripts and the stylesheet
    script = await api.get("/static/bench.js")
    assert script.headers["cache-control"] == "no-cache" and "hintBox" in script.text
    # Pantry comes before Taste Test, left to right.
    assert page.text.index('data-tab="dataset"') < page.text.index('data-tab="bench"')


async def test_versions_tried_are_capped_at_eight_name_their_model_and_can_be_cleared(api, dataset):
    images = (await api.get("/api/dataset")).json()["images"]
    await api.post("/api/test/pin", json={"id": images[0]["id"], "pinned": True})
    for n in range(10):
        data = (await api.post("/api/test/run", json={"label": f"v{n}"})).json()
    assert [r["label"] for r in data["runs"]] == [f"v{n}" for n in range(9, 1, -1)]          # the newest eight
    assert data["runs"][0]["model"] == "mock-captioner" and data["runs"][0]["verifier_model"] == "mock-captioner"
    after = (await api.delete(f"/api/test/run/{data['runs'][0]['id']}")).json()
    assert len(after["runs"]) == 7 and after["runs"][0]["label"] == "v8"
    assert (await api.delete("/api/test/runs")).json()["runs"] == []

    status = await run_batch(api)
    assert status["model"] == "mock-captioner" and status["verifier_model"] == "mock-captioner"


async def test_review_can_filter_by_health_badge(api, dataset):
    from PIL import Image

    Image.new("RGBA", (900, 900), (0, 0, 0, 0)).save(dataset / "see_through.png")
    (dataset / "see_through.txt").write_text("1girl")
    await api.post("/api/dataset/rescan")
    await run_batch(api)
    rows = (await api.get("/api/review", params={"q": "health:transparency"})).json()["rows"]
    assert [r["rel"] for r in rows] == ["see_through.png"]
    assert len((await api.get("/api/review", params={"q": "NOT health:transparency"})).json()["rows"]) == 3


async def test_hint_with_braces_and_quotes_is_sent_as_written(api, dataset, mock_url):
    images = (await api.get("/api/dataset")).json()["images"]
    hint = 'The sign reads "{OPEN}" and 100% of it is {visible}.'
    await api.put(f"/api/image/{images[0]['id']}/note", json={"text": hint})
    requests = mock_url[1].state.mock["requests"]
    requests.clear()
    status = await run_batch(api, image_ids=[images[0]["id"]])
    assert status["done"] == 1 and status["failed"] == 0
    assert hint in requests[0]["body"]["messages"][1]["content"][0]["text"]
