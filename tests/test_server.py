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
    rows = (await api.get("/api/review")).json()["rows"]
    assert [r["rel"] for r in rows][-1] == "plain.png" and rows[0]["score"] >= rows[1]["score"] > 0
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
    looks = [r["body"] for r in mock_url[1].state.mock["requests"] if "caption_fields" in str(r["body"].get("response_format"))]
    assert "Mention the hat." in looks[-1]["messages"][0]["content"]


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
    reset = (await api.post("/api/prefills/style/reset")).json()["prefills"]
    assert "sparkly" not in next(p for p in reset if p["id"] == "style")["banned_words"]
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
