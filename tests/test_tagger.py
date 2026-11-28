import json

import httpx
import numpy as np
import pytest
from PIL import Image

from candycaption import server
from candycaption.config import save_settings
from candycaption.tagger import Tagger, TaggerError, apply_check, check_schema, get_tagger, split_tags, status

from conftest import make_image

# name, category (0 general, 4 character, 9 rating), which color channel raises it (blue, green, red), bias
TOY_TAGS = [
    ("general", 9, None, 3.0), ("sensitive", 9, None, -3.0),
    ("1girl", 0, None, 9.0), ("red_hair", 0, 2, -6.0), ("blue_sky", 0, 0, -6.0), ("orange_hair", 0, None, -9.0),
    ("^_^", 0, 2, -6.0), ("hatsune_miku", 4, 1, -6.0), ("faint_thing", 0, None, -1.0),
]


@pytest.fixture
def toy_tagger(tmp_path):
    """A tiny real ONNX tagger: the average blue, green and red of the picture decide the tags."""
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    weights = np.zeros((3, len(TOY_TAGS)), dtype=np.float32)
    bias = np.zeros(len(TOY_TAGS), dtype=np.float32)
    for column, (_, _, channel, offset) in enumerate(TOY_TAGS):
        bias[column] = offset
        if channel is not None:
            weights[channel, column] = 12.0 / 255.0
    graph = helper.make_graph(
        [helper.make_node("ReduceMean", ["input", "axes"], ["mean"], keepdims=0),
         helper.make_node("MatMul", ["mean", "weights"], ["scaled"]),
         helper.make_node("Add", ["scaled", "bias"], ["logits"]),
         helper.make_node("Sigmoid", ["logits"], ["output"])],
        "toy_tagger",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 8, 8, 3])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, len(TOY_TAGS)])],
        [numpy_helper.from_array(np.array([1, 2], dtype=np.int64), "axes"),
         numpy_helper.from_array(weights, "weights"), numpy_helper.from_array(bias, "bias")])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 9
    folder = tmp_path / "toy tagger"
    folder.mkdir()
    onnx.save(model, str(folder / "model.onnx"))
    rows = ["tag_id,name,category,count"] + [f"{n},{name},{category},100" for n, (name, category, _, _) in enumerate(TOY_TAGS)]
    (folder / "selected_tags.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    return folder


def test_the_tagger_reads_the_picture_and_orders_its_tags(toy_tagger, tmp_path):
    tagger = Tagger(toy_tagger, "toy")
    assert tagger.size == 8 and tagger.device == "CPU" and "orange hair" in tagger.vocabulary

    Image.new("RGB", (300, 200), (250, 10, 10)).save(tmp_path / "red.png")
    red = tagger.tag(tmp_path / "red.png")
    # Underscores become spaces, except in a face drawn with punctuation. The rating is reported apart.
    assert set(red["tags"]) == {"1girl", "red hair", "^_^"} and red["rating"] == "general"
    assert red["tags"][0] == "1girl" and red["scores"]["1girl"] > red["scores"]["red hair"]

    # A see-through picture is filled with the background color before it is tagged, like everywhere else.
    Image.new("RGBA", (64, 64), (0, 0, 0, 0)).save(tmp_path / "clear.png")
    assert tagger.tag(tmp_path / "clear.png", background="#0000ff")["tags"] == ["1girl", "blue sky"]

    Image.new("RGB", (64, 64), (10, 250, 10)).save(tmp_path / "green.png")
    assert tagger.tag(tmp_path / "green.png")["tags"] == ["hatsune miku", "1girl"]            # characters lead
    assert "faint thing" in tagger.tag(tmp_path / "green.png", general=0.2)["tags"]            # a lower bar lets more in
    assert "hatsune miku" not in tagger.tag(tmp_path / "green.png", character=0.9999)["tags"]


def test_a_missing_or_broken_tagger_says_what_to_do(tmp_path, data_dir):
    empty = tmp_path / "empty"
    empty.mkdir()
    settings = {"tagger_path": str(empty)}
    assert status(settings)["ready"] is False
    with pytest.raises(TaggerError, match="needs both model.onnx and selected_tags.csv"):
        get_tagger(settings)
    (empty / "model.onnx").write_bytes(b"not a model")
    (empty / "selected_tags.csv").write_text("tag_id,name,category,count\n0,1girl,0,5\n", encoding="utf-8")
    with pytest.raises(TaggerError, match="could not be loaded"):
        get_tagger(settings)
    # Without a local folder the files go in the app's data folder, never in a dataset.
    assert str(data_dir) in status({"tagger_repo": "someone/some-tagger"})["folder"]


def test_only_safe_tag_changes_are_made():
    tags = ["1girl", "red hair", "blue eyes", "hat", "kandrevel", "outdoors"]
    vocab = {"1girl", "2girls", "red hair", "orange hair", "blue eyes", "hat", "outdoors"}
    reply = json.dumps({"remove": ["hat", "kandrevel", "not in the list"],
                        "replace": [{"from": "red hair", "to": "Orange_Hair"}, {"from": "blue eyes", "to": "sparkly made up tag"},
                                    {"from": "1girl", "to": "1girl"}]})
    out = apply_check(tags, reply, {"kandrevel"}, vocab)
    # The trigger word is never touched; a made-up replacement only removes the wrong tag; order is kept.
    assert out["tags"] == ["1girl", "orange hair", "kandrevel", "outdoors"]
    assert out["removed"] == ["blue eyes", "hat"] and out["replaced"] == [["red hair", "orange hair"]] and not out["refused"]

    wipe = apply_check(tags, json.dumps({"remove": tags, "replace": []}), set(), vocab)
    assert wipe["tags"] == tags and "more than half" in wipe["refused"]
    assert apply_check(tags, "I think they look fine.", set(), vocab)["refused"]
    assert apply_check(tags, '{"remove": [], "replace": []}', set(), vocab) == {"tags": tags, "removed": [], "replaced": [], "refused": ""}
    # The reply can only name tags from the list where the server enforces the schema.
    assert check_schema(tags)["properties"]["remove"]["items"]["enum"] == tags


@pytest.fixture
async def api(settings, tmp_path, toy_tagger):
    folder = tmp_path / "to tag"
    folder.mkdir()
    Image.new("RGB", (400, 400), (250, 10, 10)).save(folder / "a red.png")
    Image.new("RGB", (400, 400), (10, 10, 250)).save(folder / "b blue.png")
    make_image(folder / "c mine.png", tags="1girl, mock_wrong, mock_red, kept tag, mock_made_up_swap, outdoors, tree")
    make_image(folder / "d wipe.png", tags="1girl, mock_wipe, hat")
    save_settings(dict(settings, tagger_path=str(toy_tagger)))
    server.state.project = server.state.runner = server.state.style_job = server.state.tag_job = None
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=server.app), base_url="http://app") as client:
        assert (await client.post("/api/dataset/open", json={"path": str(folder)})).status_code == 200
        client.folder = folder
        yield client
    if server.state.project:
        server.state.project.store.close()


async def run_tagger(api, **body):
    started = await api.post("/api/tagger/run", json=body)
    assert started.status_code == 200, started.text
    await server.state.tag_job.wait()
    return server.state.tag_job.status()


async def test_tagging_fills_in_missing_tag_files_and_leaves_yours_alone(api):
    data = (await api.get("/api/dataset")).json()
    assert data["tagged"] == 2 and data["tagger"]["ready"] and data["tagger"]["local"]

    job = await run_tagger(api, scope="missing", check=False)
    assert job["state"] == "done" and job["written"] == 2 and "2 tag files written" in job["message"]
    folder = api.folder
    assert (folder / "a red.txt").read_text(encoding="utf-8") == "1girl, ^_^, red hair\n"
    assert (folder / "b blue.txt").read_text(encoding="utf-8") == "1girl, blue sky\n"
    assert "mock_wrong" in (folder / "c mine.txt").read_text(encoding="utf-8")       # an existing tag file is not replaced
    assert sorted(p.name for p in folder.iterdir() if p.suffix not in (".png", ".txt")) == []   # nothing else lands in the dataset

    # Tagging everything again replaces them, and the old tags go into the image's history.
    job = await run_tagger(api, scope="all", check=False)
    assert job["written"] == 4 and (folder / "c mine.txt").read_text(encoding="utf-8").startswith("1girl")
    mine = next(i for i in server.state.project.store.images() if i["rel"] == "c mine.png")
    assert any("mock_wrong" in str(h) for h in server.state.project.store.history(mine["id"]))


async def test_the_model_strikes_and_corrects_wrong_tags_and_each_change_can_be_undone(api, mock_url):
    requests = mock_url[1].state.mock["requests"]
    requests.clear()
    job = await run_tagger(api, scope="none", check=True)
    assert job["state"] == "done" and job["changed"] == 1 and job["step_total"] == 2
    folder = api.folder
    # Struck: mock_wrong, and the swap whose new tag is not a real one. Corrected: mock_red. The rest stays in place.
    assert split_tags((folder / "c mine.txt").read_text(encoding="utf-8")) == ["1girl", "orange hair", "kept tag", "outdoors", "tree"]
    assert (folder / "d wipe.txt").read_text(encoding="utf-8").strip() == "1girl, mock_wipe, hat"   # refused whole
    checks = [r["body"] for r in requests]
    assert len(checks) == 2 and all(b["response_format"]["json_schema"]["name"] == "tag_check" for b in checks)
    assert all(sum(1 for part in b["messages"][-1]["content"] if part["type"] == "image_url") == 1 for b in checks)

    listed = (await api.get("/api/dataset")).json()["images"]
    rows = {r["rel"]: r for r in listed}
    # The Pantry list carries the changes, so they can be seen before any caption exists.
    assert rows["c mine.png"]["tagging"]["removed"] == ["mock_wrong", "mock_made_up_swap"] and rows["a red.png"]["tagging"] is None
    mine = (await api.get(f"/api/review/{rows['c mine.png']['id']}")).json()
    assert mine["tagging"]["removed"] == ["mock_wrong", "mock_made_up_swap"] and mine["tagging"]["replaced"] == [["mock_red", "orange hair"]]
    assert mine["tagging"]["checked_by"] == "mock-captioner"
    wipe = (await api.get(f"/api/review/{rows['d wipe.png']['id']}")).json()
    assert "more than half" in wipe["tagging"]["refused"]

    back = (await api.post(f"/api/review/{mine['id']}/tags/undo", json={"tag": "mock_red"})).json()
    assert split_tags(back["tags"]) == ["1girl", "mock_red", "kept tag", "outdoors", "tree"] and back["tagging"]["replaced"] == []
    back = (await api.post(f"/api/review/{mine['id']}/tags/undo", json={"tag": "mock_wrong"})).json()
    assert split_tags(back["tags"]) == ["1girl", "mock_red", "kept tag", "outdoors", "tree", "mock_wrong"]
    assert back["tagging"]["removed"] == ["mock_made_up_swap"]

    # Already checked, so a second check has nothing to do and does not undo the restores.
    requests.clear()
    job = await run_tagger(api, scope="none", check=True)
    assert job["step_total"] == 0 and not requests

    data = (await api.get("/api/dataset")).json()
    assert data["tags_changed"] == 1 and data["tags_unchecked"] == 0


async def test_tag_then_check_in_one_go_and_refusals(api):
    job = await run_tagger(api, scope="missing", check=True)
    # Two images tagged, then all four tag files checked: the two new ones and the two that were already there.
    assert job["written"] == 2 and job["steps"] == 2 and job["step_total"] == 4 and job["changed"] == 1
    assert (await api.post("/api/tagger/run", json={"scope": "none", "check": False})).status_code == 400
    settings = dict(server.load_settings(), model="")
    save_settings(settings)
    refused = await api.post("/api/tagger/run", json={"scope": "none", "check": True})
    assert refused.status_code == 400 and "Captioner" in refused.json()["detail"]
