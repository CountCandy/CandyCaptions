"""Flavor Profile (style): the macro pass, the style lens and the ringer test."""

import random

import httpx
import pytest
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageOps

from candycaption import server
from candycaption.config import save_settings
from candycaption.imaging import measure
from candycaption.similar import caption_pairs, decode, picture_pairs
from candycaption.style import MIN_SET, macro_flags, pixel_outliers


def art(path, seed, size=(900, 900), gray=False, blur=0, quality=None, palette=None):
    """A busy, colorful picture. The same seed always draws the same thing."""
    rng = random.Random(seed)
    base = palette or (rng.randint(60, 200), rng.randint(60, 200), rng.randint(60, 200))
    img = Image.new("RGB", size, base)
    draw = ImageDraw.Draw(img)
    for _ in range(140):
        x, y, r = rng.randint(0, size[0]), rng.randint(0, size[1]), rng.randint(8, 90)
        color = tuple(min(255, max(0, c + rng.randint(-90, 90))) for c in base)
        (draw.ellipse if rng.random() < 0.5 else draw.rectangle)([x - r, y - r, x + r, y + r], fill=color)
    if gray:
        img = img.convert("L").convert("RGB")
    if blur:
        img = img.filter(ImageFilter.GaussianBlur(blur))
    if quality:
        img.save(path.with_suffix(".jpg"), "JPEG", quality=quality)
        Image.open(path.with_suffix(".jpg")).save(path)       # a JPEG saved again as PNG, like the user's screenshots
        path.with_suffix(".jpg").unlink()
    else:
        img.save(path)
    return path


@pytest.fixture
def styled(tmp_path):
    folder = tmp_path / "styled"
    folder.mkdir()
    for n in range(1, 13):
        art(folder / f"ta ({n}).png", n)
    return folder


def test_macro_pass_finds_each_kind_of_source_quality_outlier(styled):
    art(styled / "soft.png", 50, blur=6)
    art(styled / "tiny.png", 51, size=(260, 260))
    art(styled / "blocky.png", 52, quality=12)
    art(styled / "mono.png", 53, gray=True)
    files = sorted(styled.glob("*.png"))
    flags = dict(zip([f.name for f in files], macro_flags([measure(f) for f in files])))
    assert [f["check"] for f in flags["soft.png"]] == ["soft"]
    assert "small" in [f["check"] for f in flags["tiny.png"]]
    assert "blocky" in [f["check"] for f in flags["blocky.png"]]
    assert "mono" in [f["check"] for f in flags["mono.png"]]
    assert all(not flags[f"ta ({n}).png"] for n in range(1, 13))          # the ordinary ones are left alone


def test_nothing_is_flagged_in_an_even_set_or_a_tiny_one(styled):
    files = sorted(styled.glob("*.png"))
    stats = [measure(f) for f in files]
    assert all(not f for f in macro_flags(stats))
    # Fewer than the minimum: there is no "rest of the set" to compare with.
    assert MIN_SET == 8 and all(not f for f in macro_flags(stats[:5]))
    assert all(p["z"] == 0 for p in pixel_outliers(stats[:5]))


def test_compression_shared_by_the_whole_set_is_not_an_outlier(tmp_path):
    files = [art(tmp_path / f"shot{n}.png", n, quality=35) for n in range(12)]
    stats = [measure(f) for f in files]
    assert min(s["blocking"] for s in stats) > 1.05              # every image carries the block pattern...
    assert all(not f for f in macro_flags(stats))                  # ...so none of them is the odd one out


def test_pixels_notice_a_different_palette(styled):
    for n in range(1, 13):                                          # a warm, similar set
        art(styled / f"ta ({n}).png", n, palette=(190 + n, 120, 70))
    art(styled / "cold.png", 99, palette=(40, 90, 210))
    files = sorted(styled.glob("*.png"))
    px = dict(zip([f.name for f in files], pixel_outliers([measure(f) for f in files])))
    assert px["cold.png"]["z"] > 3.5 and px["cold.png"]["z"] == max(p["z"] for p in px.values())
    assert "a different palette" in px["cold.png"]["reasons"]


@pytest.fixture
async def api(settings, styled):
    save_settings(settings)
    server.state.project = server.state.runner = server.state.style_job = None
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=server.app), base_url="http://app") as client:
        yield client
    if server.state.project:
        server.state.project.store.close()


async def run_style(api, **body):
    assert (await api.post("/api/style/run", json=body)).status_code == 200
    await server.state.style_job.wait()
    return (await api.get("/api/style")).json()


async def test_style_lens_names_the_odd_one_and_only_that_one(api, styled, mock_url):
    art(styled / "sketch.png", 77, gray=True)
    await api.post("/api/dataset/open", json={"path": str(styled)})
    before = (await api.get("/api/style")).json()
    assert before["enough"] and before["count"] == 13 and before["card"] is None and before["checked"] == 0
    assert [r["rel"] for r in before["macro"]] == ["sketch.png"]          # the macro pass needs no model

    requests = mock_url[1].state.mock["requests"]
    requests.clear()
    data = await run_style(api)
    assert data["job"]["state"] == "done" and "1 piece tastes different" in data["job"]["message"]
    assert data["card"].startswith("Digital illustration") and data["checked"] == 13
    assert [r["rel"] for r in data["odd"]] == ["sketch.png"]
    odd = data["odd"][0]
    assert odd["match"] < 0.2 and odd["measured"] and "pencil" in odd["note"]
    assert data["made"]["model"] == "mock-captioner"
    # 13 notes, one card, 13 checks; every request with a picture holds exactly one.
    assert len(requests) == 27
    for request in requests:
        pictures = str(request["body"]).count("'image_url': {")
        assert pictures <= 1 and "stream" in request["body"]

    why = (await api.post("/api/style/explain", json={"id": odd["id"]})).json()["why"]
    assert "monochrome" in why
    assert str(requests[-1]["body"]).count("'image_url': {") == 4            # the image in question and three typical ones
    assert (await api.get("/api/style")).json()["odd"][0]["why"] == why

    # A second run reuses the notes: one card and 13 checks.
    requests.clear()
    await run_style(api)
    assert len(requests) == 14


async def test_an_even_set_reports_that_nothing_stands_out(api, styled):
    await api.post("/api/dataset/open", json={"path": str(styled)})
    data = await run_style(api)
    assert data["odd"] == [] and data["job"]["message"] == "Batch tasted: nothing stands out."
    assert min(r["match"] for r in data["rows"]) > 0.9


async def test_plant_a_ringer_ranks_it_without_adding_it(api, styled, tmp_path):
    await api.post("/api/dataset/open", json={"path": str(styled)})
    await run_style(api)
    ringer = art(tmp_path / "ringer.png", 5, gray=True)
    result = (await api.post("/api/style/ringer", json={"path": str(ringer)})).json()
    assert result["model_rank"] == 1 and result["match"] < 0.2 and result["count"] == 12
    assert "mono" in [f["check"] for f in result["macro"]]
    fits = art(tmp_path / "fits.png", 6)
    assert (await api.post("/api/style/ringer", json={"path": str(fits)})).json()["match"] > 0.9
    assert (await api.get("/api/style")).json()["count"] == 12 and not (styled / "ringer.png").exists()
    assert (await api.post("/api/style/ringer", json={"path": str(tmp_path / "nope.png")})).status_code == 404


async def test_style_check_needs_enough_images(api, tmp_path):
    few = tmp_path / "few"
    few.mkdir()
    for n in range(4):
        art(few / f"a{n}.png", n)
    await api.post("/api/dataset/open", json={"path": str(few)})
    refused = await api.post("/api/style/run", json={})
    assert refused.status_code == 409 and "at least 8" in refused.json()["detail"]
    assert (await api.get("/api/style")).json()["enough"] is False


async def test_checking_only_some_images_and_against_target_images(api, styled, mock_url):
    for n in range(20, 26):
        art(styled / f"sketch ({n}).png", n, gray=True)                 # six sketches among twelve color pieces
    await api.post("/api/dataset/open", json={"path": str(styled)})
    images = {i["rel"]: i["id"] for i in (await api.get("/api/dataset")).json()["images"]}
    color = [images[f"ta ({n}).png"] for n in range(1, 13)]
    sketches = [images[f"sketch ({n}).png"] for n in range(20, 26)]

    # Only the chosen images are checked and ranked.
    chosen = color[:9] + sketches[:1]
    data = (await api.put("/api/style/selection", json={"ids": chosen})).json()
    assert data["count"] == 10 and data["total"] == 18 and sorted(data["selection"]) == sorted(chosen)
    requests = mock_url[1].state.mock["requests"]
    requests.clear()
    data = await run_style(api)
    assert len(requests) == 21 and data["checked"] == 10                 # 10 notes, 1 card, 10 checks
    assert [r["rel"] for r in data["odd"]] == ["sketch (20).png"]
    assert [r["rel"] for r in data["macro"]] == ["sketch (20).png"]      # the pixel pass follows the choice too

    # Changing what is checked makes the old results stale instead of mixing them in.
    data = (await api.put("/api/style/selection", json={"ids": None})).json()
    assert data["selection"] is None and data["count"] == 18 and data["checked"] == 0 and data["card"] is None

    # Target images: the style is written from them alone, and "How does it differ?" compares with them.
    data = (await api.put("/api/style/targets", json={"ids": sketches[:2]})).json()
    assert [t["rel"] for t in data["targets"]] == ["sketch (20).png", "sketch (21).png"]
    requests.clear()
    data = await run_style(api)
    # One target has a note from the run above and the other still needs one; no other image needs any.
    assert len(requests) == 1 + 1 + 18 and data["job"]["steps"] == 2
    card_request = next(r["body"] for r in requests if "You are given style notes" in str(r["body"]))
    assert str(card_request).count("pencil sketch") == 2 and "digital illustration" not in str(card_request["messages"][1])
    assert data["made"]["targets"] == 2 and sum(1 for r in data["rows"] if r["target"]) == 2
    await api.post("/api/style/explain", json={"id": color[0]})
    assert str(requests[-1]["body"]).count("'image_url': {") == 3        # the image and the two targets

    # A small set can be checked once it has targets, though the pixel passes still need eight images.
    data = (await api.put("/api/style/selection", json={"ids": color[:3]})).json()
    assert data["count"] == 3 and data["can_run"] and not data["enough"]
    assert (await api.put("/api/style/targets", json={"ids": []})).json()["can_run"] is False


async def test_macro_list_is_in_file_name_order(api, styled):
    art(styled / "ta (20).png", 60, blur=6)
    art(styled / "ta (3).png", 61, gray=True)
    art(styled / "ta (100).png", 62, blur=6)
    await api.post("/api/dataset/open", json={"path": str(styled)})
    assert [r["rel"] for r in (await api.get("/api/style")).json()["macro"]] == ["ta (3).png", "ta (20).png", "ta (100).png"]


async def test_only_one_long_job_runs_at_a_time(api, styled, tmp_path):
    from candycaption.mock_server import create_app
    from conftest import _serve
    from candycaption.config import load_settings

    slow, url = _serve(create_app(delay=0.3))
    try:
        save_settings(dict(load_settings(), base_url=url))
        await api.post("/api/dataset/open", json={"path": str(styled)})
        assert (await api.post("/api/style/run", json={})).status_code == 200
        for path, body in (("/api/run/start", {}), ("/api/test/run", {}), ("/api/style/run", {}), ("/api/review/1/regenerate", {})):
            refused = await api.post(path, json=body)
            assert refused.status_code == 409 and "style check is running" in refused.json()["detail"], path
        other = tmp_path / "other"
        other.mkdir()
        assert (await api.post("/api/dataset/open", json={"path": str(other)})).status_code == 409
        assert (await api.put("/api/style/targets", json={"ids": [1]})).status_code == 409
        await server.state.style_job.wait()
        assert (await api.post("/api/run/start", json={})).status_code == 200      # free again once it finishes
        await server.state.runner.wait()
    finally:
        slow.should_exit = True


async def test_targets_cost_one_request_per_image_not_one_per_target(api, styled, mock_url):
    await api.post("/api/dataset/open", json={"path": str(styled)})
    images = (await api.get("/api/dataset")).json()["images"]
    await api.put("/api/style/targets", json={"ids": [images[0]["id"], images[1]["id"], images[2]["id"]]})
    requests = mock_url[1].state.mock["requests"]
    requests.clear()
    data = await run_style(api)
    # Three notes for the targets, one card, then twelve checks: not 12 x 3, and no notes for the other nine.
    assert len(requests) == 3 + 1 + 12
    assert data["checked"] == 12 and data["job"]["steps"] == 2 and data["job"]["step_total"] == 12
    assert sum(1 for r in data["rows"] if r["note"]) == 3


# -- Two of a Kind: images or captions that are too alike ------------------

def test_look_alikes_are_found_whatever_was_done_to_the_copy(styled):
    first = Image.open(styled / "ta (1).png")
    ImageEnhance.Color(first).enhance(0.3).save(styled / "z_recolored.png")
    ImageOps.mirror(first).save(styled / "z_mirrored.png")
    first.crop((60, 40, 860, 880)).save(styled / "z_cropped.png")
    first.crop((230, 310, 870, 830)).save(styled / "z_cropped_hard.png")
    patched = first.copy()
    ImageDraw.Draw(patched).ellipse([380, 300, 470, 380], fill=(250, 240, 230))
    patched.save(styled / "z_patched.png")
    files = sorted(styled.glob("*.png"))
    names = [f.name for f in files]
    pairs = picture_pairs([decode(measure(f)["sig"]) for f in files])
    one = names.index("ta (1).png")
    kinds = {names[b]: (score, kind) for (a, b), (score, kind) in pairs.items() if a == one}
    assert kinds["z_recolored.png"][1] == "same" and kinds["z_mirrored.png"][1] == "mirrored"
    assert kinds["z_cropped.png"][1] == "crop" and kinds["z_cropped_hard.png"][1] == "crop" and kinds["z_patched.png"][1] == "edited"
    assert all(score > 0.93 for score, _ in kinds.values())
    # Twelve different pictures: none of them pairs with another.
    assert not [pair for pair in pairs if names[pair[0]].startswith("ta") and names[pair[1]].startswith("ta")]


def test_caption_pairs_ignore_the_words_every_caption_shares():
    opening = "A digital painting in a soft watercolor style. "
    texts = [
        opening + "A red fox sleeps under a birch tree beside a frozen lake at dawn.",
        opening + "A red fox is sleeping under a birch tree next to a frozen lake at dawn.",
        opening + "Two sailors mend a torn net on a crowded harbor wall in summer rain.",
        opening + "An old clockmaker bends over a brass gear at a cluttered workbench.",
        "",
    ]
    pairs = caption_pairs(texts)
    assert list(pairs) == [(0, 1)] and pairs[(0, 1)] > 0.5      # the shared opening alone makes nothing a match


async def test_similar_lists_pairs_and_lets_one_be_marked_fine(api, styled):
    first = Image.open(styled / "ta (1).png")
    first.resize((450, 450)).save(styled / "small copy.png")
    (styled / "ta (3)_nl.txt").write_text("A knight in green armor rides a gray horse across a stone bridge.\n", encoding="utf-8")
    (styled / "ta (4)_nl.txt").write_text("A knight in green armor is riding a gray horse over a stone bridge.\n", encoding="utf-8")
    others = ["Three paper lanterns float above a quiet canal at night.", "A baker dusts flour over a wooden table.",
              "Two owls perch on a snowy fence post.", "A lighthouse stands against a violet storm.",
              "Children chase a kite along a windy beach.", "A violinist plays beneath a flickering street lamp."]
    for n, text in enumerate(others, start=5):
        (styled / f"ta ({n})_nl.txt").write_text(text + "\n", encoding="utf-8")
    await api.post("/api/dataset/open", json={"path": str(styled)})
    data = (await api.get("/api/similar")).json()
    assert data["images"] == 13 and data["captions"] == 8 and data["dismissed"] == 0
    found = {(p["a"]["rel"], p["b"]["rel"]): p for p in data["pairs"]}
    assert set(found) == {("small copy.png", "ta (1).png"), ("ta (3).png", "ta (4).png")}
    copy, words = found[("small copy.png", "ta (1).png")], found[("ta (3).png", "ta (4).png")]
    assert copy["picture"] > 0.97 and copy["kind"] == "same" and copy["caption"] is None
    assert words["picture"] is None and words["caption"] > 0.5 and "knight" in words["captions"][0]

    await api.put("/api/similar/fine", json={"a": copy["a"]["id"], "b": copy["b"]["id"]})
    after = (await api.get("/api/similar")).json()
    assert len(after["pairs"]) == 1 and after["dismissed"] == 1
    again = (await api.request("DELETE", "/api/similar/fine")).json()
    assert len(again["pairs"]) == 2 and again["dismissed"] == 0
