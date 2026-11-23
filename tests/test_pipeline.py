import asyncio

import pytest

from candycaption.client import ModelClient
from candycaption.connect import test_connection as probe_connection
from candycaption.dataset import Project
from candycaption.prefills import find_banned
from candycaption.runner import Runner
from candycaption.signals import combine

from conftest import make_image


async def run(folder, settings, **dataset_settings):
    project = Project(folder)
    ds = project.store.settings()
    ds.update(dataset_settings)
    project.store.save_settings(ds)
    events = []
    runner = Runner(project, settings, publish=events.append)
    runner.start()
    await runner.wait()
    return project, runner, events


def by_name(project):
    return {image["rel"]: image for image in project.store.images()}


async def test_captions_are_written_next_to_images(dataset, settings):
    project, runner, events = await run(dataset, settings)
    assert runner.state == "done" and runner.done == 3 and runner.failed == 0
    for name in ("plain", "unsure", "clash"):
        text = (dataset / f"{name}_nl.txt").read_text(encoding="utf-8")
        assert len(text.split()) > 10
    # The Style prefill keeps the style field out of the caption.
    assert "digital illustration" not in (dataset / "plain_nl.txt").read_text()
    assert by_name(project)["plain.png"]["fields"]["style"].startswith("digital illustration")
    assert any(e["type"] == "delta" for e in events)
    assert runner.message == "Batch done: 3 captions wrapped."


async def test_every_request_sets_stream_and_holds_one_image(dataset, settings, mock_url):
    requests = mock_url[1].state.mock["requests"]
    requests.clear()
    await run(dataset, settings, field_checks="all")
    assert requests
    for request in requests:
        body = request["body"]
        assert "stream" in body
        images = [p for m in body["messages"] if isinstance(m["content"], list) for p in m["content"] if p["type"] == "image_url"]
        assert len(images) <= 1
        assert len([m for m in body["messages"] if m["role"] != "system"]) == 1  # no earlier turns resent


async def test_bad_replies_go_to_the_retry_list(dataset, settings):
    make_image(dataset / "bad.png", (1, 2, 3), tags="mock_bad_json")
    make_image(dataset / "empty.png", (4, 5, 6), tags="mock_empty")
    make_image(dataset / "error.png", (7, 8, 9), tags="mock_error")
    make_image(dataset / "flaky.png", (10, 11, 12), tags="mock_flaky, 1girl")
    project, runner, _ = await run(dataset, settings)
    images = by_name(project)
    assert runner.failed == 3 and runner.done == 4
    for name in ("bad.png", "empty.png", "error.png"):
        assert images[name]["status"] == "failed" and images[name]["error"]
        assert not (dataset / name).with_name(name[:-4] + "_nl.txt").exists()
    # One malformed reply is retried once and then succeeds.
    assert images["flaky.png"]["status"] == "done"
    assert "1 misshape" not in runner.message and "3 misshapes to sort" in runner.message


async def test_banned_words_trigger_one_rewrite(dataset, settings):
    make_image(dataset / "banned.png", (90, 90, 9), tags="mock_banned, 1girl")
    project, _, _ = await run(dataset, settings)
    caption = (dataset / "banned_nl.txt").read_text()
    banned = project.store.snapshot(by_name(project)["banned.png"]["snapshot_id"])["prefill"]["banned_words"]
    assert find_banned(caption, banned) == []
    assert "MOCK_BANNED" not in caption


async def test_resume_skips_finished_and_overwrite_keeps_history(dataset, settings):
    project, first, _ = await run(dataset, settings)
    assert first.done == 3
    _, second, _ = await run(dataset, settings)
    assert second.done == 0 and second.skipped == 3
    (dataset / "plain_nl.txt").write_text("An old caption written by hand.\n")
    project, third, _ = await run(dataset, settings, overwrite=True)
    assert third.done == 3
    image = by_name(project)["plain.png"]
    assert [h["old"] for h in project.store.history(image["id"])] == ["An old caption written by hand."]


async def test_signals_flag_the_right_field(dataset, settings):
    make_image(dataset / "doubt.png", (9, 99, 9), tags="1girl, mock_doubt, mock_unsure")
    project, _, _ = await run(dataset, settings, field_checks="flagged")
    images = by_name(project)
    weights = settings["score_weights"]

    reasons = {r["signal"]: r for r in combine(images["unsure.png"]["signals"], weights)[1]}
    assert reasons["uncertain"]["field"] == "clothing"
    assert "red" in reasons["uncertain"]["message"] and "orange" in reasons["uncertain"]["message"]

    reasons = {r["signal"]: r for r in combine(images["clash.png"]["signals"], weights)[1]}
    assert reasons["tags"]["field"] == "appearance" and "red_hair" in reasons["tags"]["message"]

    reasons = {r["signal"]: r for r in combine(images["doubt.png"]["signals"], weights)[1]}
    assert reasons["checks"]["field"] == "clothing"
    assert images["doubt.png"]["signals"]["checks"]["measured"] is True
    # "Flagged images only" leaves clean images unchecked.
    assert images["plain.png"]["signals"]["checks"]["available"] is False
    assert images["doubt.png"]["score"] > images["unsure.png"]["score"] > images["plain.png"]["score"]


async def test_run_pauses_itself_when_the_server_stops_answering(dataset, settings):
    dead = dict(settings, base_url="http://127.0.0.1:9/v1")
    project = Project(dataset)
    runner = Runner(project, dead)
    runner.start()
    for _ in range(200):
        await asyncio.sleep(0.05)
        if runner.state == "paused":
            break
    assert runner.state == "paused" and "not answering" in runner.message
    assert runner.failed == 0  # nothing is thrown onto the retry list for a server outage
    runner.app.update(base_url=settings["base_url"])
    runner.client_factory = lambda _: ModelClient(settings)
    runner.stop()
    await runner.wait()
    assert runner.state == "stopped"


async def test_recompose_only_never_looks_at_images_again(dataset, settings, mock_url):
    project, _, _ = await run(dataset, settings)
    requests = mock_url[1].state.mock["requests"]
    requests.clear()
    ds = project.store.settings()
    ds["prefill_id"] = "character"
    project.store.save_settings(ds)
    runner = Runner(project, settings)
    runner.start(mode="compose")
    await runner.wait()
    assert runner.done == 3
    assert all("image_url" not in str(r["body"]) for r in requests)
    # The Character prefill lets the style field into the caption.
    assert "digital illustration" in (dataset / "plain_nl.txt").read_text().lower()


async def test_lmstudio_gets_probabilities_through_responses(dataset, settings, lmstudio_url):
    lm = dict(settings, backend="lmstudio", base_url=lmstudio_url[0], caps={})
    result = await probe_connection(lm)
    assert result["ok"]
    caps = result["caps"]
    assert caps["chat_logprobs"] is False and caps["responses_logprobs"] is True
    assert caps["responses_image"] == "base64" and caps["responses_schema"] is True
    lm["caps"] = caps
    requests = lmstudio_url[1].state.mock["requests"]
    requests.clear()
    project, runner, _ = await run(dataset, lm)
    assert runner.done == 3
    assert any(r["path"] == "responses" for r in requests)
    assert "min_p" not in str([r["body"] for r in requests])  # LM Studio has no min_p, so it is never sent
    reasons = combine(by_name(project)["unsure.png"]["signals"], lm["score_weights"])[1]
    assert any(r["signal"] == "uncertain" for r in reasons)


async def test_without_probabilities_checks_fall_back_to_plain_yes_no(dataset, settings, lmstudio_url):
    make_image(dataset / "doubt.png", (9, 99, 9), tags="1girl, mock_doubt, mock_contradict, red_hair")
    lm = dict(settings, backend="lmstudio", base_url=lmstudio_url[0], caps={"tested": True})
    project, _, _ = await run(dataset, lm, field_checks="all")
    signals = by_name(project)["doubt.png"]["signals"]
    assert signals["uncertain"]["available"] is False
    assert signals["checks"]["measured"] is False
    assert signals["checks"]["values"]["clothing"] == 0.0


@pytest.mark.parametrize("key", ["sk-unsloth-test"])
async def test_api_key_is_sent(tmp_path, settings, key):
    from candycaption.mock_server import create_app
    from conftest import _serve

    server, url = _serve(create_app(api_key=key))
    try:
        refused = await probe_connection(dict(settings, base_url=url, backend="unsloth"))
        assert not refused["ok"] and "API key" in refused["lines"][0]["text"]
        accepted = await probe_connection(dict(settings, base_url=url, backend="unsloth", api_key=key))
        assert accepted["ok"]
    finally:
        server.should_exit = True


async def test_pause_finishes_images_in_flight_then_resume_and_interrupted_run(tmp_path, settings):
    from candycaption.mock_server import create_app
    from conftest import _serve

    folder = tmp_path / "many"
    folder.mkdir()
    for n in range(8):
        make_image(folder / f"img{n}.png", (n * 30, 40, 200 - n * 20), tags="1girl")
    server, url = _serve(create_app(delay=0.15))
    slow = dict(settings, base_url=url)
    try:
        project = Project(folder)
        ds = project.store.settings()
        ds.update(parallel=2, overwrite=True, field_checks="off")
        project.store.save_settings(ds)
        runner = Runner(project, slow)
        runner.start()
        while runner.done < 1:
            await asyncio.sleep(0.02)
        runner.pause()
        while runner.state != "paused":
            await asyncio.sleep(0.02)
        done_at_pause = runner.done
        assert not runner.in_flight and 1 <= done_at_pause < 8
        await asyncio.sleep(0.5)
        assert runner.done == done_at_pause            # nothing new starts while paused
        runner.resume()
        while runner.done < done_at_pause + 2:
            await asyncio.sleep(0.02)
        runner.stop()
        await runner.wait()
        stopped_at = runner.done
        assert runner.state == "stopped" and stopped_at < 8
        assert runner.status()["interrupted"]["run_id"] == runner.run_id

        # Resuming an interrupted overwrite run skips what that run already finished.
        again = Runner(project, slow)
        again.start(resume=True)
        await again.wait()
        assert again.state == "done" and again.skipped == stopped_at and again.done == 8 - stopped_at
        assert again.status()["interrupted"] is None
        assert again.status()["seconds_per_image"] > 0
    finally:
        server.should_exit = True
