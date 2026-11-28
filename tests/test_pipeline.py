import asyncio

import pytest

from candycaption.client import ModelClient
from candycaption.connect import test_connection as probe_connection
from candycaption.dataset import Project
from candycaption.prefills import find_banned
from candycaption.runner import Runner
from candycaption.signals import combine

from conftest import _serve, make_image


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
    total = 30   # far more than can finish before the stop below, so the test does not depend on timing
    for n in range(total):
        make_image(folder / f"img{n}.png", (n * 8, 40, 250 - n * 8), size=(320, 320), tags="1girl")
    server, url = _serve(create_app(delay=0.2))
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
        assert not runner.in_flight and 1 <= done_at_pause < total
        await asyncio.sleep(0.5)
        assert runner.done == done_at_pause            # nothing new starts while paused
        runner.resume()
        while runner.done < done_at_pause + 2:
            await asyncio.sleep(0.02)
        runner.stop()
        await runner.wait()
        stopped_at = runner.done
        assert runner.state == "stopped" and stopped_at < total
        assert runner.status()["interrupted"]["run_id"] == runner.run_id

        # Resuming an interrupted overwrite run skips what that run already finished.
        again = Runner(project, slow)
        again.start(resume=True)
        await again.wait()
        assert again.state == "done" and again.skipped == stopped_at and again.done == total - stopped_at
        assert again.status()["interrupted"] is None
        assert again.status()["seconds_per_image"] > 0
    finally:
        server.should_exit = True


async def test_connection_test_accepts_a_bare_address_and_rejects_a_wrong_path(settings):
    bare = settings["base_url"].removesuffix("/v1")
    assert (await probe_connection(dict(settings, base_url=bare, model="")))["ok"]

    # A server that answers 200 with something that is not a model list is not "connected".
    import httpx

    transport = httpx.MockTransport(lambda request: httpx.Response(200, json={"error": "Unexpected endpoint or method."}))
    wrong = await probe_connection(dict(settings, base_url="http://elsewhere/api"), transport=transport)
    assert not wrong["ok"] and "did not return a model list" in wrong["lines"][0]["text"]


async def test_a_reply_that_is_all_thinking_says_so(dataset, settings):
    import httpx

    def thinking_only(request):
        import json

        if json.loads(request.content).get("stream"):
            chunks = [{"choices": [{"delta": {"reasoning_content": "Let me think"}, "finish_reason": None}]},
                      {"choices": [{"delta": {}, "finish_reason": "length"}]}]
            body = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"
            return httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})
        return httpx.Response(200, json={"choices": [{"finish_reason": "length", "message": {
            "role": "assistant", "content": "", "reasoning_content": "Let me think about the notes..."}}]})

    client = ModelClient(settings, transport=httpx.MockTransport(thinking_only))
    reply = await client.complete("m", [{"role": "user", "content": "hi"}], sampling={}, max_tokens=50)
    assert reply.text == "" and reply.thought and "turn thinking off" in reply.why_empty()
    await client.close()

    def unfinished_think_tag(request):
        return httpx.Response(200, json={"choices": [{"finish_reason": "length", "message": {"content": "<think>The scarf is"}}]})

    client = ModelClient(settings, transport=httpx.MockTransport(unfinished_think_tag))
    reply = await client.complete("m", [{"role": "user", "content": "hi"}], sampling={}, max_tokens=50)
    assert reply.text == "" and reply.thought
    await client.close()

    project = Project(dataset)
    runner = Runner(project, settings, client_factory=lambda s: ModelClient(s, transport=httpx.MockTransport(thinking_only)))
    runner.start()
    await runner.wait()
    assert runner.failed == 3
    assert all("turn thinking off" in image["error"] for image in project.store.images())


async def test_a_fault_after_the_caption_is_written_never_loses_the_caption(dataset, settings, monkeypatch):
    from candycaption import pipeline
    from candycaption.client import ModelServerError

    async def broken_check(self, image_url, fields, keys=None):
        raise ModelServerError("http", "The model server refused the request (400): no")

    def broken_scoring(self, image, fields, reply, checks, previous=None):
        raise KeyError("surprise")

    monkeypatch.setattr(pipeline.Captioner, "check", broken_check)
    project, runner, _ = await run(dataset, settings, field_checks="all")
    assert runner.done == 3 and runner.failed == 0
    image = by_name(project)["plain.png"]
    assert image["status"] == "done" and (dataset / "plain_nl.txt").exists()
    assert "refused" in image["signals"]["checks"]["error"]

    monkeypatch.setattr(pipeline.Captioner, "signals_for", broken_scoring)
    project, runner, _ = await run(dataset, settings, overwrite=True)
    assert runner.done == 3 and runner.failed == 0
    image = by_name(project)["plain.png"]
    assert image["status"] == "done" and image["fields"] and "KeyError" in image["signals"]["error"]


# -- thinking --------------------------------------------------------------

def _thinking_server(**options):
    from candycaption.mock_server import create_app

    app = create_app(**options)
    server, url = _serve(app)
    return app, server, url


@pytest.mark.parametrize("options,control,seen,phrase", [
    # LM Studio as its developers describe it: reasoning_effort on chat completions.
    ({"thinking_switches": ("reasoning_effort",)}, "reasoning_effort", False, "switched per request"),
    # A llama.cpp-style server that is set to think but obeys the template switch.
    ({"thinking_switches": ("chat_template_kwargs",), "thinks": True}, "chat_template_kwargs", False, "switched per request"),
    # Thinking set on the server, and no switch obeyed: it cannot be stopped from here, only given room.
    ({"thinking_switches": (), "thinks": True}, None, True, "takes no per-request switch"),
    # No thinking at all.
    ({"thinking_switches": ()}, None, False, "does not think"),
])
async def test_connection_test_finds_how_thinking_is_switched(settings, options, control, seen, phrase):
    app, server, url = _thinking_server(**options)
    try:
        result = await probe_connection(dict(settings, base_url=url, backend="lmstudio"))
    finally:
        server.should_exit = True
    caps = result["caps"]
    assert result["ok"] and caps["thinking_control"] == control and caps["thinking_seen"] is seen
    assert any(phrase in line["text"] for line in result["lines"])
    # Whatever the server does about thinking, the rest of the test still got its answers.
    assert caps["json_schema"] and any("read the test image" in line["text"] for line in result["lines"])


async def test_thinking_is_used_for_the_look_step_only_and_never_empties_a_caption(dataset, settings):
    app, server, url = _thinking_server(thinking_switches=("reasoning_effort",))
    try:
        found = await probe_connection(dict(settings, base_url=url, backend="lmstudio"))
        settings = dict(settings, base_url=url, backend="lmstudio", caps=found["caps"])
        project = Project(dataset)
        project.scan()
        ds = project.store.settings()
        ds["sampling"].update(thinking=True, max_tokens_thinking=600)
        project.store.save_settings(ds)
        app.state.mock["requests"].clear()
        runner = Runner(project, settings)
        runner.start()
        await runner.wait()
    finally:
        server.should_exit = True
    bodies = [r["body"] for r in app.state.mock["requests"]]
    looks = [b for b in bodies if (b.get("response_format") or {}).get("json_schema", {}).get("name") == "caption_fields"
             or (b.get("text") or {}).get("format", {}).get("name") == "caption_fields"]
    others = [b for b in bodies if b not in looks]
    effort = lambda b: b.get("reasoning_effort") or (b.get("reasoning") or {}).get("effort")
    assert looks and all(effort(b) == "medium" for b in looks)
    assert others and all(effort(b) == "none" for b in others)
    limit = lambda b: b.get("max_tokens") or b.get("max_output_tokens")
    assert all(limit(b) >= 1800 + 600 for b in looks)                 # the fields limit plus the thinking room
    done = [i for i in project.store.images() if i["status"] == "done"]
    assert done and all(project.read_caption(i).strip() for i in done)


async def test_a_reply_used_up_by_thinking_is_asked_for_again_with_more_room(settings):
    # The server thinks whatever it is told, and the app has not been told (no Test connection yet).
    app, server, url = _thinking_server(thinking_switches=(), thinks=True, thinking_tokens=900)
    try:
        client = ModelClient(dict(settings, base_url=url, backend="lmstudio"))
        reply = await client.complete("mock-captioner", [{"role": "user", "content": "Say ok."}],
                                      sampling={"temperature": 0}, max_tokens=24)
        await client.close()
    finally:
        server.should_exit = True
    sent = [r["body"]["max_tokens"] for r in app.state.mock["requests"]]
    assert sent == [24, 24 + 2048] and reply.text and reply.thought and not reply.cut_off


@pytest.mark.parametrize("accepts", [True, False])
async def test_thinking_and_token_probabilities_on_an_lm_studio_style_server(settings, accepts):
    """Probabilities only on /v1/responses, the thinking switch proven on chat completions."""
    app, server, url = _thinking_server(chat_logprobs=False, thinking_switches=("reasoning_effort",), responses_reasoning=accepts)
    try:
        found = await probe_connection(dict(settings, base_url=url, backend="lmstudio"))
        caps = found["caps"]
        assert caps["thinking_control"] == "reasoning_effort" and caps["responses_logprobs"] and caps["responses_reasoning"] is accepts
        client = ModelClient(dict(settings, base_url=url, backend="lmstudio", caps=caps))
        ask = [{"role": "user", "content": "Say ok."}]
        quiet = client.build("mock-captioner", ask, sampling={"thinking": True}, max_tokens=50, want_logprobs=True, think=False)
        loud = client.build("mock-captioner", ask, sampling={"thinking": True}, max_tokens=50, want_logprobs=True, think=True)
        await client.close()
    finally:
        server.should_exit = True
    if accepts:
        assert quiet[0] == loud[0] == "responses"
        assert quiet[1]["reasoning"] == {"effort": "none"} and loud[1]["reasoning"] == {"effort": "medium"}
    else:
        # The switch is never sent where it was refused. A request that thinks goes to chat completions instead.
        assert quiet[0] == "responses" and "reasoning" not in quiet[1]
        assert loud[0] == "chat" and loud[1]["reasoning_effort"] == "medium" and loud[1]["max_tokens"] == 50 + 2048
        assert any("does not take the thinking switch" in line["text"] for line in found["lines"])
