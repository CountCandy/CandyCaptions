from candycaption.cli import main
from candycaption.config import save_settings


def test_cli_shares_the_core(dataset, settings, capsys):
    save_settings(dict(settings, model=""))
    assert main(["caption", str(dataset)]) == 2                       # no model chosen
    args = ["caption", str(dataset), "--backend", "llamacpp", "--base-url", settings["base_url"], "--model", "mock-captioner"]
    assert main(args + ["--prefill", "character", "--trigger", "kyarine", "--parallel", "2"]) == 0
    out = capsys.readouterr().out
    assert "Character prefill" in out and "Batch done: 3 captions wrapped." in out
    assert (dataset / "plain_nl.txt").exists()

    assert main(args) == 0
    assert "Skipped 3 already captioned" in capsys.readouterr().out

    assert main(["health", str(dataset)]) == 0
    assert "3 images checked" in capsys.readouterr().out
    assert main(["review", str(dataset)]) == 0
    out = capsys.readouterr().out
    assert "clash.png" in out and "red_hair" in out
    assert main(["health", str(dataset / "nope")]) == 2
