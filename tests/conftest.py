"""Shared fixtures. Every test talks to the bundled mock model server, so no GPU is needed."""

import socket
import threading
import time

import pytest
import uvicorn
from PIL import Image, ImageDraw

from candycaption.mock_server import create_app


def _serve(app):
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    threading.Thread(target=server.run, daemon=True).start()
    for _ in range(100):
        try:
            socket.create_connection(("127.0.0.1", port), timeout=0.1).close()
            break
        except OSError:
            time.sleep(0.05)
    return server, f"http://127.0.0.1:{port}/v1"


@pytest.fixture(scope="session")
def mock_url():
    """A llama.cpp-like mock: token probabilities on chat completions."""
    app = create_app()
    server, url = _serve(app)
    yield url, app
    server.should_exit = True


@pytest.fixture(scope="session")
def lmstudio_url():
    """An LM Studio-like mock: token probabilities only on /v1/responses."""
    app = create_app(chat_logprobs=False)
    server, url = _serve(app)
    yield url, app
    server.should_exit = True


@pytest.fixture(autouse=True)
def data_dir(tmp_path, monkeypatch):
    path = tmp_path / "appdata"
    monkeypatch.setenv("CANDYCAPTION_DATA", str(path))
    return path


@pytest.fixture
def settings(mock_url):
    from candycaption.config import load_settings

    values = load_settings()
    values.update(backend="llamacpp", base_url=mock_url[0], model="mock-captioner", tokenizer_repo="")
    return values


def make_image(path, colour=(200, 60, 60), size=(900, 900), tags=None, mode="RGB"):
    fill = colour + ((255,) if mode == "RGBA" else ())
    img = Image.new(mode, size, fill)
    ImageDraw.Draw(img).rectangle([size[0] // 4, size[1] // 4, size[0] // 2, size[1] // 2], fill=tuple(255 - c for c in colour))
    img.save(path)
    if tags is not None:
        path.with_suffix(".txt").write_text(tags, encoding="utf-8")
    return path


@pytest.fixture
def dataset(tmp_path):
    folder = tmp_path / "dataset"
    folder.mkdir()
    make_image(folder / "plain.png", (200, 60, 60), tags="1girl, red_hair, blue_eyes")
    make_image(folder / "unsure.png", (60, 200, 60), tags="1girl, mock_unsure")
    make_image(folder / "clash.png", (60, 60, 200), tags="2girls, red_hair, mock_contradict")
    return folder
