# CandyCaption - Agent Guidelines

## Running commands

Always use `uv run`:

```bash
uv run pytest                    # tests; they use the bundled mock model server, no GPU
uv run candycaption serve        # the web GUI on :8200
uv run candycaption mock         # fake OpenAI-style model server on :1234
```

## Rules the code keeps

- The browser never talks to the model server. Every model request goes through `client.py`, holds
  exactly one image, sets `stream` explicitly, and resends no earlier messages.
- A sampling setting the backend ignores (`backends.py`) is never sent and is grayed out in the UI.
- Fields are data: prefills carry their own field list and the JSON schema is built from it.
- Nudity handling lives only in prefill text (`prefills.py`). No other module mentions it; a test checks this.
- The `_nl.txt` files are the source of truth. The SQLite store only describes them and lives in the
  app's data folder, never in the dataset.
- Review chat is the only place a conversation exists, it covers one image, and it is discarded on close.
  Promote-to-rule is the only way information moves between images.
- Every candy name in the UI shows its plain name next to it. Errors say what is wrong in plain words.
- TagGUI is GPL-3.0: borrow ideas, not code.

## Frontend

Plain HTML, CSS and JS in `static/`, no build step. `core.js` has the helpers (`h()` builds DOM nodes);
each tab has its own file and registers itself in `App.tabs`.
