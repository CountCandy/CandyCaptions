# CandyCaption 🍬

CandyCaption is a local web app that captions a LoRA dataset one image at a time through any
OpenAI-compatible model server, then shows which captions are most likely wrong. It is built first for
MOSS-Image LoRAs (style, character and concept) trained with BluFlow-Trainer. Images only ever go to the
model server you connect, normally on the same PC.

For every image it writes `<image>_nl.txt` next to the image and its tag file `<image>.txt`, which is the
layout BluFlow reads. Remember `caption_mode = "mixed"` in BluFlow's `[dataset.caption]`, or it trains on
tags only.

## Start

You need [uv](https://docs.astral.sh/uv/) and a running model server (LM Studio, Unsloth Studio, llama.cpp
or vLLM) with a vision model loaded. CandyCaption does not load or host models.

- **Windows:** double-click `run.bat`. The page opens at http://localhost:8200 once the app is up.
- **Linux:** `./run.sh`

Then work through the tabs from left to right:

| Tab | Plain name | What you do there |
| --- | --- | --- |
| Candy Shop | Connect | Pick the backend, models and the optional verifier. **Test connection** sends one tiny image and records what the server supports. |
| Recipe Box | Prefills | Edit the Style, Character and Concept prefills: prompts, rules, fields and their order, option toggles, banned words. |
| Taste Test | Test bench | Pin 5 to 10 hard images, write a few facts for each, and compare recipe versions on them before a full run. |
| Pantry | Dataset | Open a folder. Health checks run at once. Find and replace lives here too. |
| Candy Factory | Run | Start, pause, resume and stop the batch, and watch it live. |
| Sorting Table | Review | The doubtful captions, worst first, with the field in doubt named. Fix them by hand or by talking to the model. |

### Try it without a GPU

A small fake model server is included. It cannot see images, but it answers like a model would:

```
uv run candycaption mock            # fake server on http://localhost:1234/v1
uv run candycaption serve           # the app on http://localhost:8200
```

Put `mock_unsure`, `mock_contradict`, `mock_doubt`, `mock_bad_json`, `mock_flaky` or `mock_banned` in an
image's tag file to make the fake server waver, disagree with the tagger, or fail for that image.

## How a caption is made

Each image gets two short requests, and every request stands alone with exactly one image in it.

1. **Look.** The image (turned upright, transparency filled, downscaled) and its tags go to the model,
   which fills in twelve fields as JSON.
2. **Compose.** A text-only request turns the fields the prefill keeps into one paragraph of 40 to 70
   words. Switching prefill later only reruns this step (**Recompose only**).
3. Banned words trigger one rewrite, the caption is written to disk at once, and the fields are scored.

Four measured signals flag captions for review. The model is never asked to rate its own confidence.

| Flag | Plain name | Catches | Needs |
| --- | --- | --- | --- |
| Mixed Flavours | uncertain words | the model wavered between two different facts, such as red or orange | token probabilities |
| Taste Check | field checks | a field value the model, or the verifier, will not confirm when asked yes or no | one short request per field |
| Flavour Clash | tag contradictions | a field disagrees with the tagger | a tag file |
| Same Old Flavour | template repetition | the caption repeats long phrases found in many others | nothing |

Which word pairs count as "different facts" and which tags are checked against which field are editable
lists (Recipe Box, at the bottom).

## Command line

The command-line mode shares the captioning core with the web GUI, and uses the connection saved there
unless you override it.

```
uv run candycaption caption D:\datasets\my_lora --model gemma-4-31b --prefill character --trigger kyarine
uv run candycaption health  D:\datasets\my_lora
uv run candycaption review  D:\datasets\my_lora
uv run candycaption caption --help
```

## LM Studio probe

`scripts/probe_lmstudio.py` answers the open questions about LM Studio on your own machine: images on
`/v1/responses`, which sampling settings and JSON schema work there, where token probabilities come from,
and whether context is split across parallel requests.

```
uv run python scripts/probe_lmstudio.py --parallel 4
```

It prints a report and saves `probe_report.json`.

## Token counter

Review and the health checks count tags plus description against BluFlow's 512-token default. The
tokenizer is looked for in this order: the local folder set in the Candy Shop tab (BluFlow bundles the
MOSS/Qwen tokenizer), the cached download in `data/tokenizer`, a first-run download from Hugging Face.
Only if none of those works is the count approximate, and the page says so.

## Where things are kept

Everything the app remembers is under `data/` next to the launcher (or `CANDYCAPTION_DATA`): settings,
prefills, one SQLite project store per dataset, backups of images fixed in place, and the tokenizer cache.
Nothing but `_nl.txt` and tag files is ever written into a dataset folder, and the `_nl.txt` files stay the
source of truth.

## Development

```
uv sync --extra dev
uv run pytest          # every test talks to the mock model server; no GPU needed
```

```
src/candycaption/
├── server.py       FastAPI app the browser talks to
├── cli.py          command-line mode
├── runner.py       the queue: parallel requests, pause/resume, retry list, live progress
├── pipeline.py     look, compose, field checks, scoring, review chat
├── prefills.py     prefills, fields, presets, and the requests built from them
├── signals.py      the four review signals
├── wordlists.py    editable word lists and tag map
├── client.py       OpenAI-compatible client (chat completions and /v1/responses)
├── backends.py     what each backend honours
├── connect.py      Test connection
├── dataset.py      scanning, caption files, health checks, find and replace
├── store.py        the per-dataset SQLite store
├── testbench.py    test bench runs, fact scoring, promote
├── filters.py      review filters with AND, OR, NOT
├── tokens.py       token counter
├── imaging.py      image preparation and hashes
└── mock_server.py  the fake model server
static/             the page: plain HTML, CSS and JS, no build step
```

The look comes from [Candy Echo](https://github.com/CountCandy/candyecho): its three themes, frosted
panels and rock-candy progress bar.

## Not in v1

The Style tab (style card, CSD fit check), the tag editor and auto-tagger, the watched folder and the
BluFlow export are planned for v2.

## Licence

MIT.
