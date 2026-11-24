# CandyCaption 🍬

CandyCaption is a local web app that captions a LoRA dataset one image at a time through any
OpenAI-compatible model server, then shows which captions are most likely wrong. It is built first for
MOSS-Image LoRAs (style, character and concept) trained with BluFlow-Trainer. Images only ever go to the
model server you connect, normally on the same PC.

For every image it writes `<image>_nl.txt` next to the image and its tag file `<image>.txt`, which is the
layout BluFlow reads. Remember `caption_mode = "mixed"` in BluFlow's `[dataset.caption]`, or it trains on
tags only.

## What you need

- Windows 11 or Linux with an NVIDIA GPU. CandyCaption itself is light; the GPU is for the model server.
- [Git](https://git-scm.com/downloads), to download and update the app.
- [uv](https://docs.astral.sh/uv/getting-started/installation/), which installs Python and the app's
  dependencies by itself.
- A model server with a vision model. These instructions use [LM Studio](https://lmstudio.ai/download).
  Unsloth Studio, llama.cpp server and vLLM work too.

CandyCaption does not load or host models. The model server does that.

## Install

Open a terminal (PowerShell on Windows) in the folder where you want the app to live:

```
git clone https://github.com/CountCandy/CandyCaptions.git
cd CandyCaptions
uv sync
```

`git clone` downloads the app into a new `CandyCaptions` folder. `uv sync` installs its dependencies into
a `.venv` folder inside it; the first time takes a minute.

## Download the models

The design targets a 96 GB card, which fits a captioner and a separate verifier at 8-bit with room for
parallel requests. On a smaller card, use one model for both jobs or pick a smaller quantization.

| Role | Model | LM Studio key |
| --- | --- | --- |
| Captioner | Gemma 4 31B, the family MOSS-Image's own captions came from | `google/gemma-4-31b` |
| Verifier (optional) | Qwen3.5-27B, a different family so it does not share Gemma's blind spots | `qwen/qwen3.5-27b` |

Check on the model's LM Studio page that the verifier you pick accepts images. Field checks send it one.

1. Install LM Studio and open it once. That also installs its command-line tool, `lms`.
2. Download the models. Either search for them in LM Studio's Discover tab and pick the 8-bit (Q8) file,
   or use the terminal, which lists the quantizations to choose from:

   ```
   lms get google/gemma-4-31b --always-show-download-options
   lms get qwen/qwen3.5-27b --always-show-download-options
   ```

3. Load them. The identifier is the name CandyCaption will show in its model dropdowns:

   ```
   lms load google/gemma-4-31b --identifier captioner --gpu max --context-length 16384
   lms load qwen/qwen3.5-27b --identifier verifier --gpu max --context-length 16384
   ```

   You can also load them in the app. `lms ls` lists what is on disk and `lms ps` what is loaded.
4. In LM Studio's settings for each loaded model, turn **thinking off** and set **Max Concurrent
   Predictions** to the Batch Size you plan to use in CandyCaption (4 is the default for both).
5. Start the server: `lms server start`, or the Developer tab in the app. It listens on
   http://localhost:1234/v1.

The context length has to hold one image, the instructions and the reply for every request in flight.
`scripts/probe_lmstudio.py` (below) measures what one image costs and whether LM Studio splits the context
between parallel requests, so treat 16384 as a starting point.

## Run

- **Windows:** double-click `run.bat`. The page opens at http://localhost:8200 once the app is up.
- **Linux:** `./run.sh`

Then work through the tabs from left to right. In the Candy Shop tab, press **Test connection** before the
first run: it sends one tiny image and records what your server supports.

| Tab | Plain name | What you do there |
| --- | --- | --- |
| Candy Shop | Connect | Pick the backend, the captioner and the optional verifier, then Test connection. |
| Recipe Box | Prefills | Edit the Style, Character and Concept prefills: prompts, rules, fields and their order, option toggles, banned words. |
| Pantry | Dataset | Open a folder. Health checks run at once. Each image has an optional hint line: one sentence telling the model what is in it, given as fact whenever that image is captioned. |
| Taste Test | Test bench | Pin 5 to 10 hard images, give an image a Hint if the model needs help, write a few answer-key facts, and compare recipe versions on them before a full run. |
| Candy Factory | Run | Start, pause, resume and stop the batch, and watch it live. |
| Sorting Table | Review | The doubtful captions, worst first, with the field in doubt named. Fix them by hand or by talking to the model. All caption editing is here, including find and replace. |

## Update

### The app

Close CandyCaption first (close its window, or press Ctrl+C in its terminal). Then, in the `CandyCaptions`
folder:

```
git pull
uv sync
```

`git pull` downloads the latest version from GitHub and `uv sync` brings the dependencies in line with it.
Start the app again with `run.bat` or `./run.sh`.

- Your settings, prefills, project stores and backups live in the `data` folder, which Git ignores, so an
  update never touches them. Your datasets are outside the app folder altogether.
- If `git pull` says your local changes would be overwritten, you have edited a file that is part of the
  app. `git stash` sets your edits aside, `git pull` then works, and `git stash pop` puts them back.
- `git log -1` shows which version you are on. To go back to an earlier one, `git checkout <commit>`
  then `uv sync`; `git switch main`, `git pull` and `uv sync` return you to the latest.

### The models

Models do not update themselves. To move to a newer upload or another quantization of the same model, run
`lms get` again (or download it in the Discover tab), then unload the old one and load the new one:

```
lms get google/gemma-4-31b --always-show-download-options
lms unload captioner
lms load google/gemma-4-31b --identifier captioner --gpu max --context-length 16384
```

Keeping the same identifier means CandyCaption needs no change. Press **Test connection** again after any
model or LM Studio update, because what the server supports can change. Old model files you no longer
want can be deleted from LM Studio's My Models page. LM Studio itself updates from inside the app.

A new model changes how captions come out. Rerun your test set in the Taste Test tab before captioning a
whole dataset with it.

## Trying the app with a fake model server

Real use always has a GPU behind it. For working on the app itself, and for the automated tests, a small
fake model server is included. It cannot see
images, but it answers like a model would:

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
| Mixed Flavors | uncertain words | the model wavered between two different facts, such as red or orange | token probabilities |
| Taste Check | field checks | a field value the model, or the verifier, will not confirm when asked yes or no | one short request per field |
| Flavor Clash | tag contradictions | a field disagrees with the tagger | a tag file |
| Same Old Flavor | template repetition | the caption repeats long phrases found in many others | nothing |

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
uv run pytest          # every test talks to the fake model server, so GitHub can run them
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
├── backends.py     what each backend honors
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

## License

MIT.
