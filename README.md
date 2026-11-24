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
   Predictions** to the Batch Size you plan to use in CandyCaption (4 is the default for both). With
   thinking on, a model can spend its whole reply reasoning and return an empty caption; CandyCaption
   says so when that happens.
5. Start the server: `lms server start`, or the Developer tab in the app. It listens on
   http://localhost:1234/v1. If you paste the address without `/v1` into CandyCaption, it adds it.

The context length has to hold one image, the instructions and the reply for every request in flight.
`scripts/probe_lmstudio.py` (below) measures what one image costs and whether LM Studio splits the context
between parallel requests, so treat 16384 as a starting point.

## Run

- **Windows:** double-click `run.bat`. The page opens at http://localhost:8200 once the app is up.
- **Linux:** `./run.sh`

The tabs are numbered in working order, and a line under the tab bar says what the open one is for. In
the Candy Shop tab, press **Test connection** before the first run: it sends one tiny image and records
what your server supports.

| Tab | Plain name | What you do there |
| --- | --- | --- |
| Candy Shop | 1 · Connect | Pick the backend, the Captioner and the optional Taste Tester (verifier), then Test connection. Also the token counter (Wrapper Size) and the finish sound (Gumdrop Bell). |
| Recipe Box | 2 · Prefills | Edit the Style, Character and Concept recipes: prompts, rules, fields and their order, option toggles, banned words. |
| Pantry | 3 · Dataset | Open a folder. In the Browse window you can set the folder it starts in. Health checks (Freshness Check) run at once. Each image has an optional hint line. |
| Taste Test | 4 · Test bench | Pin 5 to 10 hard images, give them hints and answer-key facts, and compare recipe versions on them before a full run. The newest 8 test batches are kept, each with the model that made it. |
| Candy Factory | 5 · Run | Start, pause, resume and stop the batch, and watch it live. |
| Sorting Table | 6 · Review | The doubtful captions with the field in doubt named, by file name or worst first. Fix them by hand or by talking to the model. All caption editing is here, including find and replace. |
| Flavor Profile | Extra · Style | For style datasets: the images that do not fit the rest of the set, and pairs of images or captions that are too alike. |

Lists are in file-name order everywhere, with numbers read as numbers, so `ta (2)` comes before `ta (16)`.
Click any picture to see it large.

### Hints

A vision model misreads some images every time: a small figure, an odd pose, an object it does not know.
A **Hint** is one line you write for such an image, such as "An elf girl sits on a tree stump and plays an
ocarina." It is given to the model as fact whenever that image is captioned, in the Taste Test and in the
Candy Factory, and it stays with the image. Fill hints in from the Pantry list before a batch, only for the
images that need one.

**Tasting Notes** in the Taste Test are the opposite: an answer key the model never sees, used only to
score test batches.

## Update

### The app

Close CandyCaption first (close its window, or press Ctrl+C in its terminal). Then, in the `CandyCaptions`
folder:

```
git pull
uv sync
```

`git pull` downloads the latest version from GitHub and `uv sync` brings the dependencies in line with it.
Start the app again with `run.bat` or `./run.sh`. The page picks up the new version by itself. The first
time you open a folder after some updates, its images are measured again, which takes a moment.

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

Put `mock_unsure`, `mock_contradict`, `mock_doubt`, `mock_bad_json`, `mock_flaky`, `mock_banned`,
`mock_empty` or `mock_error` in an image's tag file to make the fake server waver, disagree with the
tagger, or fail for that image.

## How a caption is made

Each image gets two short requests, and every request stands alone with exactly one image in it. The one
exception in the whole app is **How does it differ?** in the Flavor Profile tab, which shows the model four
images at once so it can compare them.

1. **Look.** The image (turned upright, transparency filled, downscaled), its tags and its hint, if it
   has one, go to the model, which fills in twelve fields as JSON.
2. **Compose.** A text-only request turns the fields the prefill keeps into one paragraph of 40 to 70
   words. Switching prefill later only reruns this step (**Recompose only**).
3. Banned words trigger one rewrite, the caption is written to disk at once, and the fields are scored.
   Captions are written in American English.

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

Review and the health checks count tags plus description against a token limit you set in the Candy Shop
tab (Wrapper Size). It starts at 512, BluFlow's default; that is the trainer's setting, not a limit of the
model, so raise it if you raise it in the trainer. The
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
├── style.py        Flavor Profile: source-quality pass, style lens, ringer test
├── similar.py      Flavor Profile: pairs of images or captions that are too alike
├── store.py        the per-dataset SQLite store
├── testbench.py    test bench runs, fact scoring, promote
├── filters.py      review filters with AND, OR, NOT
├── tokens.py       token counter
├── imaging.py      image preparation and hashes
└── mock_server.py  the fake model server
static/             the page: plain HTML, CSS and JS, no build step
```

The look started from [Candy Echo](https://github.com/CountCandy/candyecho): its frosted panels and
rock-candy progress bar. There are two themes, both dark: Candy (plum, the default) and Midnight (blue).
The button in the top right corner switches between them.

## Flavor Profile: the odd one out, and two of a kind

For style datasets. Everything is relative to the folder you opened, and the result is a ranked shortlist
for your eye, not a verdict. A set needs at least 8 images.

- **Quality Control** (source quality) needs no model. It measures sharpness, resolution, JPEG block
  pattern and grain, and flags an image only when it stands apart from the rest. Compression the whole set
  shares is not flagged.
- **Odd One Out** uses the model: a short style note per image, one style summed up for the set, then one
  yes or no question per image, "is this made in that style?". Palette, light and texture measured from the
  pixels give a second opinion. **How does it differ?** shows a flagged image next to three typical ones.
- **Pick and Mix** (images to check) limits the checks above to the images you choose.
- **Gold Standard** lets you mark a few images as exactly the style you are aiming for. The style is then
  written from those alone and every image is checked against it, instead of against what most of the set
  shares. Use it when a set mixes two styles or when most of the set is not what you want. It costs one
  request per image however many targets you mark.
- **Plant a Ringer** ranks an image you know does not belong, without adding it to the dataset. If your
  ringer does not come out near the top, do not trust the ranking for that kind of difference.
- **Two of a Kind** lists pairs that are too alike, across the whole folder and without the model. Pictures:
  the same image resized, recolored, mirrored, cropped or with a part changed. Captions: two that say nearly
  the same thing, with words every caption uses counting for little. Two sliders set how alike a pair must
  be. It does not see the same pose drawn differently. Nothing is deleted; **These are fine** stops listing
  a pair.

## Not built yet

A style-embedding model (CSD) as a third opinion for subtle differences, dataset-specific banned words
from the style card, the tag editor and auto-tagger, the watched folder and the BluFlow export.

## License

MIT.
