"""Two of a Kind: pairs of images, or of captions, that are too alike.

Nothing here uses the model. Pictures are compared through a small grayscale
copy kept with each image's measurements; captions are compared by the words
they share, with words every caption uses counting for little.
"""

from __future__ import annotations

import base64
import math
import re
from collections import Counter
from functools import lru_cache
from typing import Any

import numpy as np

from .dataset import Project
from .imaging import SIGNATURE_SIDE

CELL = 16                    # every comparison happens on a 16 x 16 grid
PICTURE_FLOOR = 0.75         # pairs under these are never worth showing
CAPTION_FLOOR = 0.40
MAX_PAIRS = 400
MAX_FITS = 800              # crop fits are the slow part; only the likeliest pairs get one
KINDS = {
    "identical": "the same file twice",
    "same": "same picture",
    "edited": "same picture with a part changed",
    "mirrored": "mirrored",
    "crop": "one is a crop of the other",
}


def _spans() -> list[tuple[int, int]]:
    """The whole side, then shorter stretches of it at the start, middle and end."""
    side = SIGNATURE_SIDE
    spans = [(0, side)]
    for part in (0.85, 0.7, 0.55):
        length = round(side * part)
        spans += [(0, length), ((side - length) // 2, (side - length) // 2 + length), (side - length, side)]
    return spans


def _averager(start: float, stop: float) -> np.ndarray:
    """Matrix that shrinks the stretch start..stop of a signature row to CELL values by averaging."""
    matrix = np.zeros((CELL, SIGNATURE_SIDE), dtype=np.float32)
    width = (stop - start) / CELL
    for cell in range(CELL):
        low, high = start + cell * width, start + (cell + 1) * width
        for pixel in range(max(0, int(math.floor(low))), min(SIGNATURE_SIDE, int(math.ceil(high)))):
            matrix[cell, pixel] = max(0.0, min(high, pixel + 1) - max(low, pixel)) / width
    return matrix


_SPANS = _spans()
_AVERAGERS = [_averager(*span) for span in _SPANS]
_WINDOWS = [(y, x) for y in range(len(_SPANS)) for x in range(len(_SPANS)) if (y, x) != (0, 0)]
_WINDOW_Y = np.stack([_AVERAGERS[y] for y, _ in _WINDOWS])
_WINDOW_X = np.stack([_AVERAGERS[x] for _, x in _WINDOWS])


def _unit(vectors: np.ndarray) -> np.ndarray:
    """Remove each row's mean and scale it to length one, so a dot product is a correlation."""
    vectors = vectors - vectors.mean(axis=-1, keepdims=True)
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return np.where(norms > 1e-3, vectors / np.maximum(norms, 1e-6), 0.0).astype(np.float32)


def decode(signature: str) -> np.ndarray | None:
    try:
        grid = np.frombuffer(base64.b64decode(signature), dtype=np.uint8)
    except (ValueError, TypeError):
        return None
    if grid.size != SIGNATURE_SIDE * SIGNATURE_SIDE:
        return None
    return grid.reshape(SIGNATURE_SIDE, SIGNATURE_SIDE).astype(np.float32)


def _zscore(grid: np.ndarray) -> np.ndarray:
    grid = grid - grid.mean()
    return grid / max(float(grid.std()), 1e-3)


def _mostly_equal(a: np.ndarray, b: np.ndarray) -> float:
    """Share of the picture that is the same in both. One changed corner barely moves it."""
    return float((np.abs(_zscore(a) - _zscore(b)) < 0.3).mean())


@lru_cache(maxsize=4096)
def _cached_averager(start: float, stop: float) -> np.ndarray:
    return _averager(start, stop)


def _best_crop(big: np.ndarray, small: np.ndarray, window: tuple[float, float, float, float]) -> float:
    """Slide and resize a window over `big`, half a cell at a time, until it matches the whole of `small` best."""
    side = float(SIGNATURE_SIDE)

    def score(win: tuple[float, float, float, float]) -> float:
        y0, y1, x0, x1 = win
        if y0 < 0 or x0 < 0 or y1 > side or x1 > side or y1 - y0 < side * 0.4 or x1 - x0 < side * 0.4:
            return -1.0
        cut = _cached_averager(y0, y1) @ big @ _cached_averager(x0, x1).T
        return float(_unit(cut.reshape(1, -1))[0] @ small)

    best = score(window)
    for step in (2.0, 1.0, 0.5):
        for _ in range(8):
            moves = [tuple(v + (delta if n == edge else 0.0) for n, v in enumerate(window)) for edge in range(4) for delta in (-step, step)]
            scores = [score(move) for move in moves]
            top = int(np.argmax(scores))
            if scores[top] <= best + 1e-4:
                break
            best, window = scores[top], moves[top]
    return best


def _plainness(grid: np.ndarray) -> float:
    """How much to distrust a crop match: a picture of a few flat shapes fits a window of almost anything."""
    z = _zscore(_AVERAGERS[0] @ grid @ _AVERAGERS[0].T)
    detail = float(np.abs(np.diff(z, axis=0)).mean() + np.abs(np.diff(z, axis=1)).mean())
    return max(0.0, 0.6 - detail) * 0.5


def picture_pairs(grids: list[np.ndarray], floor: float = PICTURE_FLOOR) -> dict[tuple[int, int], tuple[float, str]]:
    """Every pair of pictures at least `floor` alike, as {(a, b): (score, kind)} with a < b."""
    count = len(grids)
    if count < 2:
        return {}
    stack = np.stack(grids)
    fine = _unit(stack.reshape(count, -1))                       # the whole signature, for "same picture"
    fine_mirrored = _unit(stack[:, :, ::-1].reshape(count, -1))
    whole_y = whole_x = _AVERAGERS[0]
    whole = _unit((whole_y @ stack @ whole_x.T).reshape(count, -1))
    shapes = [(sy, sx) for sy in _SPANS for sx in _SPANS if (sy, sx) != (_SPANS[0], _SPANS[0])]
    found: dict[tuple[int, int], tuple[float, str]] = {}
    crops: list[tuple[float, int, int, int]] = []               # (rough score, big, small, window number)

    def keep(a: int, b: int, score: float, kind: str) -> None:
        key = (min(a, b), max(a, b))
        if score >= floor and (key not in found or score > found[key][0]):
            found[key] = (min(1.0, score), kind)

    chunk = 32
    for start in range(0, count, chunk):
        part = stack[start:start + chunk]
        same = fine[start:start + chunk] @ fine.T
        flipped = fine[start:start + chunk] @ fine_mirrored.T
        # Every window of each picture in this chunk against every whole picture: is b a crop of a?
        cut = np.matmul(np.matmul(_WINDOW_Y[None], part[:, None]), _WINDOW_X.transpose(0, 2, 1)[None])
        windows = _unit(cut.reshape(len(part), len(shapes), -1))
        rough = (windows.reshape(-1, CELL * CELL) @ whole.T).reshape(len(part), len(shapes), count)
        for row in range(len(part)):
            a = start + row
            for b in np.nonzero(same[row] >= 0.6)[0]:
                b = int(b)
                if b > a:
                    plain = float(same[row, b])
                    patched = _mostly_equal(grids[a], grids[b]) if plain < 0.995 else 0.0
                    keep(a, b, max(plain, patched), "same" if plain >= patched else "edited")
            for b in np.nonzero(flipped[row] >= floor)[0]:
                if int(b) > a:
                    keep(a, int(b), float(flipped[row, b]), "mirrored")
            best = rough[row].max(axis=0)
            for b in np.nonzero(best >= 0.55)[0]:
                if int(b) != a:
                    crops.append((float(best[b]), a, int(b), int(rough[row, :, b].argmax())))
    # The rough windows only land near a crop. Fit the likeliest ones properly, most promising first.
    crops.sort(reverse=True)
    for rough_score, a, b, shape in crops[:MAX_FITS]:
        key = (min(a, b), max(a, b))
        if key in found and found[key][0] >= 0.97:
            continue
        (y0, y1), (x0, x1) = shapes[shape]
        # Picking the best of many windows flatters a pair a little, so a crop has to clear the bar with room.
        doubt = 0.02 + max(_plainness(grids[a]), _plainness(grids[b]))
        keep(a, b, _best_crop(grids[a], whole[b], (float(y0), float(y1), float(x0), float(x1))) - doubt, "crop")
    return found


_FILLER = frozenset("a an and are as at be by for from in into is it its of on or that the this to with".split())


def _stem(word: str) -> str:
    """Crude on purpose: enough for "sleeps", "sleeping" and "sleep" to count as one word."""
    for ending in ("ing", "es", "ed", "s"):
        if word.endswith(ending) and len(word) - len(ending) >= 4:
            return word[:-len(ending)]
    return word


def _terms(text: str) -> Counter:
    """The words that carry meaning, plus neighboring pairs at half weight so word order counts a little."""
    words = [_stem(w) for w in re.findall(r"[a-z0-9']+", text.lower()) if w not in _FILLER]
    terms: Counter = Counter(words)
    for a, b in zip(words, words[1:]):
        terms[f"{a} {b}"] += 0.5
    return terms


def caption_pairs(texts: list[str], floor: float = CAPTION_FLOOR) -> dict[tuple[int, int], float]:
    """Every pair of captions at least `floor` alike. Words that nearly every caption uses count for little."""
    counts = [_terms(text) for text in texts]
    filled = [n for n, c in enumerate(counts) if c]
    if len(filled) < 2:
        return {}
    spread = Counter(term for n in filled for term in counts[n])
    weight = {term: math.log((len(filled) + 1) / (seen + 1)) + 0.1 for term, seen in spread.items()}
    # A term only one caption uses cannot be shared, but it still makes that caption less like the others.
    lengths = [math.sqrt(sum((n * weight[t]) ** 2 for t, n in counts[i].items())) for i in filled]
    shared = [t for t, seen in spread.most_common(max(2000, 40_000_000 // len(filled))) if seen > 1]
    column = {term: n for n, term in enumerate(shared)}
    matrix = np.zeros((len(filled), len(shared)), dtype=np.float32)
    for row, i in enumerate(filled):
        for term, n in counts[i].items():
            if term in column:
                matrix[row, column[term]] = n * weight[term] / max(lengths[row], 1e-9)
    found = {}
    for start in range(0, len(filled), 256):
        alike = matrix[start:start + 256] @ matrix.T
        for row, other in zip(*np.nonzero(alike >= floor)):
            a, b = filled[start + int(row)], filled[int(other)]
            if a < b:
                found[(a, b)] = min(1.0, float(alike[row, other]))
    return found


def _key(a: dict[str, Any], b: dict[str, Any]) -> str:
    """Names the two pictures by content, so a pair marked as fine is asked about again if either is replaced."""
    return "-".join(sorted([a.get("sha") or f"id{a['id']}", b.get("sha") or f"id{b['id']}"]))


def dismiss(project: Project, a: int, b: int) -> None:
    first, second = project.store.image(a), project.store.image(b)
    if not first or not second:
        return
    fine = set(project.store.get_meta("similar_ok") or [])
    fine.add(_key(first, second))
    project.store.set_meta("similar_ok", sorted(fine))


def restore(project: Project) -> None:
    project.store.set_meta("similar_ok", None)


def find(project: Project) -> dict[str, Any]:
    """Pairs across the whole folder, most alike first, without the ones the user said are fine."""
    images = [i for i in project.store.images() if not (i.get("health") or {}).get("unreadable")]
    signatures = [((i.get("health") or {}).get("stats") or {}).get("sig") or "" for i in images]
    grids = [decode(signature) for signature in signatures]
    seen = [n for n, grid in enumerate(grids) if grid is not None]
    # Comparing the pictures takes a moment on a big folder, so the answer is kept until a picture changes.
    stamp = hash(tuple(signatures))
    cached = getattr(project, "_look_alikes", None)
    if not cached or cached[0] != stamp:
        cached = (stamp, picture_pairs([grids[n] for n in seen]))
        project._look_alikes = cached
    pictures = {(seen[a], seen[b]): hit for (a, b), hit in cached[1].items()}
    by_sha: dict[str, list[int]] = {}
    for n, image in enumerate(images):
        if image.get("sha"):
            by_sha.setdefault(image["sha"], []).append(n)
    for group in by_sha.values():
        for x in range(len(group)):
            for y in range(x + 1, len(group)):
                pictures[(group[x], group[y])] = (1.0, "identical")
    texts = [project.read_caption(i) for i in images]
    captions = caption_pairs(texts)

    fine = set(project.store.get_meta("similar_ok") or [])
    pairs = []
    for a, b in sorted(set(pictures) | set(captions)):
        if _key(images[a], images[b]) in fine:
            continue
        score, kind = pictures.get((a, b), (None, None))
        words = captions.get((a, b))
        pairs.append({
            "a": {"id": images[a]["id"], "rel": images[a]["rel"]}, "b": {"id": images[b]["id"], "rel": images[b]["rel"]},
            "picture": None if score is None else round(score, 3), "kind": kind, "kind_name": KINDS.get(kind or ""),
            "caption": None if words is None else round(words, 3),
            "captions": [texts[a], texts[b]] if words is not None else None,
        })
    # Most alike first, by whichever of the two measures is further above its own floor.
    def rank(pair: dict[str, Any]) -> float:
        return max(((pair["picture"] or 0) - PICTURE_FLOOR) / (1 - PICTURE_FLOOR), ((pair["caption"] or 0) - CAPTION_FLOOR) / (1 - CAPTION_FLOOR))
    pairs.sort(key=rank, reverse=True)
    return {
        "pairs": pairs[:MAX_PAIRS], "more": max(0, len(pairs) - MAX_PAIRS), "dismissed": len(fine),
        "images": len(seen), "captions": sum(1 for t in texts if t.strip()),
        "picture_floor": PICTURE_FLOOR, "caption_floor": CAPTION_FLOOR,
    }
