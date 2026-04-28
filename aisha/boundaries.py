"""
boundaries.py — three boundary-value scores for candidate Aisha outputs.

Each score is a continuous quality measure on a candidate word stream.
Higher is more admissible.  Used by `bvp_composer.py` to pick the best
candidate from a multi-sample pool.

  syntax(words)    : mean log-prob of POS bigrams under a reference
                     POS-bigram distribution from a representative
                     English corpus.
  semantic(words)  : mean pairwise α-cosine of content words in the
                     dictionary world manifold.  Captures
                     "do these content words mean compatible things?"
  context(words, route)
                   : negative KL of the output's mean π versus the
                     routed flow's typical-response π.  "Does the POS
                     shape look like a response in this register?"

The scores are corpus-derived; nothing here invents content.  These are
filters, not generators.
"""
from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from word_manifold import WordManifold        # noqa: E402

WORD_RE = re.compile(r"[a-zA-Z']+")
SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")
CONTENT_POS = (0, 1, 2, 3)   # Noun, Verb, Adj, Adv


def _to_idx(words, wm) -> list[int]:
    """Convert a list of lemma strings to manifold indices, dropping OOV."""
    idx: list[int] = []
    for w in words:
        if isinstance(w, int):
            idx.append(w); continue
        i = wm.idx.get(str(w).lower())
        if i is not None:
            idx.append(i)
    return idx


# ---------------------------------------------------------------------------
# Reference distribution loaders (cached, built once per process)
# ---------------------------------------------------------------------------

_REF_CACHE: dict = {}


def _pos_arg(wm: WordManifold) -> np.ndarray:
    """Cached argmax(pi, axis=1) per manifold.  Without this each
    boundary score call recomputes argmax on a 50k-row array — was the
    top hotspot in pool-mode profiling."""
    key = f"pos_arg:{id(wm)}"
    if key not in _REF_CACHE:
        _REF_CACHE[key] = wm.pi.argmax(axis=1)
    return _REF_CACHE[key]


def _load_world(name: str) -> WordManifold:
    key = f"world:{name}"
    if key not in _REF_CACHE:
        _REF_CACHE[key] = WordManifold.load(
            str(ROOT / "data" / "processed" / f"manifold_{name}.pkl"))
    return _REF_CACHE[key]


def _universal() -> WordManifold:
    if "universal" not in _REF_CACHE:
        _REF_CACHE["universal"] = WordManifold.load(
            str(ROOT / "data" / "processed" / "manifold.pkl"))
    return _REF_CACHE["universal"]


def _build_pos_bigram(wm_universal, sent_iter, n_max=5000) -> np.ndarray:
    """Empirical POS bigram distribution from a sentence corpus."""
    pos_arg = wm_universal.pi.argmax(axis=1)
    counts = np.zeros((8, 8), dtype=np.float64)
    n = 0
    for s in sent_iter:
        idx = _to_idx(WORD_RE.findall(s.lower()), wm_universal)
        if len(idx) < 2:
            continue
        seq = pos_arg[np.asarray(idx)]
        for a, b in zip(seq[:-1], seq[1:]):
            counts[a, b] += 1.0
        n += 1
        if n >= n_max:
            break
    counts += 1.0
    counts /= counts.sum(axis=1, keepdims=True)
    return counts


def _flow_response_pi(flow_wm, wm_universal, sent_iter,
                       n_max=2000) -> np.ndarray:
    """Mean π of typical responses in this flow corpus."""
    accum = np.zeros(8, dtype=np.float64)
    n = 0
    for s in sent_iter:
        idx = _to_idx(WORD_RE.findall(s.lower()), wm_universal)
        if not idx:
            continue
        accum += wm_universal.pi[np.asarray(idx)].mean(axis=0)
        n += 1
        if n >= n_max:
            break
    return accum / max(n, 1)


def _article_iter():
    text = (ROOT / "data" / "raw" / "wikipedia_pilot.txt").read_text(
        encoding="utf-8", errors="ignore")[:10_000_000]
    for line in text.split("\n"):
        line = line.strip()
        if line:
            for s in SENT_RE.split(line):
                s = s.strip()
                if 25 <= len(s) <= 200:
                    yield s


def _dialog_iter():
    with (ROOT / "data" / "conversations.csv").open() as f:
        for r in csv.DictReader(f):
            t = r["text"].strip()
            if 8 <= len(t) <= 200:
                yield t


def _universal_iter():
    yield from _article_iter()
    yield from _dialog_iter()


def get_refs() -> dict:
    """Build (and cache) all reference distributions used by the scorers."""
    if "refs" in _REF_CACHE:
        return _REF_CACHE["refs"]
    wm_u = _universal()
    pos_bigram = _build_pos_bigram(wm_u, _universal_iter())
    pi_article = _flow_response_pi(_load_world("article"), wm_u, _article_iter())
    pi_dialog  = _flow_response_pi(_load_world("dialog"),  wm_u, _dialog_iter())
    refs = {
        "pos_bigram": pos_bigram,
        "pi_article": pi_article,
        "pi_dialog":  pi_dialog,
    }
    _REF_CACHE["refs"] = refs
    return refs


# ---------------------------------------------------------------------------
# Score functions
# ---------------------------------------------------------------------------

def syntax_score(words, refs=None) -> float:
    """Mean log-prob of POS bigrams in `words` under reference POS-bigram.
    `words` may be lemma strings or manifold indices."""
    if refs is None: refs = get_refs()
    wm_u = _universal()
    idx = _to_idx(words, wm_u)
    if len(idx) < 2:
        return -10.0
    pos_arg = _pos_arg(wm_u)
    seq = pos_arg[np.asarray(idx)]
    log_p = np.log(refs["pos_bigram"])
    return float(log_p[seq[:-1], seq[1:]].mean())


def semantic_score(words) -> float:
    """Mean pairwise cosine of content-word α-vectors in the dictionary
    manifold.  High = the content words point in similar α-directions =
    they 'mean compatible things'."""
    wm_dict = _load_world("dictionary")
    idx = _to_idx(words, wm_dict)
    pos_arg = _pos_arg(wm_dict)
    content = [i for i in idx if int(pos_arg[i]) in CONTENT_POS]
    if len(content) < 2:
        return 0.0
    A = wm_dict.alpha[np.asarray(content)]
    A = A / np.maximum(np.linalg.norm(A, axis=1, keepdims=True), 1e-9)
    sim = A @ A.T
    np.fill_diagonal(sim, 0.0)
    k = len(content)
    return float(sim.sum() / max(k * (k - 1), 1))


def context_score(words, route: str, refs=None) -> float:
    """Negative KL(output_pi || flow_response_pi).  route ∈ {"article","dialog"}."""
    if refs is None: refs = get_refs()
    wm_u = _universal()
    idx = _to_idx(words, wm_u)
    if not idx:
        return -10.0
    pi_out = wm_u.pi[np.asarray(idx)].mean(axis=0)
    pi_out = (pi_out + 1e-3); pi_out = pi_out / pi_out.sum()
    pi_ref = (refs["pi_article"] if route == "article" else refs["pi_dialog"]) + 1e-3
    pi_ref = pi_ref / pi_ref.sum()
    kl = float((pi_out * np.log(pi_out / pi_ref)).sum())
    return -kl


def _content_alpha_mean(text_or_words) -> np.ndarray | None:
    """Mean α-vector over content-word lemmas, normalised to unit length.
    Uses the dictionary world manifold so the α captures definitional
    semantics, not corpus surface co-occurrence."""
    wm_dict = _load_world("dictionary")
    if isinstance(text_or_words, str):
        toks = WORD_RE.findall(text_or_words.lower())
        idx = [wm_dict.idx[t] for t in toks if t in wm_dict.idx]
    else:
        idx = _to_idx(text_or_words, wm_dict)
    pos_arg = _pos_arg(wm_dict)
    content = [i for i in idx if int(pos_arg[i]) in CONTENT_POS]
    if len(content) == 0:
        return None
    A = wm_dict.alpha[np.asarray(content)]
    m = A.mean(axis=0)
    n = np.linalg.norm(m)
    return m / max(n, 1e-9)


def relevance_score(query: str, response_words) -> float:
    """Cosine of (mean content α of query) and (mean content α of
    response).  Higher = response's content words point in the same
    α-direction as the query's content words = topically related.

    Calibrated against real (Q, R) pairs (alpaca + conversations).
    Identical-text would score 1.0 (parrot); orthogonal would score 0.
    Real Q-A pairs sit in roughly 0.3–0.8.

    Returns 0.0 if either side has no in-vocab content words.
    """
    a_q = _content_alpha_mean(query)
    a_r = _content_alpha_mean(response_words)
    if a_q is None or a_r is None:
        return 0.0
    return float(a_q @ a_r)


# ---------------------------------------------------------------------------
# Combined / floor utilities
# ---------------------------------------------------------------------------

# Empirical floors — populated from real-corpus 25th percentile.
# `relevance_*` floors are populated by calibrate_relevance_floors.py
# and saved to data/processed/relevance_floors.json.
DEFAULT_FLOORS = {
    "syntax":             -1.794,
    "semantic":            0.393,
    "context":            -0.334,
    "relevance_article":   0.0,    # placeholder; calibrated separately
    "relevance_dialog":    0.0,    # placeholder; calibrated separately
}


def _load_calibrated_floors():
    """Override placeholders if the calibration file exists."""
    fp = ROOT / "data" / "processed" / "relevance_floors.json"
    if fp.exists():
        import json
        d = json.loads(fp.read_text())
        DEFAULT_FLOORS["relevance_article"] = float(d["article"])
        DEFAULT_FLOORS["relevance_dialog"]  = float(d["dialog"])


_load_calibrated_floors()


def all_scores(words, route: str, query: str | None = None,
                refs=None) -> dict:
    out = {
        "syntax":   syntax_score(words, refs=refs),
        "semantic": semantic_score(words),
        "context":  context_score(words, route, refs=refs),
    }
    if query is not None:
        out["relevance"] = relevance_score(query, words)
    return out


def relevance_floor(route: str, floors: dict | None = None) -> float:
    if floors is None: floors = DEFAULT_FLOORS
    key = "relevance_article" if route == "article" else "relevance_dialog"
    return floors[key]


def passes_all(scores: dict, route: str | None = None,
                floors: dict | None = None) -> bool:
    if floors is None: floors = DEFAULT_FLOORS
    base = all(scores[k] >= floors[k] for k in ("syntax", "semantic", "context"))
    if "relevance" in scores and route is not None:
        return base and scores["relevance"] >= relevance_floor(route, floors)
    return base


def composite(scores: dict, route: str | None = None,
                weights: dict | None = None,
                floors: dict | None = None) -> float:
    """Single scalar for ranking.  Re-centres each score on its floor."""
    if floors is None:  floors = DEFAULT_FLOORS
    if weights is None: weights = {"syntax": 1.0, "semantic": 1.0,
                                     "context": 1.0, "relevance": 1.0}
    s  = weights["syntax"]   * (scores["syntax"]   - floors["syntax"])
    s += weights["semantic"] * (scores["semantic"] - floors["semantic"])
    s += weights["context"]  * (scores["context"]  - floors["context"])
    if "relevance" in scores and route is not None:
        s += weights.get("relevance", 1.0) * (
            scores["relevance"] - relevance_floor(route, floors))
    return s


def min_clearance(scores: dict, route: str | None = None,
                    floors: dict | None = None) -> float:
    """How much the worst boundary clears its floor.  Used as a
    secondary key for ranking when no candidate passes all four."""
    if floors is None: floors = DEFAULT_FLOORS
    clears = [
        scores["syntax"]   - floors["syntax"],
        scores["semantic"] - floors["semantic"],
        scores["context"]  - floors["context"],
    ]
    if "relevance" in scores and route is not None:
        clears.append(scores["relevance"] - relevance_floor(route, floors))
    return min(clears)
