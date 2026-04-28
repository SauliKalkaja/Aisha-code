"""
octant_transition.py — build and use the 8×8 sentence-octant
transition matrix.

Sentence octant = sign-vector of mean(M_n, chi_n, spin_n) over content
words. Index = 4*sign(M) + 2*sign(chi) + sign(spin).

Matrix M[input_oct][output_oct] = empirical count of pairs (A, B)
with those octant signatures.  Built from corpus pairs once; loaded
at runtime by responder.
"""
from __future__ import annotations

import csv
import json
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
PROC = ROOT / "data" / "processed"
DATA = ROOT / "data"
OCT8_TRANSITION_PATH = PROC / "oct8_transition.npy"


def sentence_octant(text: str, R) -> int | None:
    """Compute 8-bit octant signature of a sentence using R's manifold.
    Tries content-words-only first; falls back to ALL in-vocab tokens
    for short queries that have no content tokens.  Returns None only
    if the query has zero in-vocab tokens at all."""
    from corpus_deep import tokenize, CONTENT_OCTANTS
    idx_all = tokenize(text, R.wm)
    # Pass 1: content-only
    idx = [i for i in idx_all if R.mask10k[i]
           and int(R.oct8[i]) in CONTENT_OCTANTS]
    # Pass 2: any in-vocab token (fallback for short queries)
    if not idx:
        idx = [i for i in idx_all if i < R.wm.N]
    if not idx:
        return None
    arr = np.asarray(idx)
    m_mean = float(R.M_n[arr].mean())
    chi_mean = float(R.chi_n[arr].mean())
    spin_mean = float(R.spin_n[arr].mean())
    return (4 * int(m_mean > 0) +
            2 * int(chi_mean > 0) +
                int(spin_mean > 0))


def octant_weight(word_oct: int, target_oct: int) -> float:
    """Weight multiplier for a word's pool ranking based on how many
    axes its octant matches the target octant."""
    matches = 3 - bin(word_oct ^ target_oct).count("1")
    return [0.3, 1.0, 2.0, 4.0][matches]


def _gather_pairs(max_pairs: int = 60_000):
    pairs = []
    # Conversations: consecutive turns
    convs: dict[int, list] = {}
    with (DATA / "conversations.csv").open() as r:
        for row in csv.DictReader(r):
            cid = int(row["conv_id"])
            convs.setdefault(cid, []).append(row)
    for cid, turns in convs.items():
        turns.sort(key=lambda t: int(t["turn_id"]))
        for k in range(len(turns) - 1):
            ta = (turns[k].get("text") or "").strip()
            tb = (turns[k + 1].get("text") or "").strip()
            if 5 <= len(ta) <= 280 and 5 <= len(tb) <= 280:
                pairs.append((ta, tb))
    # Alpaca Q→R
    with (DATA / "raw" / "alpaca_pairs.json").open() as f:
        blob = json.load(f)
    items = blob.get("train", blob if isinstance(blob, list) else [])
    for item in items:
        q = (item.get("q") or "").strip()
        r_ = (item.get("r") or "").strip()
        if 5 <= len(q) <= 280 and 5 <= len(r_) <= 280:
            pairs.append((q, r_))
        if len(pairs) >= max_pairs:
            break
    return pairs


def build_transition(R, out_path: Path = OCT8_TRANSITION_PATH,
                       max_pairs: int = 60_000) -> dict:
    pairs = _gather_pairs(max_pairs)
    rng = random.Random(0)
    rng.shuffle(pairs)
    pairs = pairs[:max_pairs]
    n = len(pairs)
    n_train = int(0.8 * n)
    train, test = pairs[:n_train], pairs[n_train:]

    M = np.zeros((8, 8), dtype=np.int64)
    n_train_used = 0
    n_train_skip = 0
    for a, b in train:
        oa = sentence_octant(a, R)
        ob = sentence_octant(b, R)
        if oa is None or ob is None:
            n_train_skip += 1
            continue
        M[oa, ob] += 1
        n_train_used += 1

    # Laplace smoothing so empty cells still get small probability
    M_smooth = M.astype(np.float64) + 0.5

    # Holdout validation
    n_test = 0
    n_test_skip = 0
    n_correct1 = 0
    n_correct3 = 0
    for a, b in test:
        oa = sentence_octant(a, R)
        ob = sentence_octant(b, R)
        if oa is None or ob is None:
            n_test_skip += 1
            continue
        row = M_smooth[oa]
        pred1 = int(np.argmax(row))
        top3 = set(np.argsort(-row)[:3].tolist())
        n_test += 1
        if pred1 == ob: n_correct1 += 1
        if ob in top3: n_correct3 += 1

    np.save(out_path, M_smooth)

    return {
        "pairs_total": n,
        "train_total": len(train),
        "train_used": n_train_used,
        "train_skip": n_train_skip,
        "test_total": len(test),
        "test_used": n_test,
        "top1_acc": n_correct1 / max(n_test, 1),
        "top3_acc": n_correct3 / max(n_test, 1),
        "matrix_int": M,
    }


def load_transition(path: Path = OCT8_TRANSITION_PATH) -> np.ndarray | None:
    if not path.exists():
        return None
    return np.load(path)


def predict_target_octant(input_oct: int, M: np.ndarray,
                              sample: bool = True) -> int:
    """If sample=True, draw output octant weighted by transition row.
    Adds variability — same input can produce different target octants.
    If sample=False, return argmax (deterministic)."""
    row = M[input_oct].astype(np.float64)
    if not sample:
        return int(np.argmax(row))
    s = row.sum()
    if s <= 0:
        return int(np.argmax(row))
    p = row / s
    return int(np.random.choice(8, p=p))


# ---- CLI: build matrix from scratch -------------------------------
def main():
    sys.path.insert(0, str(ROOT))
    from aisha_respond import Responder
    print("[oct8] loading Responder…")
    R = Responder()
    print("[oct8] building transition matrix…")
    info = build_transition(R)
    out_path = Path("/tmp/oct8_transition_build.txt")
    with out_path.open("w") as f:
        f.write("Transition matrix build report\n")
        f.write("=" * 50 + "\n")
        for k, v in info.items():
            if k == "matrix_int":
                f.write("matrix (raw counts):\n")
                for i in range(8):
                    row = "  ".join(f"{int(v[i,j]):>5d}" for j in range(8))
                    f.write(f"  {i}: [{row}]\n")
            elif "acc" in k:
                f.write(f"{k}: {v*100:.1f}%\n")
            else:
                f.write(f"{k}: {v}\n")
    print(f"saved {OCT8_TRANSITION_PATH}")
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
