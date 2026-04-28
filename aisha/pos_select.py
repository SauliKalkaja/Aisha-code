"""
pos_select.py — POS-guided multi-sample selection.

For each A-prompt, generate N candidate Aisha responses with different
RNG seeds (injecting the noise you suggested), score each by how well
its POS sequence matches the corpus POS bigram distribution, return
the best.

POS comes natively from wm.pi (8-simplex over
Noun/Verb/Adj/Adv/Pro/Prep/Conj/Interj), already trained into the
manifold.

Two scoring metrics:
  (1) POS-bigram log-likelihood under corpus P(POS_{t+1} | POS_t)
  (2) POS-sequence cosine to style-specific corpus POS histogram

Default: sum of both (weights tunable).

Usage:
  python pos_select.py --sentence "…" --samples 50
  python pos_select.py --demo 8 --samples 40
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from aisha_respond import Responder                         # noqa: E402
from layer2 import JumpLayer                                # noqa: E402

CHANNEL_NAMES = ("Noun", "Verb", "Adj", "Adv", "Pro", "Prep", "Conj", "Interj")
_WORD_RE = re.compile(r"[a-zA-Z']+")


def tokenize(text, wm, mask):
    return [wm.idx[t] for t in [t.lower() for t in _WORD_RE.findall(text)]
             if t in wm.idx and mask[wm.idx[t]]]


class POSSelector:
    def __init__(self):
        print("[pos-sel] loading Responder…", flush=True)
        self.R = Responder(rng_seed=0)
        self.wm = self.R.wm
        # Share state with Responder — avoids re-loading manifold.
        self.layer = JumpLayer(responder=self.R)

        self._build_corpus_pos_stats()

    # --------------------------------------------------------------
    def _build_corpus_pos_stats(self):
        """Compute POS bigram transitions and per-style POS histograms."""
        self.pos_bigram = np.zeros((8, 8), dtype=np.float64) + 0.5   # Laplace
        self.style_pos_hist: dict[str, np.ndarray] = {}
        style_counts: dict[str, np.ndarray] = {}

        with open(ROOT / "data" / "conversations.csv") as f:
            for row in csv.DictReader(f):
                idx = tokenize(row["text"], self.wm, self.R.mask10k)
                if len(idx) < 2: continue
                pos_seq = [int(self.wm.pi[t].argmax()) for t in idx]

                # bigram
                for a, b in zip(pos_seq[:-1], pos_seq[1:]):
                    self.pos_bigram[a, b] += 1

                # per-style histogram
                st = row["style"]
                if st not in style_counts:
                    style_counts[st] = np.zeros(8)
                for p in pos_seq:
                    style_counts[st][p] += 1

        # Row-normalise bigram
        self.pos_bigram = self.pos_bigram / self.pos_bigram.sum(axis=1, keepdims=True)
        # Log-probs for scoring
        self.log_bigram = np.log(self.pos_bigram)

        for st, c in style_counts.items():
            self.style_pos_hist[st] = c / max(c.sum(), 1)

        print(f"[pos-sel] corpus POS bigram: shape {self.pos_bigram.shape}")
        print(f"[pos-sel] per-style POS histograms: {list(self.style_pos_hist)}")

    # --------------------------------------------------------------
    def pos_sequence(self, idx_seq: list[int]) -> list[int]:
        return [int(self.wm.pi[t].argmax()) for t in idx_seq]

    def score_bigram(self, idx_seq: list[int]) -> float:
        """Mean POS-bigram log-likelihood per step."""
        if len(idx_seq) < 2: return -np.inf
        pos = self.pos_sequence(idx_seq)
        lp = 0.0
        for a, b in zip(pos[:-1], pos[1:]):
            lp += self.log_bigram[a, b]
        return float(lp / (len(pos) - 1))

    def score_style_hist(self, idx_seq: list[int], style: str) -> float:
        """Cosine similarity between candidate POS hist and style POS hist."""
        if style not in self.style_pos_hist: return 0.0
        pos = self.pos_sequence(idx_seq)
        h = np.bincount(pos, minlength=8).astype(float)
        h /= max(h.sum(), 1)
        s = self.style_pos_hist[style]
        return float(h @ s / (np.linalg.norm(h) * np.linalg.norm(s) + 1e-9))

    def composite_score(self, idx_seq, style,
                         w_bigram=1.0, w_hist=2.0) -> float:
        return (w_bigram * self.score_bigram(idx_seq) +
                 w_hist   * self.score_style_hist(idx_seq, style))

    # --------------------------------------------------------------
    def respond(self, a_text: str, style: str | None = None,
                 n_samples: int = 30,
                 return_all: bool = False,
                 seed_offset: int = 0) -> dict:
        candidates = []
        for seed in range(n_samples):
            self.R.rng = np.random.default_rng(
                seed * 13 + hash(a_text) % 10000 + seed_offset * 9973)
            out = self.R.respond(a_text, style=style)
            if len(out["b_idx"]) < 3: continue
            score = self.composite_score(out["b_idx"],
                                             style or out["style"])
            pos = self.pos_sequence(out["b_idx"])
            out["pos_score"] = score
            out["pos_seq"]   = pos
            out["pos_tags"]  = [CHANNEL_NAMES[p] for p in pos]
            candidates.append(out)

        if not candidates:
            return {"text": "", "pos_score": -np.inf, "candidates": []}

        candidates.sort(key=lambda c: -c["pos_score"])
        best = candidates[0]
        best["n_candidates"] = len(candidates)
        if return_all:
            best["candidates"] = candidates
        return best


# ======================================================================
def cmd_sentence(args) -> None:
    sel = POSSelector()
    out = sel.respond(args.sentence, style=args.style,
                       n_samples=args.samples, return_all=True)
    print()
    print(f"A>      {args.sentence}")
    print(f"B best> {out['text']}")
    print(f"  pos  : " + "  ".join(
        f"{w}:{t[:3]}" for w, t in zip(
            [sel.wm.lemmas[i] for i in out["b_idx"]], out["pos_tags"])))
    print(f"  score: {out['pos_score']:+.3f}   "
          f"(selected from {out['n_candidates']} candidates)")
    print()
    print("Top 5 alternatives:")
    for c in out["candidates"][1:6]:
        print(f"  {c['pos_score']:+.3f}   {c['text']}")


def cmd_demo(args) -> None:
    sel = POSSelector()
    rng = np.random.default_rng(args.seed)
    by_style: dict = {"casual":[], "civilized":[], "emotional":[],
                       "heated":[], "scientific":[]}
    with open(ROOT / "data" / "conversations.csv") as f:
        for row in csv.DictReader(f):
            if row["speaker"] == "A" and row["style"] in by_style:
                by_style[row["style"]].append(row["text"])

    picks = []
    per_style = max(1, args.demo // len(by_style))
    for st, texts in by_style.items():
        if not texts: continue
        k = min(per_style, len(texts))
        ids = rng.choice(len(texts), size=k, replace=False)
        for i in ids: picks.append((st, texts[i]))

    print()
    print("=" * 100)
    print(f"POS-select demo  —  {len(picks)} prompts × {args.samples} samples each")
    print("=" * 100)
    for style, a_text in picks:
        out = sel.respond(a_text, style=style,
                           n_samples=args.samples, return_all=True)
        if not out["candidates"]: continue
        print()
        print(f"[{style}]")
        print(f"  A>        {a_text}")
        words = [sel.wm.lemmas[i] for i in out["b_idx"]]
        pos = out["pos_tags"]
        print(f"  Aisha>    {out['text']}")
        print(f"    words:  {words}")
        print(f"    POS  :  {[p[:3] for p in pos]}")
        print(f"    score:  {out['pos_score']:+.3f}   "
              f"n_valid={out['n_candidates']}/{args.samples}")
        print(f"  Alt1 ({out['candidates'][1]['pos_score']:+.3f}):  "
              f"{out['candidates'][1]['text']}")
        print(f"  Alt2 ({out['candidates'][2]['pos_score']:+.3f}):  "
              f"{out['candidates'][2]['text']}")
        # Worst alternative
        worst = out['candidates'][-1]
        print(f"  Worst ({worst['pos_score']:+.3f}): {worst['text']}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sentence", type=str, default=None)
    ap.add_argument("--style", type=str, default=None,
                     choices=["casual","civilized","emotional","heated","scientific"])
    ap.add_argument("--samples", type=int, default=30)
    ap.add_argument("--demo", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.sentence:
        cmd_sentence(args)
    else:
        cmd_demo(args)


if __name__ == "__main__":
    main()
