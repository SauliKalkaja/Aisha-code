"""
layer3.py — analytical jumps at PHRASE granularity.

Same mechanism as layer 2 (jump detection + grouping), but the units
are Phrase objects instead of words.  Output: Clause objects, each
containing one or more Phrases, separated by signature jumps.

Phrase signature (7-dim): (M, χ, s, v, a) means + length + coherence.

Jump measures between consecutive phrases:
  - Δsig5    Euclidean distance in (M, χ, s, v, a) means
  - Δoct     Jensen-Shannon divergence of octant histograms
  - Δlen     |Δlog(length)|
  - Δpair    pair-torsion drop between phrase-mean q-vectors

Composite jump = per-sentence z-scored sum.  Boundary at local maxima
above σ threshold.

The hierarchy:
    word    → layer 2 → phrase
    phrase  → layer 3 → clause
    clause  → layer 4 (future) → sentence / discourse

Core API:
    layer3 = ClauseLayer(layer2)
    clauses = layer3.clause(list_of_phrases)
    detail  = layer3.diagnose(list_of_phrases)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from layer2 import JumpLayer, Phrase, OCTANT_SHORT                  # noqa: E402


@dataclass
class Clause:
    """A clause emitted by layer 3 — a group of bound phrases."""
    phrases: list[Phrase]
    mean_coord: dict                       # axis-wise mean across phrases
    octant_hist: np.ndarray                # combined 8-octant hist
    coherence: float                       # mean inter-phrase pair-torsion
    total_words: int = field(init=False)

    def __post_init__(self):
        self.total_words = sum(p.size for p in self.phrases)

    def __repr__(self) -> str:
        inner = "  |  ".join(" ".join(p.words) for p in self.phrases)
        return f"Clause[{inner}]"


def _jensen_shannon(p: np.ndarray, q: np.ndarray) -> float:
    p = p + 1e-9; q = q + 1e-9
    p /= p.sum(); q /= q.sum()
    m = 0.5 * (p + q)
    return 0.5 * float(np.sum(p * np.log(p / m)) +
                        np.sum(q * np.log(q / m)))


class ClauseLayer:
    """Layer 3 — groups layer-2 phrases into clauses via signature jumps."""

    def __init__(self, layer2: JumpLayer | None = None):
        self.layer2 = layer2 if layer2 is not None else JumpLayer()

    # --------------------------------------------------------------
    def phrase_signature(self, phrase: Phrase) -> dict:
        c = phrase.mean_coord
        return {
            "M":   c["M"], "chi": c["chi"], "s": c["s"],
            "v":   c["v"], "a":   c["a"],
            "ecc": c["ecc"],
            "length":    float(phrase.size),
            "coherence": phrase.coherence,
            "oct_hist":  phrase.octant_hist,
        }

    def phrase_mean_q(self, phrase: Phrase) -> np.ndarray:
        idx = np.asarray(phrase.idx)
        q = self.layer2.q_u[idx].mean(axis=0)
        n = np.linalg.norm(q)
        return q / (n + 1e-30)

    # --------------------------------------------------------------
    def jumps(self, phrases: list[Phrase]) -> dict:
        """Return per-measure arrays of length len(phrases)-1."""
        n = len(phrases)
        if n < 2:
            return {k: np.zeros(0) for k in ("sig5","oct","len","pair")}

        sigs = [self.phrase_signature(p) for p in phrases]
        qs = np.stack([self.phrase_mean_q(p) for p in phrases])

        sig5 = np.zeros(n - 1); oct_j = np.zeros(n - 1)
        len_j = np.zeros(n - 1); pair_j = np.zeros(n - 1)

        for k in range(n - 1):
            s1 = sigs[k]; s2 = sigs[k + 1]
            v1 = np.array([s1["M"], s1["chi"], s1["s"], s1["v"], s1["a"]])
            v2 = np.array([s2["M"], s2["chi"], s2["s"], s2["v"], s2["a"]])
            sig5[k] = float(np.linalg.norm(v2 - v1))
            oct_j[k] = _jensen_shannon(s1["oct_hist"], s2["oct_hist"])
            len_j[k] = abs(np.log(s2["length"] + 1) - np.log(s1["length"] + 1))
            # phrase-to-phrase pair-torsion drop
            cos = (qs[k].conj() * qs[k + 1]).real.sum()
            pair_j[k] = 1.0 - float(cos)

        return {"sig5": sig5, "oct": oct_j, "len": len_j, "pair": pair_j}

    def composite(self, phrases: list[Phrase]) -> np.ndarray:
        """Z-scored sum of all 4 jump measures."""
        j = self.jumps(phrases)
        if len(j["sig5"]) == 0: return np.zeros(0)
        comp = np.zeros(len(j["sig5"]))
        for m in ("sig5","oct","len","pair"):
            arr = j[m]
            sd = arr.std()
            if sd < 1e-9: continue
            comp += (arr - arr.mean()) / sd
        return comp

    # --------------------------------------------------------------
    def boundaries(self, phrases: list[Phrase],
                    threshold_sigma: float = 1.0) -> list[int]:
        scores = self.composite(phrases)
        if len(scores) < 2: return []
        mu = float(scores.mean()); sd = float(scores.std())
        if sd < 1e-9: return []
        thr = mu + threshold_sigma * sd
        bounds = []
        for k in range(len(scores)):
            if scores[k] < thr: continue
            left_ok  = (k == 0) or (scores[k] >= scores[k - 1])
            right_ok = (k == len(scores) - 1) or (scores[k] >= scores[k + 1])
            if left_ok and right_ok:
                bounds.append(k + 1)
        return bounds

    # --------------------------------------------------------------
    def clause(self, phrases: list[Phrase],
                threshold_sigma: float = 1.0) -> list[Clause]:
        if not phrases: return []
        bounds = self.boundaries(phrases, threshold_sigma=threshold_sigma)
        groups = []
        prev = 0
        for b in bounds:
            if b > prev: groups.append(phrases[prev:b])
            prev = b
        groups.append(phrases[prev:])
        groups = [g for g in groups if g]

        clauses = []
        for g in groups:
            # Aggregate signature
            mean_coord = {k: float(np.mean([p.mean_coord[k] for p in g]))
                           for k in g[0].mean_coord}
            hist = np.sum([p.octant_hist * p.size for p in g], axis=0)
            hist = hist / max(hist.sum(), 1e-9)
            # Inter-phrase pair-torsion coherence
            if len(g) >= 2:
                qs = np.stack([self.phrase_mean_q(p) for p in g])
                cs = (qs[:-1].conj() * qs[1:]).real.sum(axis=1)
                coh = float(cs.mean())
            else:
                coh = 1.0
            clauses.append(Clause(phrases=g, mean_coord=mean_coord,
                                    octant_hist=hist, coherence=coh))
        return clauses

    # --------------------------------------------------------------
    def diagnose(self, phrases: list[Phrase],
                  threshold_sigma: float = 1.0) -> dict:
        return {
            "phrases":   phrases,
            "jumps":     self.jumps(phrases),
            "composite": self.composite(phrases),
            "boundaries": self.boundaries(phrases, threshold_sigma),
            "clauses":   self.clause(phrases, threshold_sigma),
        }

    # --------------------------------------------------------------
    def render(self, clauses: list[Clause]) -> str:
        """'phrase | phrase  ||  phrase | phrase' — | inside clause, || between."""
        return "  ||  ".join(
            " | ".join(" ".join(p.words) for p in c.phrases)
            for c in clauses)


# ======================================================================
def _cli() -> None:
    import argparse, re, csv
    ap = argparse.ArgumentParser()
    ap.add_argument("--sentence", type=str, default=None)
    ap.add_argument("--sigma2", type=float, default=1.0,
                     help="layer 2 σ threshold")
    ap.add_argument("--sigma3", type=float, default=1.0,
                     help="layer 3 σ threshold")
    ap.add_argument("--demo", action="store_true",
                     help="run on a few corpus sentences")
    args = ap.parse_args()

    print("[layer3] loading…", flush=True)
    layer2 = JumpLayer()
    layer3 = ClauseLayer(layer2)

    def tokenize(text):
        wr = re.compile(r"[a-zA-Z']+")
        return [layer2.wm.idx[t] for t in [tok.lower() for tok in wr.findall(text)]
                 if t in layer2.wm.idx and layer2.mask[layer2.wm.idx[t]]]

    if args.sentence:
        sentences = [args.sentence]
    elif args.demo:
        sentences = [
            "I think we should talk about this quietly and calmly.",
            "The damaged cells release cytokines which then activate further immune responses in the tissue.",
            "You're crying and I can hear you from my room at night.",
            "I don't understand how anyone supports these new tax hikes because they're crippling small businesses.",
            "Thanks for coming in today.  What's on your mind and how can I help?",
        ]
    else:
        sentences = ["I think we should talk about this quietly."]

    for s in sentences:
        idx = tokenize(s)
        if len(idx) < 3: continue
        groups = layer2.group(idx, threshold_sigma=args.sigma2)
        phrases = [layer2.describe(g) for g in groups]
        clauses = layer3.clause(phrases, threshold_sigma=args.sigma3)
        print()
        print(f"  source:    {s}")
        print(f"  phrases ({len(phrases)}):  "
              f"{layer2.render(groups)}")
        print(f"  clauses ({len(clauses)}): "
              f"{layer3.render(clauses)}")
        for i, c in enumerate(clauses):
            m = c.mean_coord
            print(f"    [clause {i+1}]  n_phrases={len(c.phrases)}  "
                  f"words={c.total_words}  "
                  f"v={m['v']:+.2f}  a={m['a']:+.2f}  "
                  f"coh={c.coherence:+.2f}  "
                  f"dom_oct={OCTANT_SHORT[int(c.octant_hist.argmax())]}")


if __name__ == "__main__":
    _cli()
