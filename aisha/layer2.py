"""
layer2.py — Jump-boundary phrase grouping.

Layer 2 consumes a word stream from layer 1 (the quantum manifold) and
splits it into phrase-level groups by detecting analytical jumps in the
physical coordinates of consecutive words.

Foundation: we showed (layer2_jumps.py) that real corpus sentences split
into 2-3 groups of 3-4 words each when we threshold jumps at σ=1.0 —
matching typical English phrase counts and sizes, with no supervision.

Four independent jump measures:
  Δε    — Kepler energy     |ε_{k+1} − ε_k|
  Δe    — torsion ecc       |e_{k+1} − e_k|
  Δ5D   — Euclidean step in (M, χ, s, v, a) space
  Δpair — pair-torsion drop 1 − Re⟨q̂_k | q̂_{k+1}⟩

Δ5D and Δe agree strongly (Jaccard 0.53 on boundary sets).  Δpair is
mostly orthogonal (Jaccard 0.03-0.07).  The default grouping uses a
per-sentence z-scored sum of all four measures — ensemble detection.

Core API:
  layer = JumpLayer()
  groups = layer.group(idx_seq)                 # list of word-index groups
  detail = layer.diagnose(idx_seq)              # rich breakdown with jumps
  phrase = layer.describe(group_idx)            # per-group stats

  render = layer.render(groups)                 # "phrase one | phrase two"
"""
from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from word_manifold import WordManifold                    # noqa: E402

_ALLOWED = re.compile(r"^[a-z]{2,12}(?:[-'][a-z]{1,8})?$")
OCTANT_SHORT = {0:"cV",1:"cog",2:"rel",3:"coll",4:"abs",5:"dei",6:"stat",7:"cat"}
CONTENT_OCTANTS = frozenset({0, 1, 2, 3, 6, 7})
JUMP_MEASURES = ("eps", "ecc", "d5", "pair")


@dataclass
class Phrase:
    """A phrase group emitted by layer 2."""
    idx: list[int]                        # vocab indices
    words: list[str]                       # surface forms
    mean_coord: dict                       # per-axis mean (M, χ, s, v, a, ε, e)
    octant_hist: np.ndarray                # shape (8,), normalised
    dominant_octant: int                   # argmax
    coherence: float                       # mean pair-torsion within group
    size: int = field(init=False)

    def __post_init__(self):
        self.size = len(self.idx)

    def __repr__(self) -> str:
        text = " ".join(self.words)
        return (f"Phrase({text!r}, oct={OCTANT_SHORT[self.dominant_octant]}, "
                f"coh={self.coherence:+.2f})")


class JumpLayer:
    def __init__(self, top_k: int = 10000, responder=None):
        """If `responder` is given, share already-loaded state with it —
        avoids re-loading manifold.pkl and re-computing chi (saves ~3s)."""
        if responder is not None:
            self.wm   = responder.wm
            self.mask = responder.mask10k
            self.q_u  = responder.q_u
            self.M    = responder.M_n
            self.chi  = responder.chi_n
            self.s    = responder.spin_n
            self.v    = responder.v
            self.a    = responder.a
            # eps / ecc — recompute from Responder's alpha + beta if present,
            # else derive from normalised axes
            self.oct8 = responder.oct8
            # Need eps, ecc for layer-2 jumps — compute from M/chi/s
            self.ecc = np.sqrt(self.M ** 2 + self.chi ** 2 + self.s ** 2)
            # eps needs alpha + beta; just use M as proxy if unavailable
            mdat = np.load(ROOT / "m_fixed.npz")
            beta = mdat["beta"]
            alpha = np.load(ROOT / "alpha_fixed.npz")["alpha"]
            mu_mean = self.wm.mu.mean()
            alpha_mean = alpha.mean(axis=1)
            self.eps = 0.5 * beta * beta - mu_mean / np.maximum(alpha_mean, 1e-30)
            return

        self.wm = WordManifold.load(str(ROOT / "data" / "processed" / "manifold.pkl"))

        alpha = np.load(ROOT / "alpha_fixed.npz")["alpha"]
        mdat = np.load(ROOT / "m_fixed.npz")
        M = mdat["M"]; spin = mdat["spin"]; H_til = mdat["H_til"]; beta = mdat["beta"]

        passers = np.array([bool(_ALLOWED.fullmatch(l)) for l in self.wm.lemmas])
        idx_k = np.where(passers)[0][np.argsort(-self.wm.m[passers])][:top_k]
        self.mask = np.zeros(self.wm.N, dtype=bool); self.mask[idx_k] = True

        phase = (spin * beta)[:, None] * H_til
        q = alpha.astype(np.complex128) * np.exp(1j * phase)
        q_u = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-30)
        q_common = q_u[self.mask]
        chi = (q_u.conj() @ q_common.T).real.mean(axis=1)

        v_raw = np.load(ROOT / "valence.npz")["v"]
        a_raw = np.load(ROOT / "arousal.npz")["a"]

        def cn(x):
            c = x - np.median(x[self.mask])
            return c / max(np.std(c[self.mask]), 1e-9)

        self.M   = cn(M)
        self.chi = cn(chi)
        self.s   = cn(spin)
        self.v   = cn(v_raw)
        self.a   = cn(a_raw)

        mu_mean = self.wm.mu.mean()
        alpha_mean = alpha.mean(axis=1)
        self.eps = 0.5 * beta * beta - mu_mean / np.maximum(alpha_mean, 1e-30)
        self.ecc = np.sqrt(self.M ** 2 + self.chi ** 2 + self.s ** 2)

        self.q_u = q_u

        self.oct8 = (4 * (self.M   > 0).astype(int) +
                      2 * (self.chi > 0).astype(int) +
                          (self.s   > 0).astype(int))

    # ---------------------------------------------------------
    def jumps(self, idx_seq: list[int]) -> dict:
        """Return four jump arrays (length n-1) for the input sequence."""
        idx = np.asarray(idx_seq)
        n = len(idx)
        if n < 2:
            return {k: np.zeros(0) for k in JUMP_MEASURES}

        d_eps = np.abs(np.diff(self.eps[idx]))
        d_ecc = np.abs(np.diff(self.ecc[idx]))

        V = np.stack([self.M[idx], self.chi[idx], self.s[idx],
                       self.v[idx], self.a[idx]], axis=1)
        d_5d = np.linalg.norm(np.diff(V, axis=0), axis=1)

        q = self.q_u[idx]
        cos_sim = (q[:-1].conj() * q[1:]).real.sum(axis=1)
        d_pair = 1.0 - cos_sim

        return {"eps": d_eps, "ecc": d_ecc, "d5": d_5d, "pair": d_pair}

    # ---------------------------------------------------------
    def composite(self, idx_seq: list[int]) -> np.ndarray:
        """Per-sentence z-scored sum of all 4 jump measures."""
        j = self.jumps(idx_seq)
        if len(j["eps"]) == 0: return np.zeros(0)
        comp = np.zeros(len(j["eps"]))
        for m in JUMP_MEASURES:
            arr = j[m]
            sd = arr.std()
            if sd < 1e-9: continue
            comp += (arr - arr.mean()) / sd
        return comp

    # ---------------------------------------------------------
    def boundaries(self, idx_seq: list[int],
                    threshold_sigma: float = 1.0,
                    measure: str = "composite") -> list[int]:
        """Return boundary word-indices (positions into idx_seq) where
        a new phrase begins.  A boundary at position p means idx_seq[p]
        is the first word of the next phrase."""
        if measure == "composite":
            scores = self.composite(idx_seq)
        elif measure in JUMP_MEASURES:
            scores = self.jumps(idx_seq)[measure]
        else:
            raise ValueError(f"unknown measure: {measure}")

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

    # ---------------------------------------------------------
    def group(self, idx_seq: list[int], **kwargs) -> list[list[int]]:
        """Split idx_seq into phrase groups using detected boundaries."""
        bounds = self.boundaries(idx_seq, **kwargs)
        out = []
        prev = 0
        for b in bounds:
            if b > prev: out.append(list(idx_seq[prev:b]))
            prev = b
        out.append(list(idx_seq[prev:]))
        return [g for g in out if g]

    # ---------------------------------------------------------
    def describe(self, group_idx: list[int]) -> Phrase:
        """Compute per-phrase summary stats."""
        idx = np.asarray(group_idx)
        mean_coord = {
            "M":   float(self.M[idx].mean()),
            "chi": float(self.chi[idx].mean()),
            "s":   float(self.s[idx].mean()),
            "v":   float(self.v[idx].mean()),
            "a":   float(self.a[idx].mean()),
            "eps": float(self.eps[idx].mean()),
            "ecc": float(self.ecc[idx].mean()),
        }
        hist = np.bincount(self.oct8[idx], minlength=8).astype(float)
        hist /= max(hist.sum(), 1)
        dominant = int(np.argmax(hist))

        if len(idx) >= 2:
            q = self.q_u[idx]
            # mean pair-alignment across consecutive words
            cs = (q[:-1].conj() * q[1:]).real.sum(axis=1)
            coherence = float(cs.mean())
        else:
            coherence = 1.0

        return Phrase(
            idx=list(group_idx),
            words=[self.wm.lemmas[i] for i in group_idx],
            mean_coord=mean_coord,
            octant_hist=hist,
            dominant_octant=dominant,
            coherence=coherence,
        )

    # ---------------------------------------------------------
    def diagnose(self, idx_seq: list[int],
                  threshold_sigma: float = 1.0) -> dict:
        """Full breakdown — words, jumps per measure, composite scores,
        boundaries, groups, per-phrase summary."""
        words = [self.wm.lemmas[i] for i in idx_seq]
        jumps_all = self.jumps(idx_seq)
        composite_scores = self.composite(idx_seq)
        bounds = self.boundaries(idx_seq, threshold_sigma=threshold_sigma,
                                    measure="composite")
        groups_idx = self.group(idx_seq, threshold_sigma=threshold_sigma,
                                   measure="composite")
        phrases = [self.describe(g) for g in groups_idx]
        return {
            "words":     words,
            "idx":       list(idx_seq),
            "jumps":     jumps_all,
            "composite": composite_scores,
            "boundaries": bounds,
            "groups":     groups_idx,
            "phrases":    phrases,
        }

    # ---------------------------------------------------------
    def render(self, groups: list[list[int]]) -> str:
        """Render a list-of-index-groups as 'phrase1 | phrase2 | …'"""
        parts = []
        for g in groups:
            parts.append(" ".join(self.wm.lemmas[i] for i in g))
        return " | ".join(parts)


# ======================================================================
# CLI smoke-test
# ======================================================================
def _tokenize(text: str, wm, mask) -> list[int]:
    word_re = re.compile(r"[a-zA-Z']+")
    return [wm.idx[t] for t in [tok.lower() for tok in word_re.findall(text)]
             if t in wm.idx and mask[wm.idx[t]]]


def _cli() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--sentence", type=str, default=None)
    ap.add_argument("--sigma", type=float, default=1.0)
    ap.add_argument("--demo", action="store_true",
                     help="run a few built-in examples")
    args = ap.parse_args()

    print("[layer2] loading…", flush=True)
    layer = JumpLayer()

    if args.sentence:
        sentences = [args.sentence]
    elif args.demo or True:
        sentences = [
            "I think we should talk about this quietly.",
            "The damaged cells release cytokines that activate further immune responses.",
            "You're crying. I can hear you from my room.",
            "I just don't understand how anyone supports these new tax hikes.",
            "Thanks for coming in. What's on your mind today?",
        ]

    for s in sentences:
        idx = _tokenize(s, layer.wm, layer.mask)
        if len(idx) < 3:
            print(f"\n  (skipped — too few known tokens) {s}")
            continue
        d = layer.diagnose(idx, threshold_sigma=args.sigma)
        print(f"\n  source:  {s}")
        print(f"  grouped: {layer.render(d['groups'])}")
        print(f"  phrases ({len(d['phrases'])}):")
        for p in d["phrases"]:
            c = p.mean_coord
            print(f"    {p.words}  "
                  f"oct={OCTANT_SHORT[p.dominant_octant]}  "
                  f"v={c['v']:+.2f}  a={c['a']:+.2f}  "
                  f"coh={p.coherence:+.2f}")


if __name__ == "__main__":
    _cli()
