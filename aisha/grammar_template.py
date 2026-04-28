"""
grammar_template.py — strict POS-sequence templates from a small
English CFG.  Replaces the octant-based template generator for
slot-filling so that EVERY output has a valid POS structure.

Each output is a sequence of POS channels (0-7) matching the
manifold's POS bins (SPACY_TO_CH in rebuild_clean.py):
  0  NOUN
  1  VERB / AUX
  2  ADJ
  3  ADV
  4  PRON / DET   (combined — both attach to NOUN)
  5  PREP / ADP
  6  CONJ
  7  INTJ / PART

CFG rules:
  S    → NP VP
  NP   → DET (ADJ){0..2} NOUN (PP)?      |  PRON
  VP   → (AUX)? (ADV)? VERB (NP)? (PP)?
  PP   → PREP NP

Rules out word-salad like `the the`, `VERB VERB` without aux,
`DET VERB`, `PREP VERB` by construction.  Slot-filling never picks
words at illegal POS positions.

Length-flexible: takes target length, samples until close, pads with
PP if short, truncates if long.  Last token is forced to be a content
end (not DET/PREP/CONJ) so the period attaches cleanly.
"""
from __future__ import annotations

import random

# POS channel constants — must match SPACY_TO_CH in rebuild_clean.py
NOUN     = 0
VERB     = 1     # main verb or aux/modal
ADJ      = 2
ADV      = 3
PRON_DET = 4     # pronouns AND determiners share this channel
PREP     = 5
CONJ     = 6
INTJ     = 7

CONTENT_POS  = {NOUN, VERB, ADJ, ADV}
FUNCTION_POS = {PRON_DET, PREP, CONJ, INTJ}
END_OK       = {NOUN, ADJ, ADV, VERB, PRON_DET}   # ok to end a sentence
END_BAD      = {PREP, CONJ}                        # never end on these


def _rand(rng):
    """Float in [0,1) — works with either Python random or numpy Generator."""
    if hasattr(rng, "random"):
        r = rng.random()
        # numpy returns ndarray-like 0-d; coerce
        return float(r) if not isinstance(r, float) else r
    return rng.uniform(0.0, 1.0)


def _weighted_pick(rng, choices, weights):
    """Weighted choice — numpy- and stdlib-random-compatible."""
    if hasattr(rng, "choices"):                     # stdlib random
        return rng.choices(choices, weights=weights, k=1)[0]
    # numpy Generator
    import numpy as np
    p = np.asarray(weights, dtype=np.float64)
    p = p / p.sum()
    idx = rng.choice(len(choices), p=p)
    return choices[int(idx)]


def _expand_NP(rng):
    # 40% bare pronoun (was 15%) — more variety in sentence subjects
    if _rand(rng) < 0.40:
        return [PRON_DET]
    out = [PRON_DET]                                # DET
    n_adj = _weighted_pick(rng, [0, 1, 2], [0.65, 0.30, 0.05])
    out += [ADJ] * n_adj
    out.append(NOUN)
    if _rand(rng) < 0.20:                           # NP-internal PP
        out += _expand_PP(rng)
    return out


def _expand_VP(rng):
    out: list[int] = []
    if _rand(rng) < 0.30:                           # AUX
        out.append(VERB)
    if _rand(rng) < 0.15:                           # pre-verbal ADV
        out.append(ADV)
    out.append(VERB)                                # main VERB
    if _rand(rng) < 0.60:                           # direct object
        out += _expand_NP(rng)
    if _rand(rng) < 0.25:                           # PP modifier
        out += _expand_PP(rng)
    return out


def _expand_PP(rng):
    return [PREP] + _expand_NP(rng)


def build_grammar_template(length: int, rng=None) -> list[int]:
    """Generate a POS-channel sequence of `length` matching English
    declarative grammar.  Length is enforced; minimum 3."""
    rng = rng or random
    target = max(3, int(length))

    # Try up to 50 random samples; accept best-fit
    best = None
    best_diff = 1_000_000
    for _ in range(50):
        out = _expand_NP(rng) + _expand_VP(rng)
        diff = abs(len(out) - target)
        if diff < best_diff:
            best, best_diff = list(out), diff
        if diff == 0:
            break

    out = best or [PRON_DET, NOUN, VERB]

    # Pad with PPs if short
    while len(out) < target:
        out += _expand_PP(rng)
    out = out[:target]

    # Force valid sentence-end
    while out and out[-1] in END_BAD:
        out.pop()
    if not out:
        out = [PRON_DET, NOUN, VERB]
    if len(out) < target:
        while len(out) < target:
            out.append(NOUN)

    return out


if __name__ == "__main__":
    rng = random.Random(0)
    POS_NAME = ("NOUN", "VERB", "ADJ", "ADV", "PRON/DET",
                 "PREP", "CONJ", "INTJ")
    print("sample templates:")
    for L in (4, 5, 6, 7, 8, 10, 12):
        for _ in range(3):
            t = build_grammar_template(L, rng)
            label = " ".join(POS_NAME[c] for c in t)
            print(f"  L={L}:  [{label}]")
