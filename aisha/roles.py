"""
roles.py — load per-word semantic-role priors and infer slot roles
from POS templates.

Two pieces:

  load_word_roles(route)            -> (roles, classes)  shape (N, K)
  infer_template_roles(template)    -> list[str] same length as template

The template-role inference is a small rule-based grammar:
  - DET    → "det"
  - ADJ    → "mod"
  - ADV    → "mod"
  - ADP    → "prep"
  - first NOUN in a noun chunk → "head"
  - the noun right after ADP   → "pobj"
  - PRON at sentence-start     → "subj"
  - PRON after VERB / after ADP → "obj" / "pobj"
  - first VERB in a clause     → "root"
  - subsequent VERB after AUX  → still "root"
  - AUX before VERB            → "aux"

The output is a list of role names matching the template length, used
by phrase_jump.py to filter slot candidates and by grammar.py to
apply role-specific rules.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


_CACHE: dict = {}


def load_word_roles(route: str) -> tuple[np.ndarray, list[str]]:
    """Return (roles, classes).  roles is (N, K) of P(role | word)."""
    key = f"roles:{route}"
    if key in _CACHE:
        return _CACHE[key]
    p = ROOT / "data" / "processed" / f"word_roles_{route}.npz"
    if not p.exists():
        raise FileNotFoundError(
            f"role priors not built for {route}: {p} — run build_word_roles.py")
    z = np.load(p, allow_pickle=True)
    roles = z["roles"].astype(np.float32)
    classes = list(z["classes"])
    _CACHE[key] = (roles, classes)
    return roles, classes


def infer_template_roles(template: tuple) -> list[str]:
    """Heuristic POS → role labels for one phrase template.

    The template is a tuple of spaCy POS tags (NOUN/VERB/...).  We
    walk left-to-right and assign roles based on small contextual
    rules.  These labels are coarse but consistent enough for the
    role-prior to bias slot filling.
    """
    out: list[str] = []
    seen_root = False
    after_adp = False     # next NOUN/PRON should be pobj
    after_verb = False    # next NOUN should be obj, next PRON should be obj
    n = len(template)
    for i, p in enumerate(template):
        if p == "DET":
            out.append("det")
        elif p == "ADJ":
            out.append("mod")
        elif p == "ADV":
            out.append("mod")
        elif p == "ADP":
            out.append("prep")
            after_adp = True
            continue
        elif p == "AUX":
            out.append("aux")
        elif p == "VERB":
            if seen_root:
                # Subsequent verbs in the same template — most likely
                # conjuncts or compound predicates.
                out.append("conj")
            else:
                out.append("root")
                seen_root = True
            after_verb = True
            continue
        elif p == "CCONJ" or p == "SCONJ":
            out.append("cc")
        elif p == "INTJ":
            out.append("other")
        elif p in ("NOUN", "PROPN"):
            if after_adp:
                out.append("pobj")
            elif after_verb:
                out.append("obj")
            elif i == 0 and not seen_root:
                out.append("subj")
            else:
                out.append("head")
        elif p == "PRON":
            if after_adp:
                out.append("pobj")
            elif after_verb:
                out.append("obj")
            elif i == 0 and not seen_root:
                out.append("subj")
            else:
                out.append("subj")
        elif p == "PART":
            out.append("other")
        else:
            out.append("other")

        # Reset transient state after labelling a noun-shaped slot.
        if p in ("NOUN", "PROPN", "PRON"):
            after_adp = False
            after_verb = False

    # Sanity: same length.
    assert len(out) == n
    return out
