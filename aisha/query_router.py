"""
query_router.py — wraps the trained world-manifold router with a small
interrogative-pattern heuristic for short queries.

Why the heuristic: queries like "what is entropy" have only 3 words and
not enough manifold signal for a world-feature classifier (~84% holdout,
but missed all the very-short factual ones in smoke test).  A simple
regex on the leading lexical pattern catches them cleanly without
inventing content.

Public API:
    QueryRouter().route(query) -> "article" | "dialog"
    QueryRouter().route_with_proba(query) -> (label, p_article, source)
                                              source ∈ {"heuristic", "model"}

We're pinning to a binary article/dialog routing; the caller maps that
to the appropriate context-flow boundary in `boundaries.py`.
"""
from __future__ import annotations

import pickle
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from word_manifold import WordManifold

# Patterns: when the query's leading shape matches, treat as article-route.
# Conservative: only catch clear factual-question shapes.  Anything else
# falls through to the model.
_ARTICLE_PATTERNS = [
    re.compile(r"^\s*what\s+(is|are|was|were|do|does|did|causes|caused)\b", re.I),
    re.compile(r"^\s*how\s+(does|do|did|is|are|was|were|can|could)\b", re.I),
    re.compile(r"^\s*why\s+(is|are|was|were|do|does|did)\b", re.I),
    re.compile(r"^\s*when\s+(is|was|did|does|do)\b", re.I),
    re.compile(r"^\s*where\s+(is|was|are|did|does)\b", re.I),
    re.compile(r"^\s*which\s+", re.I),
    re.compile(r"^\s*(explain|describe|define|list|name|tell me about)\b", re.I),
    re.compile(r"^\s*(in physics|in mathematics|in biology|in chemistry|in history)\b", re.I),
]

# Patterns: when these match, treat as dialog regardless of model.
# Catches clear casual / personal / emotive openings.
_DIALOG_PATTERNS = [
    re.compile(r"^\s*(hey|hi|hello|yo|sup|hiya)\b", re.I),
    re.compile(r"^\s*i\s+(am|'m|miss|love|hate|need|want|feel|cannot|"
                  r"can't|don't|do not|got|just|have|had|lost|hope)\b", re.I),
    re.compile(r"^\s*(thanks|thank you|please)\b", re.I),
    re.compile(r"^\s*(leave me|stop|I am angry|this is unfair|"
                  r"do not lecture|you are wrong|I disagree|I have had enough)",
                  re.I),
    re.compile(r"^\s*(could we|would you|may i)\b", re.I),
    # Personal-possessive overrides any "how was/is X" article pattern.
    re.compile(r"\b(your|my)\s+(day|night|life|family|friend|trip|time)\b", re.I),
    # "tell me a joke" / "tell me about your X" style chat.
    re.compile(r"^\s*tell me a\b", re.I),
]

WORD_RE = re.compile(r"[a-zA-Z']+")


class QueryRouter:
    def __init__(self, router_pkl: Path | None = None):
        self.proc = ROOT / "data" / "processed"
        rp = router_pkl or (self.proc / "query_router.pkl")
        with rp.open("rb") as f:
            obj = pickle.load(f)
        self.clf = obj["clf"]
        self.classes = list(obj["classes"])
        # Lazy-load world manifolds when first needed.
        self._wms = None

    def _load_wms(self):
        if self._wms is None:
            self._wms = [
                WordManifold.load(str(self.proc / "manifold_article.pkl")),
                WordManifold.load(str(self.proc / "manifold_dialog.pkl")),
                WordManifold.load(str(self.proc / "manifold_dictionary.pkl")),
            ]
        return self._wms

    @staticmethod
    def _features(text: str, wms) -> np.ndarray:
        toks = WORD_RE.findall(text.lower())
        parts = []
        for wm in wms:
            idx = [wm.idx[t] for t in toks if t in wm.idx]
            if idx:
                a = np.asarray(idx)
                parts.append(wm.pi[a].mean(axis=0))
                parts.append(wm.alpha[a].mean(axis=0))
                parts.append(wm.omega[a].mean(axis=0))
            else:
                parts += [np.zeros(8)] * 3
        return np.concatenate(parts)

    @staticmethod
    def _heuristic(query: str) -> str | None:
        # Dialog patterns first — they're more specific (personal-possessive
        # overrides generic interrogative shapes).
        for pat in _DIALOG_PATTERNS:
            if pat.search(query):
                return "dialog"
        for pat in _ARTICLE_PATTERNS:
            if pat.search(query):
                return "article"
        return None

    def route(self, query: str) -> str:
        return self.route_with_proba(query)[0]

    def route_with_proba(self, query: str) -> tuple[str, float, str]:
        h = self._heuristic(query)
        if h is not None:
            return h, (1.0 if h == "article" else 0.0), "heuristic"
        wms = self._load_wms()
        v = self._features(query, wms).reshape(1, -1)
        p = self.clf.predict_proba(v)[0]
        idx_a = self.classes.index("article")
        p_art = float(p[idx_a])
        label = "article" if p_art >= 0.5 else "dialog"
        return label, p_art, "model"


def _cli():
    """Smoke test."""
    r = QueryRouter()
    cases = [
        ("article", "what is entropy"),
        ("article", "how does photosynthesis work"),
        ("article", "why is the sky blue"),
        ("article", "explain quantum tunneling"),
        ("article", "what is the capital of France"),
        ("article", "describe how a transformer engine works"),
        ("article", "what causes earthquakes"),
        ("article", "what is dark matter"),
        ("article", "how do neurons fire"),
        ("article", "what is the second law of thermodynamics"),
        ("dialog",  "hey how are you"),
        ("dialog",  "what's up"),
        ("dialog",  "are you bored"),
        ("dialog",  "tell me a joke"),
        ("dialog",  "how was your day"),
        ("dialog",  "I miss her so much"),
        ("dialog",  "I am feeling tired today"),
        ("dialog",  "I am scared about tomorrow"),
        ("dialog",  "leave me alone"),
        ("dialog",  "this is unfair"),
        ("dialog",  "would you kindly explain this"),
        ("dialog",  "I respectfully disagree"),
    ]
    n_ok = 0
    for expected, q in cases:
        label, p_art, src = r.route_with_proba(q)
        ok = "OK " if label == expected else "MISS"
        if ok == "OK ": n_ok += 1
        print(f"  {ok}  via {src:<9}  p_art={p_art:.2f}  {q!r}  -> {label}")
    print(f"\nrouter accuracy on smoke set: {n_ok}/{len(cases)}")


if __name__ == "__main__":
    _cli()
