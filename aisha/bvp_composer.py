"""
bvp_composer.py — boundary-value composer.

Wraps the existing pipeline:
  1. Generate N=40 candidate raw word streams via POSSelector (multi-seed).
  2. Score each candidate on the three boundaries (syntax / semantic /
     context).
  3. Pick the highest composite-scoring candidate.  Optionally also
     return the next best as alternatives.

This is the "candidate-and-reject" architecture — the manifold proposes,
the boundaries dispose.  No new content is invented; we are choosing
from the candidates the manifold already produces.

If no candidate clears all three floors, we still return the best — the
caller can inspect `passes_all` to know whether to ship or fall back.

Usage:
    from bvp_composer import BVPComposer
    c = BVPComposer()
    out = c.compose("hey how are you")
    print(out["text"])              # picked stream
    print(out["scores"])            # boundary scores of the pick
    print(out["passes_all"])        # bool
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import numpy as np

import boundaries as B               # noqa: E402
from pos_select import POSSelector  # noqa: E402
from query_router import QueryRouter  # noqa: E402

# Fallback (kept for backwards compat); the QueryRouter is preferred.
def style_to_route(style: str) -> str:
    return "article" if style == "scientific" else "dialog"


class BVPComposer:
    """N-sample, boundary-rerank composer."""

    def __init__(self,
                  n_samples: int = 40,
                  n_phrase_samples: int = 0,
                  weights: Optional[dict] = None,
                  floors:  Optional[dict] = None):
        self.sel = POSSelector()
        self.n_samples = n_samples
        self.n_phrase_samples = n_phrase_samples
        self.weights = weights
        self.floors = floors
        self.router = QueryRouter()
        # Lazy-loaded phrase generators per route.
        self._phrase_gens: dict[str, "PhraseJumpGenerator"] = {}
        # Warm the boundary refs cache once.
        B.get_refs()

    def _phrase_gen(self, route: str):
        if route not in self._phrase_gens:
            from phrase_jump import PhraseJumpGenerator
            self._phrase_gens[route] = PhraseJumpGenerator(route)
        return self._phrase_gens[route]

    def _manifold_style(self, query: str) -> str:
        """Pick one of the 5 styles purely from manifold signals.

        Uses the query's mean π in the universal manifold to decide
        register, then picks the closest of the 5 trained styles.
        Heuristic and fast — no SBERT, no neural inference.

        Mapping:
          - high pronoun fraction         → 'emotional' (very personal)
          - high noun + low pronoun       → 'scientific' (expository)
          - moderate pronoun + verbs      → 'casual' (default conversational)
          - high adv + adj                → 'civilized' (measured)
          - high interjection             → 'heated'
        """
        import re as _re
        wm = self.sel.wm
        toks = [t for t in _re.findall(r"[a-zA-Z']+", query.lower())
                  if t in wm.idx]
        if not toks:
            return "casual"
        idx = [wm.idx[t] for t in toks]
        import numpy as _np
        pi = wm.pi[_np.asarray(idx)].mean(axis=0)
        # channels: 0=Noun 1=Verb 2=Adj 3=Adv 4=Pro 5=Prep 6=Conj 7=Interj
        if pi[7] > 0.05:
            return "heated"
        if pi[4] > 0.30:
            return "emotional"
        if pi[0] > 0.40 and pi[4] < 0.10:
            return "scientific"
        if pi[2] + pi[3] > 0.20:
            return "civilized"
        return "casual"

    def _run_pipeline_for_candidates(self, query: str,
                                       style: Optional[str],
                                       seed_offset: int = 0) -> list[dict]:
        """Use POSSelector.respond with return_all=True.  Returns the
        raw candidate dicts (each with b_idx, text, pos_score, etc.).
        `seed_offset` shifts all internal seeds so calling this method
        multiple times with the same query yields DIFFERENT candidate
        pools — needed for in-process N-sample diversity without
        spawning new processes."""
        if style:
            self.sel.R.set_style(style)
        out = self.sel.respond(query, style=style,
                                  n_samples=self.n_samples,
                                  return_all=True,
                                  seed_offset=seed_offset)
        return out.get("candidates", []) or []

    def _score_candidate(self, cand: dict, route: str,
                          query: str | None = None
                          ) -> tuple[dict, float]:
        # Phrase-level candidates already carry their lemma list as `words`;
        # word-level candidates have b_idx into the universal manifold.
        if "words" in cand:
            words = cand["words"]
        else:
            words = [self.sel.wm.lemmas[i] for i in cand["b_idx"]]
        scores = B.all_scores(words, route, query=query)
        composite = B.composite(scores, route=route,
                                  weights=self.weights,
                                  floors=self.floors)
        return scores, composite

    def compose(self, query: str,
                 style: Optional[str] = None,
                 seed_offset: int = 0,
                 verbose: bool = False) -> dict:
        t0 = time.time()
        # 1. Pick style + route for this query.  Both are query-invariant
        #    across seed_offset trials, so cache by query string.
        #    Style is derived from the manifold's own π distribution — no
        #    SBERT, no neural bolt-on.  Route uses the heuristic+manifold
        #    QueryRouter (no SBERT either).
        cache = getattr(self, "_query_cache", None)
        if cache is None:
            cache = self._query_cache = {}
        if query not in cache:
            if style is None:
                style_inferred = self._manifold_style(query)
            else:
                style_inferred = style
            route_inferred = self.router.route(query)
            cache[query] = (style_inferred, route_inferred)
        cached_style, cached_route = cache[query]
        if style is None:
            style = cached_style
        route = cached_route

        # 2. Generate N candidates from word-level POSSelector.
        raw_candidates = self._run_pipeline_for_candidates(
            query, style, seed_offset=seed_offset)

        # 2b. Optionally augment with phrase-level candidates.
        if self.n_phrase_samples > 0:
            try:
                pg = self._phrase_gen(route)
                # Match the word-level Aisha length distribution loosely.
                length_choices = [6, 7, 8, 9, 10, 11, 12]
                for k in range(self.n_phrase_samples):
                    pg.rng = np.random.default_rng(
                        42 + k * 17 + abs(hash(query)) % 1000
                          + seed_offset * 9973)
                    L = length_choices[k % len(length_choices)]
                    out = pg.generate_candidate(query, length=L,
                                                  apply_grammar=True)
                    if out["b_idx"]:
                        # Phrase candidate: words are the GRAMMAR-POLISHED
                        # surface tokens (may include inflections /
                        # inserted "an"); text is the final string.
                        raw_candidates.append({
                            "b_idx": [],
                            "words": out["words"],
                            "text":  out["text"],
                            "src":   "phrase",
                        })
            except FileNotFoundError as e:
                print(f"[bvp] phrase generator unavailable: {e}",
                      flush=True)

        if not raw_candidates:
            return {"text": "", "scores": {}, "composite": -np.inf,
                    "passes_all": False, "alternatives": [],
                    "n_candidates": 0, "style": style, "route": route,
                    "elapsed": time.time() - t0}

        # 3. Score each.
        scored = []
        for c in raw_candidates:
            sc, comp = self._score_candidate(c, route, query=query)
            scored.append({
                "text": c.get("text", ""),
                "b_idx": c.get("b_idx", []),
                "src":   c.get("src", "word"),
                "scores": sc,
                "composite": comp,
                "passes_all": B.passes_all(sc, route=route, floors=self.floors),
            })

        # 4. Pick best.  Sort key:
        #     a. passes_all first (True before False)
        #     b. then prefer the candidate whose WORST-failing boundary
        #        is least bad (max of min_clearance)
        #     c. finally composite for tie-break
        for s in scored:
            s["min_clearance"] = B.min_clearance(s["scores"], route=route,
                                                    floors=self.floors)
        scored.sort(key=lambda x: (not x["passes_all"],
                                      -x["min_clearance"],
                                      -x["composite"]))
        best = scored[0]
        alts = [{"text": s["text"], "scores": s["scores"],
                  "composite": s["composite"]}
                 for s in scored[1:5]]

        if verbose:
            print(f"[bvp] query={query!r}  style={style}  route={route}", flush=True)
            print(f"[bvp] candidates={len(scored)}  pass_all={sum(s['passes_all'] for s in scored)}", flush=True)
            for s in scored[:3]:
                print(f"  {s['composite']:+.3f}  pass={s['passes_all']}  "
                      f"syn={s['scores']['syntax']:+.2f} sem={s['scores']['semantic']:+.2f} "
                      f"ctx={s['scores']['context']:+.2f}  {s['text']}",
                      flush=True)

        return {
            "text":          best["text"],
            "b_idx":         best["b_idx"],
            "scores":        best["scores"],
            "composite":     best["composite"],
            "passes_all":    best["passes_all"],
            "alternatives":  alts,
            "n_candidates":  len(scored),
            "n_passing":     sum(1 for s in scored if s["passes_all"]),
            "style":         style,
            "route":         route,
            "elapsed":       time.time() - t0,
        }


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------

def _cli():
    import argparse, json
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", "-q", action="append", default=[],
                     help="(repeatable) query string to compose for")
    ap.add_argument("--samples", type=int, default=40)
    ap.add_argument("--style", type=str, default=None,
                     choices=["casual", "civilized", "emotional",
                                "heated", "scientific"])
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    queries = args.query or [
        "hey how are you",
        "what is entropy",
        "I miss her so much",
    ]
    c = BVPComposer(n_samples=args.samples)
    for q in queries:
        out = c.compose(q, style=args.style, verbose=args.verbose)
        print(json.dumps({
            "query": q, "text": out["text"],
            "scores": out["scores"],
            "passes_all": out["passes_all"],
            "n_passing": out["n_passing"],
            "n_candidates": out["n_candidates"],
            "elapsed": out["elapsed"],
        }, indent=2, default=str))


if __name__ == "__main__":
    _cli()
