"""
phrase_jump.py — phrase-level analytical-jump generator.

Generates a candidate response by:
  1. Picking K phrase TEMPLATES (POS sequences) sampled from the
     route-appropriate distribution (article or dialog).
  2. For each template slot, sampling a WORD from the manifold whose
     dominant POS matches the slot AND whose α is well-aligned with
     the query's content α.
  3. Concatenating slots into a single word stream.

This is generation, not retrieval: the SHAPE of each phrase comes from
corpus statistics (a learned grammar of how phrases look), but every
WORD comes from the manifold's own machinery.  No real human-written
phrase is ever copied wholesale.

The key property: words within a phrase template are PICKED TOGETHER
under the same α-target, so they should be more mutually coherent than
the word-by-word random walk does on its own.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Optional

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from word_manifold import WordManifold

WORD_RE = re.compile(r"[a-zA-Z']+")

# Map spaCy POS → 8 channels.  DET/PART → channel 4 (Pro) since they're
# function determiners; we treat NUM/PUNCT/SYM as transparent.
SPACY_TO_CH: dict[str, int] = {
    "NOUN": 0, "PROPN": 0,
    "VERB": 1, "AUX":   1,
    "ADJ":  2,
    "ADV":  3,
    "PRON": 4, "DET":  4, "PART": 4,
    "ADP":  5,
    "CCONJ":6, "SCONJ":6,
    "INTJ": 7,
}
N_CH = 8


class PhraseJumpGenerator:
    """One generator per route (article / dialog)."""

    def __init__(self, route: str,
                  rng_seed: int = 0,
                  n_top_per_slot: int = 60,
                  use_phrase_manifold: bool = False):
        # `use_phrase_manifold=True` enables α-aligned template selection
        # and phrase-bigram transitions during _sample_templates.  In v12
        # preflight this lowered all-four pass rate (49 → 43) at v1
        # weights; the sophistication outweighed the signal.  Kept as
        # an opt-in flag so the data file is usable by future
        # experimentation without breaking the default path.
        self.use_phrase_manifold = use_phrase_manifold
        self.route = route
        self.rng = np.random.default_rng(rng_seed)
        self.n_top_per_slot = n_top_per_slot

        procd = ROOT / "data" / "processed"
        # Load route-specific world manifold (for filling slots) and
        # universal manifold (for query α).
        self.wm_route = WordManifold.load(
            str(procd / f"manifold_{'article' if route == 'article' else 'dialog'}.pkl"))
        self.wm_universal = WordManifold.load(str(procd / "manifold.pkl"))
        self.wm_dict = WordManifold.load(str(procd / "manifold_dictionary.pkl"))

        # Pre-compute dominant POS per word in the route manifold.
        self.pos_arg = self.wm_route.pi.argmax(axis=1)            # (N,)
        # For each channel, which lemma indices have that as dominant.
        # Sort by frequency (descending) so top-N is fast to access.
        self.cand_by_ch: dict[int, np.ndarray] = {}
        for ch in range(N_CH):
            idx = np.where(self.pos_arg == ch)[0]
            order = np.argsort(-self.wm_route.m[idx])
            self.cand_by_ch[ch] = idx[order]

        # Load semantic-role priors for this route.  If they don't exist
        # yet, fall back to no-role-bias.  Built by build_word_roles.py.
        self._role_prior = None        # (N, K) or None
        self._role_classes: list[str] = []
        try:
            from roles import load_word_roles
            self._role_prior, self._role_classes = load_word_roles(route)
        except FileNotFoundError as e:
            print(f"[phrase] role priors unavailable for {route}: {e}",
                  flush=True)

        # Load phrase manifold (per-template α/π/ω + template-bigram)
        self._phman = None
        phman_path = procd / f"phrase_manifold_{route}.npz"
        if phman_path.exists():
            z = np.load(phman_path, allow_pickle=True)
            self._phman = {
                "templates":  [tuple(t) for t in z["templates"]],
                "mean_alpha": z["mean_alpha"],
                "mean_pi":    z["mean_pi"],
                "mean_omega": z["mean_omega"],
                "counts":     z["counts"],
                "bigram":     z["bigram"],
            }
            self._phman_idx = {t: i for i, t in
                                  enumerate(self._phman["templates"])}
        else:
            self._phman_idx = {}

        # Load per-verb subcategorization priors.  When a slot is OBJ/POBJ
        # and the previous filled word was a VERB, the slot's α-target is
        # blended with that verb's expected obj_alpha — i.e. selectional
        # restrictions apply.  Without this file, we skip the blend.
        self._verb_subcat = None
        vc_path = procd / f"verb_subcat_{route}.npz"
        if vc_path.exists():
            z = np.load(vc_path, allow_pickle=True)
            self._verb_subcat = {
                "subj_alpha": z["subj_alpha"],
                "obj_alpha":  z["obj_alpha"],
                "n_subj":     z["n_subj"],
                "n_obj":      z["n_obj"],
            }

        # Load templates and their probabilities.
        tpl_path = procd / f"phrase_templates_{route}.json"
        if not tpl_path.exists():
            raise FileNotFoundError(
                f"phrase templates missing: {tpl_path} — run build_phrase_templates.py")
        tpl_blob = json.loads(tpl_path.read_text())
        # tpl_blob is list of {"pos":[...], "count": N}
        self.templates = [tuple(t["pos"]) for t in tpl_blob]
        self.template_freq = np.array([t["count"] for t in tpl_blob],
                                          dtype=np.float64)
        self.template_p = self.template_freq / self.template_freq.sum()

        # Index templates by total length for length-targeted sampling.
        self.templates_by_len: dict[int, list[int]] = {}
        for i, t in enumerate(self.templates):
            self.templates_by_len.setdefault(len(t), []).append(i)

    # ------------------------------------------------------------------
    # Query α: mean α over CONTENT tokens of the query (Noun/Verb/Adj/Adv).
    # Function words are dropped because their α is generic and dilutes
    # the topic signal.  The dictionary world manifold is used so the α
    # captures definitional semantics, not corpus surface frequency.
    # ------------------------------------------------------------------
    _CONTENT_CH = (0, 1, 2, 3)
    def _query_alpha(self, query: str) -> np.ndarray | None:
        toks = WORD_RE.findall(query.lower())
        idx = [self.wm_dict.idx[t] for t in toks
                if t in self.wm_dict.idx]
        if not idx:
            return None
        # Filter to content words only.
        pos_arg = self.wm_dict.pi.argmax(axis=1)
        idx = [i for i in idx if int(pos_arg[i]) in self._CONTENT_CH]
        if not idx:
            return None
        # Use the route world manifold's α at those lemma indices so
        # the dot-product target is route-aware (different routes
        # weight the same lemma differently).
        return self.wm_route.alpha[np.asarray(idx)].mean(axis=0)

    # ------------------------------------------------------------------
    # Slot fill: pick a word for a slot of given POS, biased by α-alignment.
    # ------------------------------------------------------------------
    def _fill_slot(self, spacy_pos: str, q_alpha: np.ndarray | None,
                    used: set[int],
                    role: str | None = None,
                    prev_verb_idx: int | None = None) -> int | None:
        ch = SPACY_TO_CH.get(spacy_pos)
        if ch is None:
            return None
        cand = self.cand_by_ch.get(ch)
        if cand is None or cand.size == 0:
            return None

        # Build the α target.  Default = query α.  When this slot is an
        # OBJ/POBJ of a verb we just filled, blend in the verb's expected
        # object α (selectional restriction).  This is the predicate-
        # semantics signal: a verb's object should look like the things
        # that verb typically operates on, not just like the query topic.
        target_alpha = q_alpha
        if (role in ("obj", "pobj")
                and prev_verb_idx is not None
                and self._verb_subcat is not None
                and self._verb_subcat["n_obj"][prev_verb_idx] >= 5):
            verb_obj = self._verb_subcat["obj_alpha"][prev_verb_idx]
            if q_alpha is not None:
                # 50/50 blend: verb constraints meet query topic.
                target_alpha = 0.5 * q_alpha + 0.5 * verb_obj
            else:
                target_alpha = verb_obj

        # Score the FULL candidate set.
        if target_alpha is None or ch in (4, 5, 6):
            score = np.log1p(self.wm_route.m[cand].astype(np.float64))
        else:
            A = self.wm_route.alpha[cand]
            num = A @ target_alpha
            den = (np.linalg.norm(A, axis=1) *
                    max(np.linalg.norm(target_alpha), 1e-9))
            score = num / np.maximum(den, 1e-9)
            score += 0.05 * np.log1p(self.wm_route.m[cand].astype(np.float64))

        # Role prior intentionally disabled at slot-fill: empirical
        # tests (v9 weight 0.8 → 45/50, v10 weight 0.3 → 42/50, vs
        # v8 weight 0 → 47/50) showed any slot-fill role bias narrows
        # candidates away from α-aligned topical words, lowering
        # boundary pass rate.  Role information is consumed by the
        # grammar layer instead — subj-verb agreement and pronoun case.

        # Drop already-used words.
        used_set = set(used)
        keep_mask = np.array([i not in used_set for i in cand])
        if not keep_mask.any():
            keep_mask = np.ones_like(keep_mask)
        kept_idx = cand[keep_mask]
        kept_sc  = score[keep_mask]
        # Top-K softmax sample.
        k = min(20, kept_idx.size)
        order = np.argpartition(-kept_sc, k - 1)[:k]
        chosen_idx = kept_idx[order]
        chosen_sc = kept_sc[order]
        chosen_sc = chosen_sc - chosen_sc.max()
        p = np.exp(chosen_sc / 0.6)
        p /= p.sum()
        pick = int(self.rng.choice(chosen_idx, p=p))
        return pick

    # ------------------------------------------------------------------
    # Template sampling.
    #
    # If a phrase manifold exists for this route, the first template is
    # sampled biased by query α-alignment (mean_alpha[T] · q_alpha) +
    # frequency, and subsequent templates by the phrase bigram given the
    # previous template.  Otherwise fall back to independent
    # frequency-weighted sampling.
    # ------------------------------------------------------------------
    def _sample_templates(self, target_len: int,
                            max_phrases: int = 3,
                            q_alpha: np.ndarray | None = None
                            ) -> list[tuple[str, ...]]:
        out: list[tuple[str, ...]] = []
        remaining = target_len
        prev_idx: int | None = None

        for _ in range(max_phrases):
            if remaining <= 0:
                break
            # Templates whose length fits remaining budget.
            ok_idx = [i for i, t in enumerate(self.templates)
                       if 1 <= len(t) <= remaining]
            if not ok_idx:
                break
            ok_arr = np.asarray(ok_idx)
            base = self.template_freq[ok_arr].astype(np.float64)
            # Phrase manifold available AND opt-in flag set → factor in
            # α-alignment for the first template, bigram transition for
            # subsequent ones.  Off by default; v12 preflight showed this
            # regressed boundary pass rate at v1 weights.
            if self._phman is not None and self.use_phrase_manifold:
                # Map global template index → phrase-manifold idx
                phman_ix = np.array([
                    self._phman_idx.get(self.templates[i], -1)
                    for i in ok_idx
                ])
                valid_in_phman = phman_ix >= 0
                if valid_in_phman.any():
                    if prev_idx is None:
                        # First template: α-alignment of mean_alpha[T]
                        # with the query α.
                        score = base.copy()
                        if q_alpha is not None:
                            qan = max(np.linalg.norm(q_alpha), 1e-9)
                            for j, k in enumerate(phman_ix):
                                if k < 0:
                                    continue
                                ma = self._phman["mean_alpha"][k]
                                an = max(np.linalg.norm(ma), 1e-9)
                                cos = float(ma @ q_alpha) / (an * qan)
                                score[j] = score[j] * (1.5 + cos)
                        p = score / score.sum()
                    else:
                        # Subsequent: phrase bigram given previous.
                        prev_phman = self._phman_idx.get(
                            self.templates[prev_idx], -1)
                        if prev_phman >= 0:
                            row = self._phman["bigram"][prev_phman]
                            score = base.copy()
                            for j, k in enumerate(phman_ix):
                                if k < 0:
                                    continue
                                # Multiplicative blend of frequency and
                                # bigram transition probability.
                                score[j] *= (0.2 + row[k])
                            p = score / score.sum()
                        else:
                            p = base / base.sum()
                else:
                    p = base / base.sum()
            else:
                p = base / base.sum()

            pick = int(self.rng.choice(ok_arr, p=p))
            t = self.templates[pick]
            out.append(t)
            remaining -= len(t)
            prev_idx = pick
        return out

    # ------------------------------------------------------------------
    # Generate a candidate response.
    # ------------------------------------------------------------------
    def generate_candidate(self, query: str, length: int = 8,
                            apply_grammar: bool = True) -> dict:
        q_alpha = self._query_alpha(query)
        templates = self._sample_templates(length, q_alpha=q_alpha)
        if not templates:
            return {"text": "", "b_idx": [], "templates": [], "phrases": []}

        # Per-phrase: track template + role labels + the words filling them.
        # Also maintain `prev_verb_idx`: when we just filled a VERB slot,
        # subsequent OBJ/POBJ slots blend in that verb's expected obj α
        # (selectional restriction).
        from roles import infer_template_roles
        phrases: list[dict] = []
        used: set[int] = set()
        prev_verb_idx: int | None = None
        for t in templates:
            slot_roles = infer_template_roles(t)
            tpl_words: list[str] = []
            tpl_idx:   list[int] = []
            for pos, role in zip(t, slot_roles):
                pick = self._fill_slot(pos, q_alpha, used, role=role,
                                          prev_verb_idx=prev_verb_idx)
                if pick is not None:
                    tpl_words.append(self.wm_route.lemmas[pick])
                    tpl_idx.append(pick)
                    used.add(pick)
                    # Track this verb if it's a VERB slot (so the next
                    # OBJ/POBJ in this template — or in the next phrase —
                    # can use selectional restrictions).
                    if pos == "VERB":
                        prev_verb_idx = pick
                    elif role in ("obj", "pobj"):
                        # A verb's object slot has been consumed; clear
                        # so the next obj doesn't leak the same verb.
                        prev_verb_idx = None
                else:
                    tpl_words.append("")
                    tpl_idx.append(-1)
            phrases.append({"template": list(t),
                              "roles":   slot_roles,
                              "words":   tpl_words,
                              "idx":     tpl_idx})

        # Compose: optionally run grammar layer.
        if apply_grammar:
            from grammar import grammar_compose
            text = grammar_compose(phrases, query=query)
            # Re-extract the polished surface words so the boundary scorer
            # can score the post-grammar string.
            polished_words = [t for t in text.replace(".", "")
                                .replace("?", "")
                                .replace("!", "").split() if t]
            polished_words = [w.lower() for w in polished_words]
        else:
            flat_words = [w for ph in phrases for w in ph["words"] if w]
            text = " ".join(flat_words)
            if text:
                text = text[0].upper() + text[1:] + "."
            polished_words = flat_words

        flat_idx = [i for ph in phrases for i in ph["idx"] if i >= 0]
        return {
            "text":      text,
            "b_idx":     flat_idx,
            "words":     polished_words,
            "phrases":   phrases,
            "templates": [list(t) for t in templates],
        }


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------
def _cli():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", "-q", action="append", default=[])
    ap.add_argument("--route", default="dialog",
                     choices=["article", "dialog"])
    ap.add_argument("--n", type=int, default=5,
                     help="how many candidates per query")
    ap.add_argument("--length", type=int, default=8)
    args = ap.parse_args()

    queries = args.query or [
        "what is entropy",
        "I miss her so much",
        "how does photosynthesis work",
    ]
    g = PhraseJumpGenerator(args.route, rng_seed=42)
    print(f"[phrase] route={args.route}  "
          f"templates={len(g.templates):,}", flush=True)
    for q in queries:
        print(f"\nQ: {q}", flush=True)
        for k in range(args.n):
            g.rng = np.random.default_rng(42 + k * 17 + abs(hash(q)) % 1000)
            out = g.generate_candidate(q, length=args.length)
            print(f"  [{k}] tpl={out['templates']}  -> {out['text']}",
                  flush=True)


if __name__ == "__main__":
    _cli()
