"""Aisha responder on the POS-aligned 6D-style manifold.

Same two-mechanism architecture as `responder.Responder` but the
geometric layer is replaced with `POSKahlerRuntime` operating on
16-real / 8-complex POS-aligned coordinates.

Mapping vs old responder:
  - 8D q (abstract) → 16D q (8 POS-role + 8 POS-shift)
  - Kähler distance now meaningful in POS coordinates
  - ξ-flow now operates in POS-shift directions
"""
from __future__ import annotations

import re
import time
from pathlib import Path

import numpy as np

from aisha_respond import Responder as InnerResponder
from conversation_memory_v2 import ConversationMemory
from grammar_template import (build_grammar_template,
                                  NOUN, VERB, ADJ, ADV, PRON_DET, PREP,
                                  CONJ, INTJ, CONTENT_POS, FUNCTION_POS)
from kahler_pos_runtime import POSKahlerRuntime
from phrase_pool import PhrasePool
from pos_bigram_template import POSBigramTemplate
from word_manifold import WordManifold

try:
    import grammar as G
    _GRAMMAR = True
except ImportError:
    _GRAMMAR = False
try:
    import harper_polish as HARPER
    _HARPER = True
except ImportError:
    _HARPER = False


WORD_RE = re.compile(r"[a-zA-Z']+")
PROC = Path(__file__).resolve().parent / "data" / "processed"

_VERB_ONLY = frozenset([
    "is", "are", "was", "were", "be", "been", "being", "am",
    "do", "does", "did", "doing", "done",
    "has", "have", "had", "having",
    "will", "would", "could", "should", "shall", "must",
    "may", "might", "can", "ought",
])

# Canonical English function vocabulary — function-pool will be the
# intersection of (POS-tagged-as-function) AND (in this list).  Keeps
# real glue words; excludes `madras`, `whack`, `ix`, etc. that may have
# accidentally been POS-tagged as function words.
# PRON_DET split into the words that act as DETERMINER vs PRONOUN.
# Determiners attach to a head noun; pronouns stand alone.  Splitting
# matters at sentence-end (we want pronouns there, not articles) and
# at slots where no NOUN follows.
_DETERMINERS = frozenset([
    "the", "a", "an", "this", "that", "these", "those",
    "my", "your", "his", "her", "our", "their", "its",
    "some", "any", "no", "each", "every", "both",
    "much", "many", "few", "several", "every",
])
_PRONOUNS = frozenset([
    "i", "you", "he", "she", "it", "we", "they",
    "me", "him", "them", "us",
    "myself", "yourself", "itself", "themselves",
    "ourselves", "himself", "herself",
    "something", "anything", "nothing", "everything",
    "someone", "anyone", "nobody", "everybody", "somebody",
    "what", "which", "who", "whom", "whose",
    "all", "either", "neither",
])

_CANONICAL_FUNCTION = {
    PRON_DET: _DETERMINERS | _PRONOUNS,
    PREP: frozenset([
        "of", "in", "on", "at", "by", "for", "with", "to", "as",
        "into", "onto", "upon", "between", "among", "through",
        "across", "around", "beside", "behind", "before", "after",
        "during", "until", "since", "against",
        "about", "above", "below", "near", "over", "under",
        "off", "out", "up", "down",
        "from", "without", "within", "throughout",
        "along", "toward", "towards", "beyond",
    ]),
    CONJ: frozenset([
        "and", "but", "or", "nor", "yet", "so",
        "if", "when", "where", "while", "because", "although",
        "though", "since", "unless", "until", "as", "than",
        "that", "whether", "how", "why",
    ]),
    INTJ: frozenset([
        "well", "oh", "ah", "hey", "hmm", "yes", "no", "sure",
        "okay", "ok", "wow", "huh", "yeah",
    ]),
}


class POSResponder:
    """Symplectic-flow responder on POS-aligned manifold."""

    def __init__(self, *,
                  use_harper: bool = False,
                  memory: ConversationMemory | None = None,
                  content_boundary_k: int = 20,
                  function_pool_size: int = 50,         # tightened from 120
                  common_content_size: int = 1500):
        print("[aisha-pos] loading inner Responder…", flush=True)
        self.R = InnerResponder(rng_seed=0)
        self.wm = self.R.wm
        self.pos_arg = self.wm.pi.argmax(axis=1)
        self.use_harper = use_harper and _HARPER
        self.memory = memory

        # POS-Kähler runtime
        self.kahler = POSKahlerRuntime()
        print(f"[aisha-pos] POS-Kähler runtime ready  "
                f"({self.kahler.N} words on POS manifold)", flush=True)

        # Map old-vocab idx → new POS-manifold idx (and inverse)
        self._old_to_new = -np.ones(self.wm.N, dtype=np.int64)
        for new_i, old_i in enumerate(self.kahler.word_idx_orig):
            self._old_to_new[old_i] = new_i
        # q_all in the new manifold (16D, normalized for training)
        self._main_q = self.kahler.q_all                      # (N_pos, 16)
        # m for tier-2 fallback ranking
        self._main_m = self.kahler.m

        self.content_boundary_k = content_boundary_k
        self._build_function_pool(function_pool_size)
        self._build_common_content_pool(common_content_size)

        # Pre-cache R² and scalar per POS-manifold word, for translator biases.
        # Source: Riemann scoring saved in word_curvature.npz, indexed by old vocab.
        try:
            cur = np.load(PROC / "kahler_phase1" / "word_curvature.npz")
            R2_old = np.zeros(self.wm.N, dtype=np.float64)
            R2_old[cur["word_idx"]] = cur["R2"]
            sc_old = np.zeros(self.wm.N, dtype=np.float64)
            sc_old[cur["word_idx"]] = cur["scalar"]
            self._R2_per_pos = R2_old[self.kahler.word_idx_orig]   # (N_pos,)
            self._scalar_per_pos = sc_old[self.kahler.word_idx_orig]
        except FileNotFoundError:
            self._R2_per_pos = None
            self._scalar_per_pos = None
        # X (POS-shift) coords are already on POS manifold
        self._X_per_pos = self.kahler.X_norm                          # (N_pos, 8)

        # Translator biases — five corrections derived from the
        # cumulative-flow analysis (B vs Aisha divergence is
        # systematic + monotonic).
        # Targets are *B's measured per-step values* averaged across styles.
        self.translator_strength = 0.0    # OFF — keep analytical generation clean
        self.target_step_dist = 25.0       # B's typical step length (Aisha ~14)
        self.target_R2 = 6e4               # B's per-word R² (Aisha ~2e5)
        self.target_first_scalar = 0.0     # B starts near 0 (Aisha at +19)

        # Post-hoc translator (II) — load B's reference trajectory mean
        # values per cum_step.  These are pre-computed from the
        # flow_signature_analysis.  Used by `translate()` to substitute
        # Aisha's words with same-POS replacements that bring her path
        # closer to B's curve.
        self._ref_B_step = {
            # cum_step at step k (1-10) for B casual (averaged)
            "casual":     np.array([0, 19.5, 37.3, 53.2, 71.4, 91.1, 108.4,
                                          126.4, 144.5, 161.7]),
            "scientific": np.array([0, 24.6, 44.3, 65.0, 87.2, 109.9, 130.9,
                                          150.1, 172.6, 188.7]),
        }
        self._ref_B_R2 = {
            "casual":     np.array([6e3, 5e4, 8.5e4, 1.4e5, 2.2e5, 2.8e5,
                                          3.8e5, 4.7e5, 5.1e5, 5.9e5]),
            "scientific": np.array([6.5e4, 2e5, 3.3e5, 6.8e5, 9.4e5, 1.0e6,
                                          1.3e6, 1.4e6, 1.6e6, 2.0e6]),
        }

        # Corpus-fitted POS-bigram template sampler (used by /alt path)
        self.template_sampler = POSBigramTemplate(self.wm)
        # Phrase pool — corpus collocations for content-slot boosting
        self.phrases = PhrasePool(self.wm)
        # Strength of collocation boost (0 = off; ~1 = strong).
        # 0.3 found to balance grammar improvement against TTR dropoff.
        self.colloc_strength = 0.3
        print(f"[aisha-pos] POS-bigram + phrase pool ready", flush=True)

    # --------- pools (POS-manifold-indexed) ---------

    def _build_function_pool(self, size: int):
        """Per-POS function-word pool, intersected with a canonical English
        function vocabulary.  Stricter than before: top-50 instead of 120,
        only words that are KNOWN English glue can appear."""
        self._fn_pool: dict[int, list[int]] = {}
        m = self._main_m
        pi = np.asarray(self.wm.pi, dtype=np.float64)
        pi_total = pi.sum(axis=1) + 1e-12
        old_idx = self.kahler.word_idx_orig
        word_pos_arg = self.pos_arg[old_idx]
        word_pi = pi[old_idx]
        word_pi_total = pi_total[old_idx]
        word_lemmas = self.kahler.lemmas
        for pos_id in (PRON_DET, PREP, CONJ, INTJ):
            canonical = _CANONICAL_FUNCTION.get(pos_id, frozenset())
            share = word_pi[:, pos_id] / word_pi_total
            mask = (word_pos_arg == pos_id) & (share > 0.5) & (m > 500.0)
            # Intersect with canonical English glue vocabulary
            in_canonical = np.array(
                [word_lemmas[i] in canonical for i in range(self.kahler.N)])
            mask &= in_canonical
            cands = np.where(mask)[0]
            if len(cands) == 0:
                self._fn_pool[pos_id] = []; continue
            order = cands[np.argsort(-m[cands])]
            self._fn_pool[pos_id] = order[:size].tolist()

        # Split PRON_DET into determiner-only and pronoun-only sub-pools
        det_pool, pron_pool = [], []
        for i in self._fn_pool.get(PRON_DET, []):
            lemma = word_lemmas[i]
            if lemma in _DETERMINERS:
                det_pool.append(i)
            if lemma in _PRONOUNS:
                pron_pool.append(i)
        self._det_pool = det_pool
        self._pron_pool = pron_pool

    def _build_common_content_pool(self, size: int):
        self._content_pool: dict[int, list[int]] = {}
        m = self._main_m
        pi = np.asarray(self.wm.pi, dtype=np.float64)
        pi_total = pi.sum(axis=1) + 1e-12
        old_idx = self.kahler.word_idx_orig
        word_pos_arg = self.pos_arg[old_idx]
        word_pi = pi[old_idx]
        word_pi_total = pi_total[old_idx]
        word_lemmas = self.kahler.lemmas
        for pos_id in (NOUN, VERB, ADJ, ADV):
            share = word_pi[:, pos_id] / word_pi_total
            mask = ((word_pos_arg == pos_id) & (share > 0.7) & (m > 100.0))
            if pos_id != VERB:
                bad_mask = np.array([word_lemmas[i] in _VERB_ONLY
                                          for i in range(self.kahler.N)])
                mask &= ~bad_mask
            cands = np.where(mask)[0]
            order = cands[np.argsort(-m[cands])]
            self._content_pool[pos_id] = order[:size].tolist()

    def _strict_pos(self, idx_pos: int, target_pos: int,
                       threshold: float = 0.7) -> bool:
        """idx_pos is index into POS-manifold."""
        old_idx = int(self.kahler.word_idx_orig[idx_pos])
        if int(self.pos_arg[old_idx]) != target_pos:
            return False
        if (target_pos != VERB
                and self.kahler.lemmas[idx_pos] in _VERB_ONLY):
            return False
        row = self.wm.pi[old_idx]
        total = float(row.sum())
        if total <= 0:
            return False
        return float(row[target_pos]) / total >= threshold

    # --------- boundary & seed ---------

    def expand_content_boundary(self, query: str) -> set[int]:
        """k-NN under POS-Kähler distance from query content seeds."""
        seeds: list[int] = []
        for w in WORD_RE.findall(query.lower()):
            old_i = self.wm.idx.get(w)
            if old_i is None or self.R.is_stopword[old_i]:
                continue
            new_i = int(self._old_to_new[old_i])
            if new_i >= 0:
                seeds.append(new_i)
        if not seeds:
            return set()
        boundary: set[int] = set(seeds)
        Q = self._main_q
        for s_i in seeds:
            d2 = self.kahler.mahalanobis_to_seed(Q[s_i], Q)
            d2[s_i] = np.inf
            top = np.argpartition(d2, self.content_boundary_k
                                       )[:self.content_boundary_k]
            for j in top:
                boundary.add(int(j))
        return boundary

    # ---------- mode selector + seed strategies ----------

    def _select_mode(self, query: str) -> str:
        """Pick one of {'echo', 'anchor', 'redirect'} based on the prompt's
        surface signals.  Mirrors the three response modes humans use,
        which our position-gap diagnostic showed Aisha was lacking."""
        q = query.strip()
        words = WORD_RE.findall(q.lower())
        n_words = len(words)
        first_word = words[0] if words else ""
        ends_q = q.endswith("?")
        has_exclaim = "!" in q
        # very short prompt → redirect (carry the conversation forward)
        if n_words < 4 and not ends_q:
            return "redirect"
        # exclamation or strong-affect first word → echo (match register)
        if has_exclaim or first_word in {
            "yes", "yeah", "wow", "no", "really", "exactly", "ha",
            "haha", "oh", "hey", "ouch", "ugh", "huh", "ah",
            "great", "nice", "cool", "right", "true",
        }:
            return "echo"
        # explicit question → anchor (respond to the question)
        if ends_q:
            return "anchor"
        # default
        return "anchor"

    def _seed_for_mode(self, query: str, mode: str
                          ) -> tuple[int | None, np.ndarray]:
        """Return (seed_idx, seed_q) chosen by mode.

        anchor  : highest-q-norm content word in query  (the topic center)
        echo    : LAST content word in query             (sit near A's end)
        redirect: a Kähler-far neighbor of the query's seed (carry forward)
        """
        # Collect query content-word indices
        cand_new: list[int] = []
        cand_norms: list[float] = []
        for w in WORD_RE.findall(query.lower()):
            old_i = self.wm.idx.get(w)
            if old_i is None or self.R.is_stopword[old_i]:
                continue
            new_i = int(self._old_to_new[old_i])
            if new_i < 0:
                continue
            cand_new.append(new_i)
            cand_norms.append(float(np.linalg.norm(self._main_q[new_i])))
        if not cand_new:
            # fallback: any word in vocab
            for w in WORD_RE.findall(query.lower()):
                old_i = self.wm.idx.get(w)
                if old_i is None: continue
                new_i = int(self._old_to_new[old_i])
                if new_i >= 0:
                    return new_i, self._main_q[new_i].copy()
            return None, self._main_q.mean(axis=0)

        if mode == "echo":
            # last content word
            seed = cand_new[-1]
            return seed, self._main_q[seed].copy()
        elif mode == "redirect":
            # pick the candidate that's FARTHEST under the metric from
            # the query's centroid — a real pivot
            anchor = cand_new[int(np.argmax(cand_norms))]
            anchor_q = self._main_q[anchor]
            d2 = self.kahler.mahalanobis_to_seed(
                anchor_q, self._main_q[np.asarray(cand_new)])
            far = cand_new[int(np.argmax(d2))]
            return far, self._main_q[far].copy()
        else:  # "anchor"
            seed = cand_new[int(np.argmax(cand_norms))]
            return seed, self._main_q[seed].copy()

    def _seed_from_query(self, query: str) -> tuple[int | None, np.ndarray]:
        """Compatibility wrapper — defaults to mode-aware selection."""
        mode = self._select_mode(query)
        return self._seed_for_mode(query, mode)

    # --------- per-slot picks ---------

    def _pick_content(self, seed_q: np.ndarray, boundary: set[int],
                          target_pos: int, used: set[int],
                          prev_word: str | None = None,
                          prev_q: np.ndarray | None = None,
                          step_idx: int = 0,
                          cum_X: np.ndarray | None = None
                          ) -> tuple[str | None, int | None]:
        cands = [i for i in boundary
                    if i not in used and self._strict_pos(i, target_pos)]
        if cands:
            return self._nearest(seed_q, cands, prev_word=prev_word,
                                       prev_q=prev_q, step_idx=step_idx,
                                       cum_X=cum_X)
        cands = [i for i in self._content_pool.get(target_pos, [])
                    if i not in used]
        if cands:
            return self._nearest(seed_q, cands, prev_word=prev_word,
                                       prev_q=prev_q, step_idx=step_idx,
                                       cum_X=cum_X)
        return None, None

    def _pick_function(self, seed_q: np.ndarray, target_pos: int,
                            used: set[int],
                            sub_pool: list[int] | None = None
                            ) -> tuple[str | None, int | None]:
        """Pick function word via ξ-flow.  If sub_pool is given, restrict
        to that index list (used to force pronoun-only or determiner-only
        within PRON_DET)."""
        base_pool = sub_pool if sub_pool is not None \
                       else self._fn_pool.get(target_pos, [])
        cands = [i for i in base_pool if i not in used]
        if not cands:
            return None, None
        xi = self.kahler.hamiltonian_flow_at(seed_q[None, :])[0]
        xi_norm = float(np.linalg.norm(xi)) + 1e-12
        cands_arr = np.asarray(cands)
        disp = self._main_q[cands_arr] - seed_q[None, :]
        d_norm = np.linalg.norm(disp, axis=1) + 1e-12
        cos = (disp @ xi) / (d_norm * xi_norm)
        m = self._main_m[cands_arr]
        score = np.maximum(cos, 0.0) * (m ** 0.4)
        if score.max() <= 0:
            best = cands_arr[int(np.argmax(m))]
        else:
            best = cands_arr[int(np.argmax(score))]
        return self.kahler.lemmas[best], int(best)

    def _nearest(self, seed_q: np.ndarray, cand_idx: list[int],
                    prev_word: str | None = None,
                    prev_q: np.ndarray | None = None,
                    step_idx: int = 0,
                    cum_X: np.ndarray | None = None) -> tuple[str, int]:
        """Pick nearest candidate under Mahalanobis + bias corrections.

        Translator biases (B vs Aisha gap correction):
          B1 step-length: prefer candidates whose step distance ≈ target_step_dist
          B2 low-R²:      prefer candidates with low Riemann magnitude
          B3 POS-spread:  prefer candidates that grow the cumulative X-norm
          B5 neutral-scalar at sentence start: prefer scalar≈0 for first content
        """
        cands_arr = np.asarray(cand_idx)
        d2 = self.kahler.mahalanobis_to_seed(seed_q,
                                                  self._main_q[cands_arr])
        # memory bonus
        if self.memory is not None:
            for i, c in enumerate(cands_arr):
                old_i = int(self.kahler.word_idx_orig[c])
                bonus = self.memory.vocab_bonus(old_i)
                d2[i] /= max(bonus, 1e-6)
        # collocation boost
        if prev_word and self.colloc_strength > 0:
            for i, c in enumerate(cands_arr):
                lemma = self.kahler.lemmas[c]
                fr = self.phrases.collocation_score(prev_word, lemma)
                if fr > 0:
                    boost = 1.0 + self.colloc_strength * np.log1p(fr)
                    d2[i] /= boost
        # ---------- translator biases ----------
        if self.translator_strength > 0:
            ts = self.translator_strength
            cand_q = self._main_q[cands_arr]                     # (M, 16)
            # B1 — step-length bias: prefer step ≈ target.  Gaussian on
            # |step - target| with σ = 10.
            if prev_q is not None:
                steps = np.linalg.norm(cand_q - prev_q[None, :], axis=1)
                step_bias = np.exp(
                    -ts * ((steps - self.target_step_dist) ** 2)
                       / (2.0 * 10.0 ** 2))
                d2 /= np.maximum(step_bias, 1e-9)
            # B2 — low-R² bias: penalise high curvature
            if self._R2_per_pos is not None:
                R2 = self._R2_per_pos[cands_arr]
                # bias = exp(-ts · log1p(R²/target))   — larger R² → smaller bias
                R2_bias = np.exp(
                    -ts * 0.5 * np.maximum(0.0,
                          np.log1p(R2 / max(self.target_R2, 1.0))))
                d2 /= np.maximum(R2_bias, 1e-9)
            # B3 — POS-shift diversity: prefer candidate that grows ‖cum_X‖
            if cum_X is not None:
                new_cum = cum_X[None, :] + self._X_per_pos[cands_arr]   # (M, 8)
                gain = (np.linalg.norm(new_cum, axis=1)
                          - float(np.linalg.norm(cum_X)))
                # boost candidates whose addition increases drift
                X_bias = np.exp(ts * 0.05 * np.maximum(gain, 0.0))
                d2 /= np.maximum(X_bias, 1e-9)
            # B5 — neutral scalar at sentence start (step_idx == 0)
            if step_idx == 0 and self._scalar_per_pos is not None:
                sc = self._scalar_per_pos[cands_arr]
                neutral_bias = np.exp(-ts * (sc ** 2) / (2.0 * 5.0 ** 2))
                d2 /= np.maximum(neutral_bias, 1e-9)
        best = cands_arr[int(np.argmin(d2))]
        return self.kahler.lemmas[best], int(best)

    def _pick_article(self, slot_i: int, rank_offset: int) -> str:
        if slot_i == 0:
            DETS = ["the", "a", "this", "that", "my", "your",
                       "his", "her", "our", "their", "an",
                       "these", "those", "some", "any", "no"]
            return DETS[rank_offset % len(DETS)]
        return "a"

    _FALLBACK = {NOUN: ["thing", "moment", "way", "place"],
                    VERB: ["make", "give", "find", "take"],
                    ADJ:  ["good", "real", "small", "new"],
                    ADV:  ["well", "really", "still", "again"],
                    PRON_DET: "the", PREP: "in", CONJ: "and",
                    INTJ: "well"}

    def _fallback(self, pos: int, used_strs: set[str]) -> str:
        opts = self._FALLBACK.get(pos, "thing")
        if isinstance(opts, str): return opts
        for w in opts:
            if w not in used_strs:
                return w
        return opts[0]

    # --------- template walk ---------

    def fill_template(self, template: list[int], query: str,
                          boundary: set[int],
                          rank_offset: int = 0,
                          mode: str | None = None) -> list[str]:
        if mode is None:
            mode = self._select_mode(query)
        seed_i, seed_q = self._seed_for_mode(query, mode)
        self._last_mode = mode
        # Translator state — running cumulative path info.
        prev_q: np.ndarray | None = None
        cum_X = np.zeros(8, dtype=np.float64)
        content_step = 0                            # index among content slots
        words: list[str] = []
        used: set[int] = set([seed_i] if seed_i is not None else [])
        used_strs: set[str] = set()
        n = len(template)

        for slot_i, pos in enumerate(template):
            chosen: str | None = None; chosen_idx: int | None = None
            if pos == PRON_DET:
                next_pos = template[slot_i + 1] if slot_i + 1 < n else None
                is_last = slot_i == n - 1
                if next_pos in (NOUN, ADJ):
                    chosen = self._pick_article(slot_i, rank_offset)
                elif is_last:
                    # Sentence-end PRON_DET → must be a pronoun, not article
                    chosen, chosen_idx = self._pick_function(
                        seed_q, PRON_DET, used, sub_pool=self._pron_pool)
                else:
                    # Bare pronoun position (not before NOUN/ADJ) — pronouns
                    chosen, chosen_idx = self._pick_function(
                        seed_q, PRON_DET, used, sub_pool=self._pron_pool)
            elif pos == PREP:
                chosen, chosen_idx = self._pick_function(
                    seed_q, PREP, used)
            elif pos == CONJ:
                chosen, chosen_idx = self._pick_function(
                    seed_q, CONJ, used)
            elif pos == INTJ:
                chosen, chosen_idx = self._pick_function(
                    seed_q, INTJ, used)
            else:
                prev_w = words[-1].lower() if words else None
                chosen, chosen_idx = self._pick_content(
                    seed_q, boundary, pos, used,
                    prev_word=prev_w, prev_q=prev_q,
                    step_idx=content_step, cum_X=cum_X.copy())
            if chosen is None:
                chosen = self._fallback(pos, used_strs)

            words.append(chosen)
            used_strs.add(chosen.lower())
            if chosen_idx is not None:
                used.add(chosen_idx)
                # Update translator state
                prev_q = self._main_q[chosen_idx].copy()
                cum_X = cum_X + self._X_per_pos[chosen_idx]
                if pos in CONTENT_POS:
                    content_step += 1
                seed_q = self._main_q[chosen_idx].copy()
        return words

    def translate(self, query: str, raw_output: str,
                       style: str = "casual") -> str:
        """Post-hoc translator: take Aisha's raw output and replace each
        word with a same-POS, in-topic candidate that shifts the running
        cumulative trajectory toward B's reference curve.

        - For each word w_k at step k, compute cum_step_so_far and cum_R2.
        - Look up B's reference at the same step.
        - Score candidates of the same POS by:
            d² to current Kähler-seed (boundary k-NN style)
          + |new_step − B_ref_step| (penalize wrong step magnitude)
          + |new_R2 − B_ref_R2|     (penalize wrong curvature)
        - Pick the candidate that minimizes deviation from B's curve.
        """
        words = WORD_RE.findall(raw_output.lower())
        if len(words) < 2:
            return raw_output
        boundary = self.expand_content_boundary(query)
        ref_step = self._ref_B_step.get(style, self._ref_B_step["casual"])
        ref_R2   = self._ref_B_R2.get(style, self._ref_B_R2["casual"])

        # Walk through Aisha's words.  For each content word, try to
        # substitute toward the B-curve.  Function words pass through.
        out_words = []
        prev_q: np.ndarray | None = None
        cum_step = 0.0
        cum_R2 = 0.0
        content_step = 0
        for k, w in enumerate(words):
            old_i = self.wm.idx.get(w)
            if old_i is None:
                out_words.append(w); continue
            new_i = int(self._old_to_new[old_i])
            if new_i < 0:
                out_words.append(w); continue
            cur_pos = int(self.pos_arg[old_i])
            # only substitute content slots
            if cur_pos not in CONTENT_POS:
                out_words.append(w)
                if prev_q is not None:
                    cum_step += float(np.linalg.norm(
                        self._main_q[new_i] - prev_q))
                if self._R2_per_pos is not None:
                    cum_R2 += float(self._R2_per_pos[new_i])
                prev_q = self._main_q[new_i].copy()
                continue
            # candidate set: boundary intersected with same POS, plus content pool
            cands = [i for i in boundary
                        if self._strict_pos(i, cur_pos) and i != new_i]
            if not cands:
                cands = [i for i in self._content_pool.get(cur_pos, [])
                            if i != new_i]
            if not cands:
                out_words.append(w)
                continue
            # score each: 0.4*step_offcurve + 0.4*R2_offcurve + 0.2*Kähler-from-prev
            cands_arr = np.asarray(cands)
            new_q = self._main_q[cands_arr]
            # step offcurve
            if prev_q is not None:
                step_dist = np.linalg.norm(new_q - prev_q[None, :], axis=1)
            else:
                step_dist = np.full(len(cands), 25.0)
            new_cum_step = cum_step + step_dist
            ref_idx = min(content_step, 9)
            target_step = ref_step[ref_idx]
            step_off = (new_cum_step - target_step) ** 2
            # R² offcurve
            R2_w = (self._R2_per_pos[cands_arr] if self._R2_per_pos is not None
                       else np.zeros(len(cands)))
            new_cum_R2 = cum_R2 + R2_w
            target_R2 = ref_R2[ref_idx]
            R2_off = (np.log1p(new_cum_R2) - np.log1p(target_R2)) ** 2
            # Kähler — keep candidate close to current Kähler seed (the prev_q)
            if prev_q is not None:
                d2 = self.kahler.mahalanobis_to_seed(prev_q, new_q)
            else:
                d2 = np.zeros(len(cands))
            # Normalize each by its scale
            step_off /= max(step_off.max(), 1e-6)
            R2_off /= max(R2_off.max(), 1e-6)
            d2_n = d2 / max(d2.max(), 1e-6)
            score = 0.4 * step_off + 0.4 * R2_off + 0.2 * d2_n
            best = cands_arr[int(np.argmin(score))]
            new_word = self.kahler.lemmas[best]
            out_words.append(new_word)
            # update state
            if prev_q is not None:
                cum_step += float(np.linalg.norm(self._main_q[best] - prev_q))
            if self._R2_per_pos is not None:
                cum_R2 += float(self._R2_per_pos[best])
            prev_q = self._main_q[best].copy()
            content_step += 1
        # Capitalize first, terminate
        text = " ".join(out_words)
        if text:
            text = text[0].upper() + text[1:]
        if not text.endswith((".", "!", "?")):
            text += "."
        return text

    def respond(self, query: str, length: int | None = None,
                  rank_offset: int = 0) -> dict:
        t0 = time.time()
        # Harper polish on user input — fix typos / normalize before
        # the manifold processes it.
        if self.use_harper:
            try:
                pol_q = HARPER.polish(query, timeout=2.0)
                if pol_q and pol_q.strip():
                    query = pol_q
            except Exception:
                pass
        boundary = self.expand_content_boundary(query)
        if length is None:
            length = int(np.random.randint(4, 10))
        # Hand-coded CFG (build_grammar_template) constrains to S → NP VP
        # better than the bigram chain.  Keeping CFG for now.
        template = build_grammar_template(length, np.random)
        words = self.fill_template(template, query, boundary,
                                        rank_offset=rank_offset)

        for i in range(len(words) - 1):
            if words[i].lower() == "a":
                nxt = words[i + 1].lower()
                is_plural = (
                    (nxt.endswith("s") and not nxt.endswith(("ss","us","is")))
                    or nxt in {"children", "people", "men", "women",
                                  "feet", "teeth", "mice", "geese",
                                  "data", "media", "criteria", "phenomena"})
                if is_plural:
                    words[i] = "the"
                else:
                    words[i] = G.a_or_an(words[i + 1]) if _GRAMMAR else "a"

        if words:
            words[0] = words[0][:1].upper() + words[0][1:]
        text = " ".join(words)
        if text and not text.endswith((".", "!", "?")):
            text += "."

        if self.use_harper:
            try:
                pol = HARPER.polish(text, timeout=3.0)
                if pol and pol.strip():
                    text = pol
            except Exception:
                pass

        if self.memory is not None:
            user_idxs = [self.wm.idx[w] for w in WORD_RE.findall(query.lower())
                            if w in self.wm.idx]
            aisha_idxs = [self.wm.idx[w] for w in WORD_RE.findall(text.lower())
                            if w in self.wm.idx]
            self.memory.add_turn(user_word_idx=user_idxs,
                                       aisha_word_idx=aisha_idxs)

        return {"text": text,
                  "template": template,
                  "boundary_size": len(boundary),
                  "elapsed": time.time() - t0}
