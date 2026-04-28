"""
aisha_respond.py — v1 response generator.

Architecture: corpus-templated, manifold-scored, register-carrying.

Given A's sentence:
  1. Tokenize, compute A's mean (v, a) and content-topic q-vector,
     identify A's last-content octant.
  2. Infer style from A's (v, a) nearest to per-style centroid.
  3. Pick target register (v, a) = per-style mean.
  4. Sample B length from the style's empirical distribution.
  5. Build octant template:
       slot 0    : deictic/abstract opener
       slot 1-2  : first content octant drawn from A→B transition row
       middle    : deictic/abstract mix
       last ~30% : content octants (nominal closer)
  6. For each slot score 10k-mask words by:
        + octant match (hard filter)
        - (v - target_v)²            register-match
        - (a - target_a)²            register-match
        + cos(q_u[w], a_content_q)   topic grounding
        + torsion-alignment with previous token
      Sample top-k softmax(scores / τ).
  7. Emit lemma sequence.

Usage:
  python aisha_respond.py --demo 10
  python aisha_respond.py --sentence "I think we should talk about this."
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from word_manifold import WordManifold                    # noqa: E402
from corpus_deep import tokenize, OCTANT_SHORT, CONTENT_OCTANTS   # noqa: E402


_ALLOWED = re.compile(r"^[a-z]{1,15}(?:[-'][a-z]{1,8})?$")
STYLES = ["casual", "civilized", "emotional", "heated", "scientific"]

# Proper nouns / abbreviations that frequently leak through the automatic
# case-based detector because they are rare or absent in the conversations
# corpus.  These are common English names, places, abbreviations.
PROPER_BLACKLIST = frozenset("""
january february march april may june july august september october november december
monday tuesday wednesday thursday friday saturday sunday
england scotland wales ireland britain uk usa us america american canada canadian
france french germany german spain italy italian russia russian china chinese japan japanese
york london paris berlin moscow tokyo beijing chicago boston atlanta denver
texas california florida alaska nevada montana oregon arizona
africa european european asian australian
jack john james jim mary jane john robert michael david william richard thomas
charles tom bob bill dan daniel sarah anna alice emma olivia frank francis
charlie cooper jackson jefferson madison hamilton harrison wilson taylor
smith jones brown johnson davis miller lee king wright lopez garcia
catholic protestant christian jewish islamic buddhist hindu
fc nba nfl mlb nhl ipl tv dvd cd cpu gpu usa uk ok ceo cfo
republican democrat labour conservative
alex sam peter simon
""".split())

# Function/stopwords that sometimes land in content octants due to
# training noise.  We forbid them in content-slot candidate pools.
STOPWORDS = frozenset("""
a an the this that these those it its i me my mine you your yours he him his
she her hers we us our ours they them their theirs what which who whom whose
where when why how here there be is am are was were been being have has had
having do does did doing will would could should shall can may might must
of to in on at by for with from into onto over under about through across
and or but not no nor yes so if as than then although because while whether
now just only also too very well still even much many some any all each
more most less few other another own same such
let get got go come came went see saw look made make made take took taken
give gave given say said ask asked tell told think thought want wanted
need needed know knew known like liked feel felt felt put seem seems
way one two three four five six seven eight nine ten up down out in off
back about around above below across through between against along
than then now just only also too very well still even much many some any
i'm you're he's she's it's we're they're i've you've we've they've i'd you'd
he'd she'd we'd they'd i'll you'll he'll she'll we'll they'll don't doesn't
didn't isn't aren't wasn't weren't hasn't haven't hadn't won't wouldn't
can't couldn't shouldn't mustn't
""".split())

# Within-phrase SHAPE — corpus-measured mean of each axis at each
# within-phrase position, per phrase length.  Derived from layer2_word_order.
# Use these as position-dependent TARGETS so generated phrases respect
# the U-shape in M, the χ rise, the v/a arc.
PHRASE_SHAPE = {
    2: {"M":   [+2.30, +2.07],
         "chi": [-2.11, -1.92],
         "s":   [+0.17, +0.24],
         "v":   [-0.45, -0.33],
         "a":   [-0.66, -0.63]},
    3: {"M":   [+2.36, +1.82, +2.13],
         "chi": [-2.13, -2.24, -1.91],
         "s":   [+0.06, +0.21, +0.14],
         "v":   [-0.33, -0.20, -0.39],
         "a":   [-0.71, -0.44, -0.71]},
    4: {"M":   [+2.26, +1.90, +1.86, +2.08],
         "chi": [-2.11, -2.32, -2.24, -1.88],
         "s":   [+0.11, +0.18, +0.16, +0.13],
         "v":   [-0.28, -0.18, -0.22, -0.51],
         "a":   [-0.70, -0.48, -0.46, -0.66]},
    5: {"M":   [+2.44, +2.00, +1.97, +1.89, +2.27],
         "chi": [-2.18, -2.23, -2.25, -2.22, -1.91],
         "s":   [+0.09, +0.25, +0.25, +0.18, +0.05],
         "v":   [-0.34, -0.21, -0.20, -0.20, -0.45],
         "a":   [-0.73, -0.40, -0.46, -0.37, -0.70]},
}

# Phrase-to-phrase DECAY — phrase 2 (and later) shift toward lower M,
# lower spin, slightly lower valence.  From layer2_word_order 2-phrase
# sentence analysis: ΔM = -0.38, Δspin = -0.22, Δv = -0.15.
PHRASE_DECAY = {
    "M":   -0.40,    # each subsequent phrase has M target -0.40 lower
    "s":   -0.22,
    "v":   -0.15,
    "chi": +0.29,    # χ becomes less negative across phrases
    "a":   -0.05,
}


# Default scoring weights — tuned by optimize_weights.py 6D grid search
# against corpus physics-grammar targets.  Best config (loss=5.62):
#   shape_m=2.5  shape_chi=0.8  shape_s=0.5  pair=0.05  reg_v=3.5  reg_a=3.5
# 4/5 phrase-decay targets now within ±0.1 of corpus:
#   Δχ   +0.31 / +0.29 ✓
#   Δs   -0.26 / -0.22 ✓
#   Δv   -0.08 / -0.15 ✓
#   Δa   -0.01 / -0.05 ✓
#   ΔM   -0.07 / -0.38 (still under-expressed — need PHRASE_DECAY["M"] boost)
DEFAULT_WEIGHTS = {
    "reg_v":    3.5,     # was 1.0 — drives v decay correctly
    "reg_a":    3.5,     # was 1.0 — fixes a-decay sign
    "shape_m":  2.5,
    "shape_chi":0.8,
    "shape_s":  0.5,
    "topic":    1.0,
    "pair":     0.05,
    "freq":     0.6,
    "sw_bonus": 2.0,
    "reuse":    50.0,       # base scalar; multiplied by per-style reuse rate.
                            #   reg_v contributes up to ~35 to score magnitude,
                            #   so reuse needs to be on that order to compete.
    "pos_bigram": 3.0,      # per-slot log P(POS_cand | POS_prev) from corpus.
                            #   Reduces the need for POS-select's 40-sample
                            #   resampling by making every slot natively
                            #   produce a POS-likely next-word.
}


USE_CLEAN_MANIFOLD = True   # toggle: True → use rebuilt 26580-vocab manifold
USE_GRAMMAR_TEMPLATE = True # toggle: True → CFG-based POS templates instead
                            # of octant templates.  Hard rules out illegal
                            # POS sequences like "the the" or "VERB VERB".

if USE_GRAMMAR_TEMPLATE:
    from grammar_template import build_grammar_template


class Responder:
    def __init__(self, rng_seed: int = 0, weights: dict | None = None):
        self.score_weights = dict(DEFAULT_WEIGHTS)
        if weights: self.score_weights.update(weights)
        print("[aisha] loading manifold…", flush=True)

        if USE_CLEAN_MANIFOLD:
            manifold_path = ROOT / "data" / "processed" / "manifold_clean.pkl"
        else:
            manifold_path = ROOT / "data" / "processed" / "manifold.pkl"
        self.wm = WordManifold.load(str(manifold_path))
        self.rng = np.random.default_rng(rng_seed)

        if USE_CLEAN_MANIFOLD:
            # No training anywhere — drop the trained valence/arousal
            # score terms entirely.  Geometric structure (α, β, M, χ,
            # spin) is all analytical from corpus stats; BVP boundary
            # matrices provide the additional discrimination.
            self.score_weights["reg_v"] = 0.0
            self.score_weights["reg_a"] = 0.0
            # Boost stopword bonus — surface-form vocab has articles
            # ("the", "a") that need to win in function slots over
            # content-y aux verbs.  Default sw_bonus=2.0 too weak.
            self.score_weights["sw_bonus"] = 6.0
            # ---- Clean-manifold path ------------------------------
            # All four core axes ANALYTICAL from corpus stats — no
            # gradient training anywhere:
            #   α(N×8) from wm.alpha  (geometric formula on pos_counts)
            #   M(r)   from wm.m       (weighted token count)
            #   M(θ)   from q-coherence (computed against vocab)
            #   M(spin) from trigram asymmetry — log-ratio of
            #     "this word followed by content POS" vs
            #     "this word followed by function POS".
            #   captures whether a word LEADS INTO content (e.g.
            #   articles, prepositions: positive spin) or INTO
            #   function (e.g. nouns followed by preps: negative).
            alpha = self.wm.alpha.astype(np.complex128)   # (N, 8)
            q = self.wm.q.copy()
            M = self.wm.m.astype(np.float64)              # M(r)
            beta = self.wm.beta.copy()
            tg = self.wm.trigrams.astype(np.float64)
            out_content  = tg[:, :, 0:4].sum(axis=(1, 2))  # NOUN+VERB+ADJ+ADV
            out_function = tg[:, :, 4:].sum(axis=(1, 2))   # PRON+DET+PREP+OTHER
            spin = (np.log(out_content + 1) -
                    np.log(out_function + 1))               # M(spin)
        else:
            alpha = np.load(ROOT / "alpha_fixed.npz")["alpha"]
            mdat = np.load(ROOT / "m_fixed.npz")
            M = mdat["M"]; spin = mdat["spin"]
            H_til = mdat["H_til"]; beta = mdat["beta"]

        passers = np.array([bool(_ALLOWED.fullmatch(l)) for l in self.wm.lemmas])
        idx10k = np.where(passers)[0][np.argsort(-self.wm.m[passers])][:10000]

        # Cache directory — speeds startup after first run.
        cache_dir = ROOT / "data" / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        manifold_mtime = manifold_path.stat().st_mtime

        # ---- Proper noun set (cached) ----
        proper_cache = cache_dir / "proper_nouns.npz"
        if proper_cache.exists() and proper_cache.stat().st_mtime > manifold_mtime:
            d = np.load(proper_cache, allow_pickle=False)
            proper = set(int(i) for i in d["proper"])
            print(f"[aisha] proper nouns loaded from cache: {len(proper)}")
        else:
            proper = self._detect_proper_nouns()
            for lem in PROPER_BLACKLIST:
                if lem in self.wm.idx: proper.add(self.wm.idx[lem])
            np.savez(proper_cache, proper=np.array(sorted(proper), dtype=np.int32))
            print(f"[aisha] detected proper nouns: {len(proper)} (cached)")
        keep = np.array([i not in proper for i in idx10k])
        idx10k = idx10k[keep]

        self.mask10k = np.zeros(self.wm.N, dtype=bool); self.mask10k[idx10k] = True
        self.vocab10k = idx10k
        print(f"[aisha] vocab10k (after proper-noun filter): {len(idx10k)} words")

        if USE_CLEAN_MANIFOLD:
            # q already constructed from manifold seed
            q_u = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-30)
            self.q_u = q_u
            # v, a temporarily disabled — will be retrained on the
            # rebuilt vocab in Phase 2 follow-up.  Zeros mean reg_v
            # and reg_a contribute uniform constants and have no
            # discriminating effect.
            v_raw = np.zeros(self.wm.N, dtype=np.float64)
            a_raw = np.zeros(self.wm.N, dtype=np.float64)
        else:
            phase = (spin * beta)[:, None] * H_til
            q = alpha.astype(np.complex128) * np.exp(1j * phase)
            q_u = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-30)
            self.q_u = q_u
            # 5 normalised axes
            v_raw = np.load(ROOT / "valence.npz")["v"]
            a_raw = np.load(ROOT / "arousal.npz")["a"]

        # ---- chi (cached) ----
        chi_cache = cache_dir / "chi.npz"
        # Cache is mask-dependent.  Use deterministic blake2b on the
        # mask bytes (Python's built-in hash is salted per process).
        import hashlib
        mask_sig = int.from_bytes(
            hashlib.blake2b(self.mask10k.tobytes(), digest_size=8).digest(),
            "little", signed=False) & 0x7FFFFFFFFFFFFFFF
        if (chi_cache.exists()
                and chi_cache.stat().st_mtime > manifold_mtime):
            d = np.load(chi_cache, allow_pickle=False)
            if int(d["mask_sig"]) == mask_sig:
                chi = d["chi"]
                print(f"[aisha] chi loaded from cache")
            else:
                chi = (q_u.conj() @ q_u[self.mask10k].T).real.mean(axis=1)
                np.savez(chi_cache, chi=chi, mask_sig=np.int64(mask_sig))
                print(f"[aisha] chi recomputed (mask changed, cached)")
        else:
            chi = (q_u.conj() @ q_u[self.mask10k].T).real.mean(axis=1)
            np.savez(chi_cache, chi=chi, mask_sig=np.int64(mask_sig))
            print(f"[aisha] chi computed (cached)")

        def cn(x):
            c = x - np.median(x[self.mask10k])
            return c / max(np.std(c[self.mask10k]), 1e-9)

        self.M_n    = cn(M)
        self.chi_n  = cn(chi)
        self.spin_n = cn(spin)
        self.v      = cn(v_raw)
        self.a      = cn(a_raw)

        self.oct8 = (4 * (self.M_n   > 0).astype(int) +
                      2 * (self.chi_n > 0).astype(int) +
                          (self.spin_n > 0).astype(int))

        # Per-word "octant confidence" — L2 magnitude in (M, χ, s) space.
        # Borderline stopwords sit near zero; content words sit far out.
        self.oct_conf = np.sqrt(self.M_n**2 + self.chi_n**2 + self.spin_n**2)

        # Stopword index mask
        self.is_stopword = np.array(
            [lem in STOPWORDS for lem in self.wm.lemmas], dtype=bool)
        print(f"[aisha] stopwords in vocab: {int(self.is_stopword.sum())}")

        # =============== compute corpus stats ===============
        print("[aisha] loading corpus statistics…", flush=True)
        self._compute_corpus_stats()

        print(f"[aisha] ready")

    # ---------------------------------------------------------
    def _detect_proper_nouns(self) -> set[int]:
        """Use three independent signals to detect proper nouns:
          (1) corpus capitalisation statistics (mid-sentence uppercase)
          (2) WordNet: word has NO lowercase lemma in any synset → proper
          (3) WordNet: word has 0 synsets → rare / proper / technical
        Union of all three."""
        proper = set()

        # Signal 1: corpus case stats
        import csv as _csv
        cap = Counter(); low = Counter()
        sent_split = re.compile(r"[.!?]+")
        word_re = re.compile(r"[A-Za-z']+")
        with open(ROOT / "data" / "conversations.csv") as f:
            for row in _csv.DictReader(f):
                for sent in sent_split.split(row["text"]):
                    tokens = word_re.findall(sent)
                    for t in tokens[1:]:
                        tl = t.lower()
                        if tl not in self.wm.idx: continue
                        i = self.wm.idx[tl]
                        if t[0].isupper(): cap[i] += 1
                        else:               low[i] += 1
        for i in set(cap) | set(low):
            c, l = cap[i], low[i]
            if c + l < 3: continue
            if c / (c + l) > 0.6:
                proper.add(i)

        # Signals 2 + 3: WordNet analysis
        try:
            from nltk.corpus import wordnet as wn
        except ImportError:
            print("[aisha] nltk.wordnet not available — skipping wordnet filter")
            return proper

        # For efficiency, only check words in the top-20k vocab
        passers = np.array([bool(_ALLOWED.fullmatch(l)) for l in self.wm.lemmas])
        check_idx = np.where(passers)[0][np.argsort(-self.wm.m[passers])][:20000]

        wn_filtered = 0
        for i in check_idx:
            lem = self.wm.lemmas[i]
            if i in proper: continue
            synsets = wn.synsets(lem)
            if not synsets:
                # 0 synsets → rare/proper/technical.  Skip stopwords
                # (they aren't always in WordNet but we want to keep them).
                if lem not in STOPWORDS:
                    proper.add(i); wn_filtered += 1
                continue
            # Check if ANY synset has a lowercase lemma name matching this word
            has_common = False
            for s in synsets:
                for lm in s.lemmas():
                    name = lm.name()
                    if name.lower() == lem and name[0].islower():
                        has_common = True; break
                if has_common: break
            if not has_common:
                proper.add(i); wn_filtered += 1

        print(f"[aisha] proper-noun detection: wordnet added {wn_filtered}")
        # Stopwords are NEVER proper nouns — `i`, `is`, `was`, `i'm`
        # get falsely flagged by capitalization (always written upper)
        # or absent from WordNet (inflected forms).  Hard-exclude.
        sw_excluded = 0
        for w in STOPWORDS:
            i = self.wm.idx.get(w)
            if i is not None and i in proper:
                proper.discard(i)
                sw_excluded += 1
        if sw_excluded:
            print(f"[aisha] proper-noun detection: stopword exclusion "
                  f"removed {sw_excluded}")
        return proper

    def _compute_corpus_stats(self):
        import hashlib
        cache_dir = ROOT / "data" / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        corpus_path = ROOT / "data" / "conversations.csv"
        corpus_mtime = corpus_path.stat().st_mtime
        mask_sig = int.from_bytes(
            hashlib.blake2b(self.mask10k.tobytes(), digest_size=8).digest(),
            "little", signed=False) & 0x7FFFFFFFFFFFFFFF

        stats_cache = cache_dir / "corpus_stats.npz"
        if stats_cache.exists() and stats_cache.stat().st_mtime > corpus_mtime:
            try:
                d = np.load(stats_cache, allow_pickle=True)
                if int(d["mask_sig"]) == mask_sig:
                    self.style_v    = dict(d["style_v"].item())
                    self.style_a    = dict(d["style_a"].item())
                    self.style_lens = {k: list(v) for k, v in d["style_lens"].item().items()}
                    self.style_reuse= dict(d["style_reuse"].item())
                    self.transition = d["transition"]
                    self.pos_gradient = {k: np.asarray(v)
                                            for k, v in d["pos_gradient"].item().items()}
                    self.pos_log_bigram = d["pos_log_bigram"]
                    self.word_pos = d["word_pos"]
                    # Per-style POS bigrams (added in Stage 2.5) — optional;
                    # absent in pre-2.5 caches, in which case set_style()
                    # silently falls back to the corpus-wide matrix.
                    if "pos_log_bigram_by_style" in d.files:
                        self.pos_log_bigram_by_style = d["pos_log_bigram_by_style"].item()
                    print(f"[aisha] corpus stats loaded from cache")
                    return
            except Exception:
                pass  # fall through to rebuild

        rows = []
        with open(corpus_path) as f:
            for row in csv.DictReader(f):
                idx_all = tokenize(row["text"], self.wm)
                idx = [i for i in idx_all if self.mask10k[i]]
                if not idx: continue
                rows.append({
                    "conv_id": int(row["conv_id"]),
                    "turn_id": int(row["turn_id"]),
                    "speaker": row["speaker"],
                    "style":   row["style"],
                    "idx":     idx,
                })

        # Per-style v/a centroids (target registers)
        self.style_v = {st: float(np.mean([self.v[r["idx"]].mean()
                                             for r in rows if r["style"] == st]))
                         for st in STYLES}
        self.style_a = {st: float(np.mean([self.a[r["idx"]].mean()
                                             for r in rows if r["style"] == st]))
                         for st in STYLES}

        # Per-style sentence length distribution
        self.style_lens = {st: [len(r["idx"]) for r in rows if r["style"] == st]
                            for st in STYLES}

        # A→B first-content octant transition (rows normalised)
        by_conv = defaultdict(list)
        for r in rows: by_conv[r["conv_id"]].append(r)
        for cid in by_conv: by_conv[cid].sort(key=lambda x: x["turn_id"])

        # Per-style lexical reuse rate: fraction of B-content lemmas that
        # also appeared in A's previous turn.  Real English dialog has 0-13%
        # reuse depending on style.  Aisha matches this per-style.
        reuse_per_style: dict[str, list[float]] = {st: [] for st in STYLES}
        for cid, turns in by_conv.items():
            for i in range(len(turns) - 1):
                a, b = turns[i], turns[i + 1]
                if a["speaker"] != "A" or b["speaker"] != "B": continue
                a_set = {w for w in a["idx"]
                          if int(self.oct8[w]) in CONTENT_OCTANTS}
                b_content = [w for w in b["idx"]
                              if int(self.oct8[w]) in CONTENT_OCTANTS]
                if not b_content: continue
                reuse_per_style[a["style"]].append(
                    sum(1 for w in b_content if w in a_set) / len(b_content))
        self.style_reuse = {st: (float(np.mean(v)) if v else 0.0)
                              for st, v in reuse_per_style.items()}
        print(f"[aisha] per-style lexical reuse rate: "
              f"{ {k:f'{v:.1%}' for k, v in self.style_reuse.items()} }")

        # POS-bigram transition matrix from corpus.  Per-slot scoring
        # rewards candidates whose POS continues the previous word's
        # POS naturally.  Uses wm.pi (POS simplex, 8-dim).
        #
        # Built per-style as well as corpus-wide.  set_style() swaps the
        # active matrix so generation produces style-distinctive POS
        # sequencing (e.g. scientific likes Adj→Noun, casual likes Pro→Verb).
        pos_counts = np.zeros((8, 8), dtype=np.float64) + 0.5
        pos_counts_by_style: dict[str, np.ndarray] = {
            st: np.zeros((8, 8), dtype=np.float64) + 0.5 for st in STYLES}
        for r in rows:
            if len(r["idx"]) < 2: continue
            pos_seq = [int(self.wm.pi[t].argmax()) for t in r["idx"]]
            for a, b in zip(pos_seq[:-1], pos_seq[1:]):
                pos_counts[a, b] += 1
                if r["style"] in pos_counts_by_style:
                    pos_counts_by_style[r["style"]][a, b] += 1
        self.pos_log_bigram = np.log(
            pos_counts / pos_counts.sum(axis=1, keepdims=True))
        self.pos_log_bigram_by_style = {
            st: np.log(c / c.sum(axis=1, keepdims=True))
            for st, c in pos_counts_by_style.items()}
        # active matrix — swappable via set_style()
        self._pos_log_bigram_corpus = self.pos_log_bigram   # archival default
        self.active_style: str | None = None
        # pre-extract per-word POS argmax for fast lookup in scoring
        self.word_pos = self.wm.pi.argmax(axis=1).astype(np.int64)
        print(f"[aisha] POS bigram built  corpus mean logP="
              f"{self.pos_log_bigram.mean():.2f}; per-style available: "
              f"{list(self.pos_log_bigram_by_style)}")

        trans = np.zeros((8, 8), dtype=float)
        for cid, turns in by_conv.items():
            for i in range(len(turns) - 1):
                a, b = turns[i], turns[i + 1]
                if a["speaker"] != "A" or b["speaker"] != "B": continue
                al = self._last_content(a["idx"])
                bf = self._first_content(b["idx"])
                if al is not None and bf is not None:
                    trans[al, bf] += 1
        self.transition = trans / np.maximum(trans.sum(1, keepdims=True), 1)
        # Where a row is empty (no pairs), fall back to uniform content
        for a in range(8):
            if trans[a].sum() == 0:
                for k in CONTENT_OCTANTS:
                    self.transition[a, k] = 1.0 / len(CONTENT_OCTANTS)

        # Per-style position gradient: P(content octant) at each
        # relative sentence position (0-1).  Used to shape template.
        POS_BINS = 5
        self.pos_gradient = {st: np.zeros((POS_BINS, 8), dtype=float)
                              for st in STYLES}
        counts = {st: np.zeros(POS_BINS, dtype=float) for st in STYLES}
        for r in rows:
            if r["speaker"] != "B": continue
            L = len(r["idx"])
            if L < 2: continue
            for k, i in enumerate(r["idx"]):
                rel = min(int(k / max(L - 1, 1) * POS_BINS), POS_BINS - 1)
                self.pos_gradient[r["style"]][rel, self.oct8[i]] += 1
                counts[r["style"]][rel] += 1
        for st in STYLES:
            self.pos_gradient[st] /= np.maximum(counts[st][:, None], 1)

        # Save to cache
        try:
            np.savez(
                stats_cache,
                mask_sig=np.int64(mask_sig),
                style_v=self.style_v,
                style_a=self.style_a,
                style_lens=self.style_lens,
                style_reuse=self.style_reuse,
                transition=self.transition,
                pos_gradient=self.pos_gradient,
                pos_log_bigram=self.pos_log_bigram,
                pos_log_bigram_by_style=self.pos_log_bigram_by_style,
                word_pos=self.word_pos,
            )
            print(f"[aisha] corpus stats cached")
        except Exception as e:
            print(f"[aisha] corpus stats cache save failed: {e}")

    # ------------------------------------------------------------------
    def set_memory_perturbation(self, dv: float, da: float) -> None:
        """Bias next response's (v, a) target by (dv, da).  Persists across
        the n_samples loop in PosSelector.respond — Pipeline clears it
        explicitly after the full generation completes."""
        self._mem_dv = float(dv)
        self._mem_da = float(da)

    def consume_memory_perturbation(self) -> tuple[float, float]:
        """Return current (dv, da) WITHOUT resetting, so every candidate
        in PosSelector's n_samples loop gets the same perturbation."""
        return (getattr(self, "_mem_dv", 0.0),
                getattr(self, "_mem_da", 0.0))

    # ------------------------------------------------------------------
    def set_style(self, style: str | None) -> None:
        """Activate a per-style POS bigram matrix for scoring.

        Per-style matrices may be missing if the corpus_stats cache pre-dates
        the per-style build — in that case this is a no-op (silent fallback).
        """
        by_style = getattr(self, "pos_log_bigram_by_style", None)
        if not by_style:
            return
        # Lazily snapshot the corpus-wide matrix on first style activation.
        if not hasattr(self, "_pos_log_bigram_corpus"):
            self._pos_log_bigram_corpus = self.pos_log_bigram
        if style is not None and style in by_style:
            self.pos_log_bigram = by_style[style]
            self.active_style = style
        else:
            self.pos_log_bigram = self._pos_log_bigram_corpus
            self.active_style = None

    def _first_content(self, idx_seq):
        for i in idx_seq:
            if int(self.oct8[i]) in CONTENT_OCTANTS: return int(self.oct8[i])
        return None
    def _last_content(self, idx_seq):
        for i in reversed(idx_seq):
            if int(self.oct8[i]) in CONTENT_OCTANTS: return int(self.oct8[i])
        return None
    def _last_content_idx(self, idx_seq):
        for i in reversed(idx_seq):
            if int(self.oct8[i]) in CONTENT_OCTANTS: return i
        return None

    # ---------------------------------------------------------
    def _infer_style(self, v_mean: float, a_mean: float) -> str:
        best = min(STYLES, key=lambda st:
                    (v_mean - self.style_v[st])**2 + (a_mean - self.style_a[st])**2)
        return best

    def _sample_length(self, style: str) -> int:
        lens = self.style_lens[style]
        if not lens: return 10
        # Floor at corpus per-style 20th percentile — avoid 3-4 word
        # responses that can't carry any real structure.
        floor = int(max(6, np.percentile(lens, 20)))
        L = int(self.rng.choice(lens))
        return max(L, floor)

    def _segment_phrase_lens(self, total_length: int) -> list[int]:
        """Divide total_length into 2-3 phrase lengths of ~3-4 each.
        Matches the corpus distribution of phrase sizes."""
        if total_length <= 4:
            return [total_length]
        elif total_length <= 7:
            p1 = total_length // 2
            return [p1, total_length - p1]
        else:
            p1 = total_length // 3
            p2 = total_length // 3
            return [p1, p2, total_length - p1 - p2]

    def _build_template(self, length: int, a_last_oct: int | None,
                         style: str) -> list[int]:
        """Pick a target octant for each B slot."""
        # First content octant: sample from transition row if available
        if a_last_oct is not None:
            row = self.transition[a_last_oct].copy()
        else:
            row = np.zeros(8)
            for k in CONTENT_OCTANTS: row[k] = 1.0 / len(CONTENT_OCTANTS)
        # Restrict to content octants when choosing "content" slots
        content_row = np.zeros(8)
        for k in CONTENT_OCTANTS: content_row[k] = row[k]
        content_row = content_row / max(content_row.sum(), 1e-9)
        first_content_oct = int(self.rng.choice(8, p=content_row))

        POS_BINS = self.pos_gradient[style].shape[0]
        template = []
        content_slots_used = 0
        for k in range(length):
            rel = min(int(k / max(length - 1, 1) * POS_BINS), POS_BINS - 1)
            pos_probs = self.pos_gradient[style][rel].copy()
            # Inject the transition bias for the first content slot
            if content_slots_used == 0 and \
                self.oct8[0] is not None:      # always true; placeholder
                # Reinforce first_content_oct if it's likely at this position
                pos_probs = 0.5 * pos_probs + 0.5 * \
                    np.eye(8)[first_content_oct] * pos_probs.sum()
            pos_probs = pos_probs / max(pos_probs.sum(), 1e-9)
            chosen = int(self.rng.choice(8, p=pos_probs))
            template.append(chosen)
            if chosen in CONTENT_OCTANTS:
                content_slots_used += 1
        return template

    def _position_targets(self, within_pos: int, phrase_len: int,
                            phrase_idx: int,
                            base_v: float, base_a: float) -> dict:
        """Return position-adjusted axis targets for a slot inside a phrase.
        Uses PHRASE_SHAPE (within-phrase arc) + PHRASE_DECAY (phrase-to-phrase)."""
        L = phrase_len
        if L not in PHRASE_SHAPE:
            # Fall back to closest length
            L = min(PHRASE_SHAPE, key=lambda x: abs(x - phrase_len))
        p = min(within_pos, len(PHRASE_SHAPE[L]["M"]) - 1)

        shape = PHRASE_SHAPE[L]
        # Within-phrase deviation from phrase mean
        def dev(axis):
            arr = np.asarray(shape[axis])
            return float(arr[p] - arr.mean())

        v_dev = dev("v"); a_dev = dev("a")
        # Apply phrase-to-phrase decay
        decay_m   = PHRASE_DECAY["M"]   * phrase_idx
        decay_chi = PHRASE_DECAY["chi"] * phrase_idx
        decay_s   = PHRASE_DECAY["s"]   * phrase_idx
        decay_v   = PHRASE_DECAY["v"]   * phrase_idx
        decay_a   = PHRASE_DECAY["a"]   * phrase_idx

        return {
            "target_v":   base_v + v_dev + decay_v,
            "target_a":   base_a + a_dev + decay_a,
            "target_M":   shape["M"][p]   + decay_m,
            "target_chi": shape["chi"][p] + decay_chi,
            "target_s":   shape["s"][p]   + decay_s,
        }

    def _score_candidates(self, target_oct: int,
                           target_v: float, target_a: float,
                           a_topic_q: np.ndarray,
                           prev_idx: int | None,
                           used_idx: set,
                           phrase_info: tuple | None = None,
                           a_content_set: frozenset | None = None,
                           reuse_strength: float = 0.0,
                           ) -> tuple[np.ndarray, np.ndarray]:
        """Return (cand_idx, score) for words at target.

        When USE_GRAMMAR_TEMPLATE: target_oct is a POS channel (0-7).
        Otherwise: target_oct is a manifold octant (0-7).
        """
        if USE_GRAMMAR_TEMPLATE:
            # POS-channel filter (grammar mode)
            mask = self.mask10k & (self.word_pos == target_oct)
            CONTENT_POS = {0, 1, 2, 3}
            is_content_slot = target_oct in CONTENT_POS
            if is_content_slot:
                base_mask = mask & (~self.is_stopword)
                if a_content_set:
                    rescue = np.zeros(self.wm.N, dtype=bool)
                    for w in a_content_set:
                        if (int(self.word_pos[w]) == target_oct and
                                self.mask10k[w] and
                                not self.is_stopword[w]):
                            rescue[w] = True
                    mask = base_mask | rescue
                else:
                    mask = base_mask
            else:
                mask = mask & (self.is_stopword | (self.oct_conf < 1.3))
        else:
            # Octant filter (legacy)
            mask = self.mask10k & (self.oct8 == target_oct)
            is_content_slot = target_oct in CONTENT_OCTANTS
            if is_content_slot:
                base_mask = mask & (self.oct_conf > 0.7) & (~self.is_stopword)
                if a_content_set:
                    rescue = np.zeros(self.wm.N, dtype=bool)
                    for w in a_content_set:
                        if (int(self.oct8[w]) == target_oct and
                                self.mask10k[w] and
                                not self.is_stopword[w]):
                            rescue[w] = True
                    mask = base_mask | rescue
                else:
                    mask = base_mask
            else:
                mask = mask & (self.is_stopword | (self.oct_conf < 1.3))

        if used_idx:
            exclude = np.zeros(self.wm.N, dtype=bool)
            for u in used_idx: exclude[u] = True
            mask = mask & (~exclude)
        cand_idx = np.where(mask)[0]
        if cand_idx.size == 0:
            return np.array([]), np.array([])

        # Position-aware targets — the physics-grammar kicks in here
        if phrase_info is not None:
            wp, pl, pi = phrase_info
            pt = self._position_targets(wp, pl, pi, target_v, target_a)
            eff_v = pt["target_v"]; eff_a = pt["target_a"]
            m_tgt = pt["target_M"]; chi_tgt = pt["target_chi"]; s_tgt = pt["target_s"]
        else:
            eff_v = target_v; eff_a = target_a
            m_tgt = chi_tgt = s_tgt = None

        reg_v = -(self.v[cand_idx] - eff_v) ** 2
        reg_a = -(self.a[cand_idx] - eff_a) ** 2

        # Shape penalties: how well does this candidate's M / χ / spin
        # match what the within-phrase arc wants at this position?
        if m_tgt is not None:
            shape_m   = -(self.M_n[cand_idx]   - m_tgt)   ** 2
            shape_chi = -(self.chi_n[cand_idx] - chi_tgt) ** 2
            shape_s   = -(self.spin_n[cand_idx] - s_tgt)  ** 2
        else:
            shape_m = shape_chi = shape_s = np.zeros(len(cand_idx))

        q_cand = self.q_u[cand_idx]
        topic = (q_cand.conj() * a_topic_q[None, :]).real.sum(axis=1)

        if prev_idx is not None:
            q_prev = self.q_u[prev_idx]
            pair = (q_cand.conj() * q_prev[None, :]).real.sum(axis=1)
            # Active POS bigram: log P(POS_cand | POS_prev).
            prev_pos = int(self.word_pos[prev_idx])
            cand_pos = self.word_pos[cand_idx]
            pos_bigram = self.pos_log_bigram[prev_pos, cand_pos]
        else:
            pair = np.zeros_like(topic)
            pos_bigram = np.zeros(len(cand_idx), dtype=np.float32)

        freq_prior = np.log(np.maximum(self.wm.m[cand_idx], 1e-9))
        freq_prior = freq_prior - freq_prior.mean()

        w = self.score_weights
        # Use is_content_slot computed above (works for both octant
        # and POS-channel modes).
        if not is_content_slot:
            sw_bonus = self.is_stopword[cand_idx].astype(np.float32) * w["sw_bonus"]
        else:
            sw_bonus = np.zeros(len(cand_idx), dtype=np.float32)

        # Lexical-reuse bonus: reward picking A's own content words.
        if (is_content_slot and a_content_set
                and reuse_strength > 0):
            reuse_flag = np.array(
                [1.0 if i in a_content_set else 0.0 for i in cand_idx],
                dtype=np.float32)
            reuse_bonus = reuse_strength * reuse_flag
        else:
            reuse_bonus = np.zeros(len(cand_idx), dtype=np.float32)

        # Topic q-coherence and shape penalties were designed for
        # content slots — function slots have extreme M_n/χ_n that
        # get wiped out by shape_m squared penalties.
        if is_content_slot:
            topic_weight = w["topic"]
            shape_m_w = w["shape_m"]
            shape_chi_w = w["shape_chi"]
            shape_s_w = w["shape_s"]
        else:
            topic_weight = 0.0
            shape_m_w = 0.0
            shape_chi_w = 0.0
            shape_s_w = 0.0

        score = (w["reg_v"]      * reg_v +
                  w["reg_a"]      * reg_a +
                  shape_m_w       * shape_m +
                  shape_chi_w     * shape_chi +
                  shape_s_w       * shape_s +
                  topic_weight    * topic +
                  w["pair"]       * pair +
                  w["freq"]       * freq_prior +
                  w["pos_bigram"] * pos_bigram +
                  sw_bonus +
                  reuse_bonus)
        return cand_idx, score

    def _sample(self, idx_arr, score_arr, temperature=0.6, top_k=20):
        if idx_arr.size == 0: return None
        k = min(top_k, idx_arr.size)
        top = np.argpartition(-score_arr, k - 1)[:k]
        s = score_arr[top]
        s = (s - s.max()) / max(temperature, 1e-3)
        p = np.exp(s); p /= p.sum()
        choice = int(self.rng.choice(idx_arr[top], p=p))
        return choice

    # ---------------------------------------------------------
    def score_substitutes(self,
                            prev_word_idx: int | None,
                            target_word_idx: int,
                            pos_constraint: int | None = None,
                            attractor_strength: float = 2.0,
                            top_k: int = 50,
                            exclude_indices: set | None = None,
                            ) -> list:
        """Two-anchor scoring step — analytical-jump path mode.

        Score vocab words as substitutes that:
          - are pulled toward target_word_idx via q-vector attractor
            (the same q-coherence the topic term uses, but anchored on
            the target instead of the user-query topic)
          - pair grammatically with prev_word_idx (existing pair-torsion
            and POS-bigram logic)
          - optionally restricted to a specific POS class (so the swap
            preserves grammaticality)

        Returns top-k (word_idx, score) tuples sorted by score desc.

        This method does NOT modify any existing scoring logic.  It
        invokes one additive scoring step using the same q_u manifold
        and pos_log_bigram tables, with two anchors instead of one.
        The slot-filling iteration of `respond()` is untouched.
        """
        cand_mask = self.mask10k & (~self.is_stopword)
        cand_idx = np.where(cand_mask)[0]
        if pos_constraint is not None:
            cand_idx = cand_idx[self.word_pos[cand_idx] == int(pos_constraint)]
        if exclude_indices:
            cand_idx = cand_idx[~np.isin(cand_idx, list(exclude_indices))]
        if cand_idx.size == 0:
            return []

        q_cand = self.q_u[cand_idx]

        # Path attractor: real-part inner product with target's q-vector
        q_target = self.q_u[int(target_word_idx)]
        attractor = (q_cand.conj() * q_target[None, :]).real.sum(axis=1)

        # Pair-torsion coherence with prev (preserves local grammar)
        if prev_word_idx is not None:
            q_prev = self.q_u[int(prev_word_idx)]
            pair = (q_cand.conj() * q_prev[None, :]).real.sum(axis=1)
            prev_pos = int(self.word_pos[int(prev_word_idx)])
            cand_pos = self.word_pos[cand_idx]
            pos_bigram = self.pos_log_bigram[prev_pos, cand_pos]
        else:
            pair = np.zeros_like(attractor)
            pos_bigram = np.zeros(len(cand_idx))

        # No frequency boost or penalty for substitution — let the
        # attractor + pair coherence + POS-bigram drive the pick on
        # their own.  We only floor truly-obscure words so we don't
        # land on foreign-corpus artifacts.
        log_freq = np.log(np.maximum(self.wm.m[cand_idx], 1e-9))
        too_rare = log_freq < np.log(20.0)

        w = self.score_weights
        score = (attractor_strength * attractor +
                  w["pair"]      * pair +
                  w["pos_bigram"] * pos_bigram)
        score[too_rare] = -1e9

        k = min(top_k, len(cand_idx))
        top = np.argpartition(-score, k - 1)[:k]
        top = top[np.argsort(-score[top])]
        return [(int(cand_idx[i]), float(score[i])) for i in top]

    # ---------------------------------------------------------
    def respond(self, a_text: str, style: str | None = None,
                  verbose: bool = False) -> dict:
        idx_all = tokenize(a_text, self.wm)
        idx = [i for i in idx_all if self.mask10k[i]]
        if not idx:
            return {"text": "", "b_idx": [], "style": "?",
                     "a_mean_v": 0.0, "a_mean_a": 0.0, "reason": "no tokens"}

        a_v = float(self.v[idx].mean())
        a_a = float(self.a[idx].mean())
        inferred_style = style or self._infer_style(a_v, a_a)

        # Topic q-vector — A's LAST content word is the most focused anchor
        # (see carry_structure: A's end-signal predicts B's content well).
        content_idx = [i for i in idx if int(self.oct8[i]) in CONTENT_OCTANTS]
        if content_idx:
            a_topic_q = self.q_u[content_idx[-1]].copy()
        else:
            a_topic_q = np.zeros(self.q_u.shape[1], dtype=np.complex128)

        # Target register — slight amplification toward style centroid,
        # plus an optional memory perturbation that pulls toward the
        # conversation's recent (v, a) trajectory.
        target_v = 0.3 * a_v + 0.7 * self.style_v[inferred_style]
        target_a = 0.3 * a_a + 0.7 * self.style_a[inferred_style]
        mem_dv, mem_da = self.consume_memory_perturbation()
        if mem_dv or mem_da:
            target_v += mem_dv
            target_a += mem_da
            if verbose:
                print(f"  memory perturbation: dv={mem_dv:+.2f}  da={mem_da:+.2f}")

        # B length and template + phrase segmentation
        length = max(4, min(20, self._sample_length(inferred_style)))
        a_last_oct = self._last_content(idx)
        if USE_GRAMMAR_TEMPLATE:
            # Strict CFG POS sequence — guarantees no "the the", no
            # adjacent VERB-VERB without aux, etc.  Slot integer is
            # interpreted as POS channel (0-7), not octant.
            template = build_grammar_template(length, self.rng)
        else:
            template = self._build_template(length, a_last_oct, inferred_style)

        # Segment into phrases → slot → (within_pos, phrase_len, phrase_idx)
        phrase_lens = self._segment_phrase_lens(length)
        slot_info = {}
        s = 0
        for pi, pl in enumerate(phrase_lens):
            for wp in range(pl):
                slot_info[s] = (wp, pl, pi)
                s += 1

        if verbose:
            print(f"  style: {inferred_style}   A_mean(v)={a_v:+.2f}  "
                  f"A_mean(a)={a_a:+.2f}")
            print(f"  target: v={target_v:+.2f}  a={target_a:+.2f}  "
                  f"length={length}  phrases={phrase_lens}")
            print(f"  A_last_oct = {OCTANT_SHORT[a_last_oct] if a_last_oct is not None else '—'}")
            print(f"  template octants: {[OCTANT_SHORT[o] for o in template]}")

        # Fill slots.  used = set() — Aisha won't repeat her OWN words,
        # but A's content words ARE allowed (real dialog reuses them 3-13%).
        b_idx = []
        used = set()
        a_content_set = frozenset(content_idx)
        reuse_rate = self.style_reuse.get(inferred_style, 0.0)
        reuse_strength = self.score_weights["reuse"] * reuse_rate

        # Pre-bucket A's content words.  In grammar mode, bucket by
        # POS channel; in legacy mode, by octant.
        a_by_bucket: dict[int, list[int]] = {}
        for w in a_content_set:
            if USE_GRAMMAR_TEMPLATE:
                key = int(self.word_pos[w])
            else:
                key = int(self.oct8[w])
            a_by_bucket.setdefault(key, []).append(w)

        prev = None
        for slot, target_oct in enumerate(template):
            phrase_info = slot_info.get(slot)

            # Forced reuse path: with probability = style_reuse_rate,
            # if A has a content word at this slot's bucket and we
            # haven't used it, take it directly.
            chosen = None
            if USE_GRAMMAR_TEMPLATE:
                CONTENT_BUCKET = {0, 1, 2, 3}
            else:
                CONTENT_BUCKET = CONTENT_OCTANTS
            if (target_oct in CONTENT_BUCKET
                    and target_oct in a_by_bucket
                    and self.rng.random() < reuse_rate):
                candidates_a = [w for w in a_by_bucket[target_oct]
                                 if w not in used]
                if candidates_a:
                    chosen = int(self.rng.choice(candidates_a))

            if chosen is None:
                cand_idx, scores = self._score_candidates(
                    target_oct=target_oct,
                    target_v=target_v, target_a=target_a,
                    a_topic_q=a_topic_q,
                    prev_idx=prev, used_idx=used,
                    phrase_info=phrase_info,
                    a_content_set=a_content_set,
                    reuse_strength=reuse_strength)
                chosen = self._sample(cand_idx, scores, temperature=0.6, top_k=20)
            if chosen is None: continue
            b_idx.append(chosen)
            used.add(chosen)
            prev = chosen

        words = [self.wm.lemmas[i] for i in b_idx]
        if words:
            words[0] = words[0].capitalize()
            text = " ".join(words) + "."
        else:
            text = ""
        return {
            "text": text, "b_idx": b_idx, "style": inferred_style,
            "a_mean_v": a_v, "a_mean_a": a_a,
            "target_v": target_v, "target_a": target_a,
            "template": template,
        }


# ======================================================================
def cmd_sentence(args) -> None:
    r = Responder(rng_seed=args.seed)
    out = r.respond(args.sentence, style=args.style, verbose=True)
    print()
    print("A>  " + args.sentence)
    print(f"B>  {out['text']}")
    print(f"    (style={out['style']}  target v={out['target_v']:+.2f}  "
          f"a={out['target_a']:+.2f})")


def cmd_demo(args) -> None:
    r = Responder(rng_seed=args.seed)
    # Sample N A→B pairs from the corpus
    rows = []
    with open(ROOT / "data" / "conversations.csv") as f:
        for row in csv.DictReader(f):
            rows.append(row)

    by_conv = defaultdict(list)
    for row in rows: by_conv[int(row["conv_id"])].append(row)
    for cid in by_conv:
        by_conv[cid].sort(key=lambda x: int(x["turn_id"]))

    # Pick args.demo diverse A→B pairs across styles
    pairs_per_style: dict[str, list] = {st: [] for st in STYLES}
    for cid, turns in by_conv.items():
        for i in range(len(turns) - 1):
            a, b = turns[i], turns[i + 1]
            if a["speaker"] != "A" or b["speaker"] != "B": continue
            pairs_per_style[a["style"]].append((a["text"], b["text"]))

    n_each = max(1, args.demo // len(STYLES))
    picks = []
    for st in STYLES:
        plist = pairs_per_style[st]
        k = min(n_each, len(plist))
        if k == 0: continue
        idxs = r.rng.choice(len(plist), size=k, replace=False)
        for j in idxs: picks.append((st, plist[j][0], plist[j][1]))

    print()
    print("=" * 90)
    print(f"Aisha demo — {len(picks)} A→B pairs  "
          f"(3 samples each; corpus-labeled style forced)")
    print("=" * 90)
    for style, a_text, b_actual in picks:
        print()
        print(f"[{style}]")
        print(f"  A>       {a_text}")
        for k in range(3):
            # Use k-th sample seed to get diversity
            r.rng = np.random.default_rng(args.seed + k * 17 + hash(a_text) % 1000)
            out = r.respond(a_text, style=style)
            tag = f"Aisha{k+1}>"
            print(f"  {tag:<9s}{out['text']}")
        print(f"  B>       {b_actual}")
        print(f"    target v={out['target_v']:+.2f}  "
              f"a={out['target_a']:+.2f}  octants={[OCTANT_SHORT[o] for o in out['template']]}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sentence", type=str, default=None)
    ap.add_argument("--style", type=str, default=None, choices=STYLES)
    ap.add_argument("--demo", type=int, default=0, help="sample N corpus A→B pairs and show generations")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.sentence:
        cmd_sentence(args)
    elif args.demo > 0:
        cmd_demo(args)
    else:
        # Default: demo 10
        args.demo = 10; cmd_demo(args)


if __name__ == "__main__":
    main()
