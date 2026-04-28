"""Corpus-fitted phrase pool for multi-word emission.

Builds bigram and trigram phrase pools from conversations.csv, indexed
by POS pattern.  Used by the responder to emit short phrases like
"a good idea", "in the morning", "of the day" instead of word-by-word
slot fill where the latter produces seams.

Pool structure:
  bigrams[pos_pattern] = list of (word_a, word_b, freq)
  trigrams[pos_pattern] = list of (word_a, word_b, word_c, freq)

Only patterns appearing > MIN_FREQ times are kept.
"""
import csv
import re
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
CSV = ROOT / "data" / "conversations.csv"
CACHE = ROOT / "data" / "processed" / "phrase_pool.npz"
WORD_RE = re.compile(r"[a-zA-Z']+")
MIN_FREQ_BIGRAM = 3
MIN_FREQ_TRIGRAM = 2


class PhrasePool:
    def __init__(self, wm):
        self.wm = wm
        self.pos_arg = wm.pi.argmax(axis=1)
        if CACHE.exists():
            self._load_from_cache()
        else:
            print("[phrase] building from conversations.csv …", flush=True)
            self._build()
            self._save_to_cache()

    def _build(self):
        bigram_counter: Counter = Counter()
        trigram_counter: Counter = Counter()
        with open(CSV) as f:
            for row in csv.DictReader(f):
                tokens = WORD_RE.findall(row["text"].lower())
                # collect (pos_seq, word_seq) for the sentence
                idxs = [self.wm.idx.get(w) for w in tokens]
                pos_seq = [int(self.pos_arg[i]) if i is not None else -1
                              for i in idxs]
                wseq = [w if i is not None else None
                          for w, i in zip(tokens, idxs)]
                # bigrams
                for k in range(len(wseq) - 1):
                    if wseq[k] is None or wseq[k+1] is None:
                        continue
                    pair = (pos_seq[k], pos_seq[k+1])
                    bigram_counter[(pair, wseq[k], wseq[k+1])] += 1
                # trigrams
                for k in range(len(wseq) - 2):
                    if any(wseq[k+j] is None for j in range(3)):
                        continue
                    triple = (pos_seq[k], pos_seq[k+1], pos_seq[k+2])
                    trigram_counter[(triple, wseq[k], wseq[k+1], wseq[k+2])] += 1
        # filter and group by pattern
        self.bigrams: dict[tuple, list[tuple[str, str, int]]] = {}
        for (pat, a, b), c in bigram_counter.items():
            if c < MIN_FREQ_BIGRAM:
                continue
            self.bigrams.setdefault(pat, []).append((a, b, c))
        self.trigrams: dict[tuple, list[tuple[str, str, str, int]]] = {}
        for (pat, a, b, c), n in trigram_counter.items():
            if n < MIN_FREQ_TRIGRAM:
                continue
            self.trigrams.setdefault(pat, []).append((a, b, c, n))
        # sort each pattern's list by freq descending
        for pat in self.bigrams:
            self.bigrams[pat].sort(key=lambda x: -x[2])
        for pat in self.trigrams:
            self.trigrams[pat].sort(key=lambda x: -x[3])

        n_bigram = sum(len(v) for v in self.bigrams.values())
        n_trigram = sum(len(v) for v in self.trigrams.values())
        print(f"[phrase] {n_bigram} unique bigram phrases across "
                f"{len(self.bigrams)} POS patterns")
        print(f"[phrase] {n_trigram} unique trigram phrases across "
                f"{len(self.trigrams)} POS patterns")

    def _save_to_cache(self):
        # Save as nested arrays — simplest: pickle since structure is dict
        import pickle
        with open(CACHE.with_suffix(".pkl"), "wb") as f:
            pickle.dump({"bigrams": self.bigrams,
                            "trigrams": self.trigrams}, f)
        print(f"[phrase] cached at {CACHE.with_suffix('.pkl')}")

    def _load_from_cache(self):
        import pickle
        with open(CACHE.with_suffix(".pkl"), "rb") as f:
            d = pickle.load(f)
        self.bigrams = d["bigrams"]
        self.trigrams = d["trigrams"]
        n_b = sum(len(v) for v in self.bigrams.values())
        n_t = sum(len(v) for v in self.trigrams.values())
        print(f"[phrase] loaded {n_b} bigram + {n_t} trigram phrases")

    def get_following_freq(self, prev_word: str) -> dict[str, int]:
        """Build a flat lookup: prev_word → {next_word: freq}.  Used by
        collocation boost during slot-fill."""
        if not hasattr(self, "_follow_cache"):
            self._follow_cache: dict[str, dict[str, int]] = {}
        if prev_word in self._follow_cache:
            return self._follow_cache[prev_word]
        out: dict[str, int] = {}
        for pat, lst in self.bigrams.items():
            for a, b, c in lst:
                if a == prev_word:
                    out[b] = out.get(b, 0) + c
        self._follow_cache[prev_word] = out
        return out

    def collocation_score(self, prev_word: str | None,
                              curr_word: str) -> float:
        """Bigram corpus frequency of (prev_word, curr_word).
        Returns 1.0 if no prev_word (sentence-start context)."""
        if prev_word is None:
            return 1.0
        return float(self.get_following_freq(prev_word).get(curr_word, 0))

    def find_bigram(self, pos_a: int, pos_b: int,
                       contains_word: str | None = None,
                       top_k: int = 20
                       ) -> list[tuple[str, str, int]]:
        """Return bigram candidates for the (pos_a, pos_b) pattern, optionally
        filtered to those containing `contains_word`."""
        cands = self.bigrams.get((pos_a, pos_b), [])
        if contains_word:
            cands = [(a, b, c) for (a, b, c) in cands
                        if a == contains_word or b == contains_word]
        return cands[:top_k]

    def find_trigram(self, pos_a: int, pos_b: int, pos_c: int,
                        contains_word: str | None = None,
                        top_k: int = 20):
        cands = self.trigrams.get((pos_a, pos_b, pos_c), [])
        if contains_word:
            cands = [t for t in cands if contains_word in t[:3]]
        return cands[:top_k]


if __name__ == "__main__":
    import pickle
    with open(ROOT / "data" / "processed" / "manifold_clean.pkl", "rb") as f:
        m = pickle.load(f)
    # tiny stub for testing
    class WM:
        def __init__(self, m):
            self.idx = {l: i for i, l in enumerate(m["lemmas"])}
            self.pi = m["pi"]
    wm = WM(m)
    pp = PhrasePool(wm)
    # spot check: most common DET+NOUN bigrams
    print("\nTop DET+NOUN bigrams:")
    for a, b, c in pp.find_bigram(4, 0)[:15]:
        print(f"  {a} {b}  (×{c})")
    print("\nTop ADJ+NOUN bigrams:")
    for a, b, c in pp.find_bigram(2, 0)[:15]:
        print(f"  {a} {b}  (×{c})")
    print("\nTop PREP+DET trigrams:")
    for a, b, c, n in pp.find_trigram(5, 4, 0)[:10]:
        print(f"  {a} {b} {c}  (×{n})")
