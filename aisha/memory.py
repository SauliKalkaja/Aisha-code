"""
memory.py — structural memory + Layer 4 (cross-turn analytical jumps).

What it stores per turn:
  - speaker (user | aisha)
  - cleaned text
  - 5-dim mean coordinate signature (M, χ, s, v, a)
  - 8-dim octant histogram
  - content words (for topic recurrence detection)
  - style, length, timestamp

What Layer 4 computes:
  - Between consecutive turns, signature jumps in 5-axis space
  - Big jumps = topic/mood shifts (phase boundaries in the conversation)

What it emits to the LM:
  - A compact ~50-token summary:
      · mood trajectory across recent turns
      · topic words recurring in ≥2 turns
      · whether the last turn shifted phase

Storage:
  - RAM: list of Turn dataclasses (fast access, negligible size)
  - Disk: append-only JSONL (persistent across sessions)

Size: ~1-2 KB per turn. 1000 turns = ~2 MB. Chat-scale trivial.
"""
from __future__ import annotations

import json
import re
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

_WORD_RE = re.compile(r"[a-zA-Z']+")


@dataclass
class Turn:
    """One stored conversational turn."""
    speaker:      str
    text:         str
    style:        str
    timestamp:    float
    length:       int
    signature:    dict           # {"M", "chi", "s", "v", "a"}
    octant_hist:  list[float]    # 8-dim
    content_words: list[str]

    def sig_vec(self) -> np.ndarray:
        """5-dim vector for analytical-jump comparisons."""
        s = self.signature
        return np.array([s["M"], s["chi"], s["s"], s["v"], s["a"]])


# ======================================================================
class Memory:
    """Persistent structural memory over a conversation."""

    CONTENT_POS = frozenset({0, 1, 2, 3})   # Noun, Verb, Adj, Adv

    def __init__(self, wm, log_path: Path | None = None):
        self.wm = wm
        self.turns: list[Turn] = []
        self.log_path = log_path
        # Load history if present
        if self.log_path and self.log_path.exists():
            self._load()

    # ------------------------------------------------------------------
    def _content_words(self, text: str) -> list[str]:
        out = []
        seen = set()
        for t in _WORD_RE.findall(text.lower()):
            if t in seen or t not in self.wm.idx: continue
            pi_argmax = int(self.wm.pi[self.wm.idx[t]].argmax())
            if pi_argmax in self.CONTENT_POS and len(t) >= 3:
                out.append(t); seen.add(t)
        return out

    def _signature(self, text: str, axes: dict) -> tuple[dict, list[float]]:
        """Compute signature and octant hist from text, given axis arrays
        {M, chi, s, v, a, oct8, mask}."""
        tokens = [t.lower() for t in _WORD_RE.findall(text)]
        idx = [self.wm.idx[t] for t in tokens
                if t in self.wm.idx and axes["mask"][self.wm.idx[t]]]
        if not idx:
            zero = {"M":0.0, "chi":0.0, "s":0.0, "v":0.0, "a":0.0}
            return zero, [0.125] * 8
        idx_arr = np.asarray(idx)
        sig = {
            "M":   float(axes["M"][idx_arr].mean()),
            "chi": float(axes["chi"][idx_arr].mean()),
            "s":   float(axes["s"][idx_arr].mean()),
            "v":   float(axes["v"][idx_arr].mean()),
            "a":   float(axes["a"][idx_arr].mean()),
        }
        hist = np.bincount(axes["oct8"][idx_arr], minlength=8).astype(float)
        hist = hist / max(hist.sum(), 1)
        return sig, hist.tolist()

    # ------------------------------------------------------------------
    def record(self, speaker: str, text: str, style: str,
                axes: dict) -> Turn:
        """Append a turn to memory."""
        sig, hist = self._signature(text, axes)
        turn = Turn(
            speaker=speaker,
            text=text,
            style=style,
            timestamp=time.time(),
            length=len(_WORD_RE.findall(text)),
            signature=sig,
            octant_hist=hist,
            content_words=self._content_words(text),
        )
        self.turns.append(turn)
        if self.log_path:
            self._append_to_log(turn)
        return turn

    def _append_to_log(self, turn: Turn):
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_path, "a") as f:
            f.write(json.dumps(asdict(turn)) + "\n")

    def _load(self):
        with open(self.log_path) as f:
            for line in f:
                line = line.strip()
                if not line: continue
                try:
                    d = json.loads(line)
                    self.turns.append(Turn(**d))
                except Exception:
                    continue

    def recent(self, n: int = 5) -> list[Turn]:
        return self.turns[-n:] if self.turns else []

    def clear(self):
        self.turns = []
        if self.log_path and self.log_path.exists():
            self.log_path.unlink()

    # ==================================================================
    # LAYER 4 — analytical jumps across turns
    # ==================================================================
    def phase_jumps(self, turns: list[Turn] | None = None) -> list[float]:
        """Return 5-axis Euclidean jump magnitudes between consecutive turns."""
        turns = turns or self.turns
        if len(turns) < 2: return []
        out = []
        for a, b in zip(turns[:-1], turns[1:]):
            out.append(float(np.linalg.norm(b.sig_vec() - a.sig_vec())))
        return out

    def detect_phase_shift(self, turns: list[Turn] | None = None,
                             threshold_sigma: float = 1.0) -> bool:
        """Was the most recent turn a phase boundary?  Uses same σ-above-mean
        threshold as layer 2/3."""
        jumps = self.phase_jumps(turns)
        if len(jumps) < 3: return False
        last = jumps[-1]
        prev = jumps[:-1]
        mu = float(np.mean(prev))
        sd = float(np.std(prev))
        if sd < 1e-6: return False
        return last > mu + threshold_sigma * sd

    # ==================================================================
    # LM prompt summarisation
    # ==================================================================
    def summary_for_llm(self, n: int = 5) -> str:
        """Compact description of recent conversation state — for prompt."""
        recent = self.recent(n)
        if len(recent) < 2: return ""

        # Mood trajectory (style across speakers, deduplicated consecutive)
        styles = [t.style for t in recent]
        mood_seq = []
        for s in styles:
            if not mood_seq or mood_seq[-1] != s:
                mood_seq.append(s)
        mood_str = " → ".join(mood_seq)

        # Topic recurrences: content words in ≥ 2 turns
        recurrence = Counter()
        for t in recent:
            for w in set(t.content_words):
                recurrence[w] += 1
        recurring = [w for w, c in recurrence.most_common()
                      if c >= 2][:5]

        # Phase shift
        shifted = self.detect_phase_shift()

        parts = []
        if mood_seq: parts.append(f"mood so far: {mood_str}")
        if recurring:
            parts.append(f"recurring topic: {', '.join(recurring)}")
        if shifted: parts.append("note: mood just shifted")
        return "; ".join(parts)


# ======================================================================
if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from aisha_respond import Responder

    r = Responder(rng_seed=0)
    axes = {
        "M": r.M_n, "chi": r.chi_n, "s": r.spin_n,
        "v": r.v, "a": r.a, "oct8": r.oct8, "mask": r.mask10k,
    }

    mem = Memory(r.wm, log_path=Path("/tmp/test_memory.jsonl"))
    mem.clear()

    demo_turns = [
        ("user",   "Hey, what's for dinner tonight?",           "casual"),
        ("aisha",  "I was thinking pasta. Works for you?",      "casual"),
        ("user",   "Pasta again? We had that last week twice!", "heated"),
        ("aisha",  "You seem frustrated. We can pick anything.", "civilized"),
        ("user",   "I'm just tired. Whatever is fine, really.",  "emotional"),
    ]
    for sp, tx, st in demo_turns:
        t = mem.record(sp, tx, st, axes)
        print(f"[{sp:<5s} {st:<9s}]  {tx}")
        print(f"         sig={t.signature}  content={t.content_words}")

    print("\nJumps between consecutive turns:")
    for j in mem.phase_jumps():
        print(f"  {j:.3f}")
    print(f"\nLast turn phase shift: {mem.detect_phase_shift()}")
    print(f"\nSummary for LM:\n  {mem.summary_for_llm()!r}")
