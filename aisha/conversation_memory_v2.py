"""
conversation_memory_v2.py — per-user short+long memory for Aisha.

Short-term:  deque of last 100 turns (in RAM, slow decay across turns)
Long-term:   persistent JSON per user_id  (data/memory/{user_id}.json)

Memory does NOT change manifold weights.  It applies a multiplicative
bonus to per-word pool ranking at scoring time:

  bonus(w) = 1 + α · short_score(w) + β · long_score(w)
           where:
             short_score = sum over recent turns where w appeared,
                            decayed by (γ ** age)
             long_score  = persistent cumulative ± from feedback

Initial weights chosen high; tune after subagent tests.
"""
from __future__ import annotations

import json
import time
from collections import deque
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parent
MEMORY_DIR = ROOT / "data" / "memory"
MEMORY_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_SHORT_MAXLEN = 100
DEFAULT_DECAY        = 0.97   # per turn (slow decay over 100 turns)
DEFAULT_BONUS_SHORT  = 2.0    # short-term multiplier (start high)
DEFAULT_BONUS_LONG   = 1.0    # long-term multiplier
DEFAULT_LONG_GROW    = 0.5    # +/- per feedback
DEFAULT_PARROT       = 3.0    # user-words-specific (parrot) bonus
DEFAULT_STALENESS    = 0.4    # anti-staleness on Aisha's recent words
DEFAULT_SHAPE_STALE  = 0.5    # anti-staleness on Aisha's recent shapes
STALENESS_FLOOR      = 0.10   # bonus can't drop below 10% of unstale value
AXIS_POS = {"M": 0, "chi": 1, "spin": 2}


class MemoryProfile:
    """Tunable parameters per user/agent.  Used to A/B-test how
    different memory shapes affect Aisha's behavior."""

    def __init__(self,
                  short_maxlen: int = DEFAULT_SHORT_MAXLEN,
                  short_decay: float = DEFAULT_DECAY,
                  alpha_short: float = DEFAULT_BONUS_SHORT,
                  beta_long: float = DEFAULT_BONUS_LONG,
                  parrot_strength: float = DEFAULT_PARROT,
                  long_grow: float = DEFAULT_LONG_GROW,
                  staleness_strength: float = DEFAULT_STALENESS,
                  shape_staleness: float = DEFAULT_SHAPE_STALE,
                  name: str = "default"):
        self.name = name
        self.short_maxlen = short_maxlen
        self.short_decay = short_decay
        self.alpha_short = alpha_short
        self.beta_long = beta_long
        self.parrot_strength = parrot_strength
        self.long_grow = long_grow
        self.staleness_strength = staleness_strength
        self.shape_staleness = shape_staleness

    def to_dict(self):
        return {k: v for k, v in self.__dict__.items()}


# Predefined profiles for subagent testing
PROFILES = {
    "default":         MemoryProfile(name="default"),
    "balanced":        MemoryProfile(name="balanced",
                                          alpha_short=2.0,
                                          parrot_strength=3.0),
    "low_parrot":      MemoryProfile(name="low_parrot",
                                          parrot_strength=0.5,
                                          alpha_short=1.0),
    "high_parrot":     MemoryProfile(name="high_parrot",
                                          parrot_strength=8.0,
                                          alpha_short=2.0,
                                          staleness_strength=0.2,
                                          shape_staleness=0.2),
    "no_parrot":       MemoryProfile(name="no_parrot",
                                          parrot_strength=0.0,
                                          alpha_short=2.0),
    "fast_decay":      MemoryProfile(name="fast_decay",
                                          short_decay=0.85,
                                          parrot_strength=3.0),
    "slow_decay":      MemoryProfile(name="slow_decay",
                                          short_decay=0.995,
                                          short_maxlen=200,
                                          parrot_strength=3.0),
    # memory_heavy was the worst lockup — bump anti-staleness so it
    # diversifies vocab/shape even while keeping high recall weights.
    "memory_heavy":    MemoryProfile(name="memory_heavy",
                                          alpha_short=5.0,
                                          beta_long=3.0,
                                          parrot_strength=4.0,
                                          staleness_strength=0.7,
                                          shape_staleness=0.7),
    "memory_light":    MemoryProfile(name="memory_light",
                                          alpha_short=0.5,
                                          beta_long=0.2,
                                          parrot_strength=1.0,
                                          staleness_strength=0.2),
    # echo *should* repeat — keep staleness low so the profile still
    # behaves like a parrot.
    "echo":            MemoryProfile(name="echo",
                                          parrot_strength=10.0,
                                          alpha_short=3.0,
                                          staleness_strength=0.1,
                                          shape_staleness=0.1),
}


class ConversationMemory:
    def __init__(self, user_id: str = "default",
                  profile: MemoryProfile | None = None,
                  short_maxlen: int = DEFAULT_SHORT_MAXLEN,
                  decay: float = DEFAULT_DECAY,
                  alpha_short: float = DEFAULT_BONUS_SHORT,
                  beta_long: float = DEFAULT_BONUS_LONG):
        self.user_id = user_id
        # If a profile is supplied, it overrides the per-arg defaults
        if profile is not None:
            self.profile = profile
            self.short_maxlen = profile.short_maxlen
            self.decay = profile.short_decay
            self.alpha_short = profile.alpha_short
            self.beta_long = profile.beta_long
            self.parrot_strength = profile.parrot_strength
            self.long_grow = profile.long_grow
            self.staleness_strength = profile.staleness_strength
            self.shape_staleness = profile.shape_staleness
        else:
            self.profile = MemoryProfile(name=user_id)
            self.short_maxlen = short_maxlen
            self.decay = decay
            self.alpha_short = alpha_short
            self.beta_long = beta_long
            self.parrot_strength = DEFAULT_PARROT
            self.long_grow = DEFAULT_LONG_GROW
            self.staleness_strength = DEFAULT_STALENESS
            self.shape_staleness = DEFAULT_SHAPE_STALE
        # Short-term: list of dicts per turn
        # each entry: {"turn_n", "user_words", "aisha_words",
        #               "user_shapes", "aisha_shapes"}
        self.short = deque(maxlen=short_maxlen)
        # Long-term: persistent dicts loaded from JSON
        self.long_vocab: dict[int, float] = {}     # word_idx → score
        self.long_shapes: dict[str, float] = {}    # "axis:shape_idx" → score
        self.turn_count = 0
        self._path = MEMORY_DIR / f"{user_id}.json"
        self._load()

    # ----- short-term turns ------------------------------------------
    def add_turn(self, user_word_idx: Iterable[int],
                    aisha_word_idx: Iterable[int],
                    user_shapes: tuple = (None, None, None),
                    aisha_shapes: tuple = (None, None, None)):
        self.turn_count += 1
        self.short.append({
            "turn_n": self.turn_count,
            "user_words": list(user_word_idx),
            "aisha_words": list(aisha_word_idx),
            "user_shapes": tuple(user_shapes),
            "aisha_shapes": tuple(aisha_shapes),
        })

    # ----- bonus lookup ----------------------------------------------
    def short_score(self, word_idx: int) -> float:
        """General short-term recency score: USER words only.  Aisha's
        own words feed into aisha_recent_score (anti-staleness), not
        here — bouncing them back to herself was the lockup we found
        in the 30-turn run.  Topic coherence already flows through
        parrot_score; this provides the broader, decayed user-word
        recall."""
        score = 0.0
        for entry in self.short:
            if word_idx in entry["user_words"]:
                age = self.turn_count - entry["turn_n"]
                score += self.decay ** age
        return score

    def parrot_score(self, word_idx: int) -> float:
        """USER-ONLY recency score.  Drives the parrot mechanism —
        boost user's content words specifically (separate from
        general short-term)."""
        score = 0.0
        for entry in self.short:
            age = self.turn_count - entry["turn_n"]
            mult = self.decay ** age
            if word_idx in entry["user_words"]:
                score += 1.0 * mult
        return score

    def long_score(self, word_idx: int) -> float:
        return self.long_vocab.get(word_idx, 0.0)

    def aisha_recent_score(self, word_idx: int) -> float:
        """Aisha-only recency.  Drives anti-staleness penalty —
        words Aisha just used get suppressed."""
        score = 0.0
        for entry in self.short:
            if word_idx in entry["aisha_words"]:
                age = self.turn_count - entry["turn_n"]
                score += self.decay ** age
        return score

    def aisha_shape_recent(self, axis: str, shape_idx: int) -> float:
        """Aisha-only short-term shape recency, per axis (M/chi/spin).
        Used to dampen the shape transition matrix toward shapes
        Aisha hasn't used recently."""
        if shape_idx is None:
            return 0.0
        pos = AXIS_POS.get(axis)
        if pos is None:
            return 0.0
        score = 0.0
        for entry in self.short:
            shapes = entry.get("aisha_shapes")
            if not shapes or shapes[pos] != shape_idx:
                continue
            age = self.turn_count - entry["turn_n"]
            score += self.decay ** age
        return score

    def vocab_bonus(self, word_idx: int) -> float:
        """Combined multiplier:
            (1 + α·short + β·long + γ·parrot) / (1 + δ·aisha_recent)
        Parrot is a separate term so user words can be tuned
        independently of general memory.  The denominator is the
        anti-staleness penalty that pushes Aisha away from words
        she just used."""
        s = self.short_score(word_idx)
        l = self.long_score(word_idx)
        p = self.parrot_score(word_idx)
        a = self.aisha_recent_score(word_idx)
        bonus = (1.0 + self.alpha_short * s
                       + self.beta_long * l
                       + self.parrot_strength * p)
        penalty = 1.0 / (1.0 + self.staleness_strength * a)
        return bonus * max(penalty, STALENESS_FLOOR)

    def shape_bonus(self, axis: str, shape_idx: int) -> float:
        """Bonus/penalty for shape transitions (used to bias matrix
        lookups).  Negative = avoid (variety), positive = reinforce."""
        return self.long_shapes.get(f"{axis}:{shape_idx}", 0.0)

    # ----- feedback --------------------------------------------------
    def thumbs(self, score: int, words: Iterable[int] = (),
                  shapes: dict | None = None,
                  grow: float | None = None):
        """Apply +1/-1 feedback to most recent Aisha response.
        Adjusts long_vocab and long_shapes accordingly."""
        if grow is None:
            grow = self.long_grow
        for w in words:
            self.long_vocab[w] = self.long_vocab.get(w, 0.0) + grow * score
        if shapes:
            for axis, shape_idx in shapes.items():
                if shape_idx is None:
                    continue
                key = f"{axis}:{shape_idx}"
                self.long_shapes[key] = self.long_shapes.get(key, 0.0) + grow * score
        self._save()

    # ----- erase / persistence ---------------------------------------
    def erase(self, short: bool = True, long: bool = True):
        if short:
            self.short.clear()
            self.turn_count = 0
        if long:
            self.long_vocab.clear()
            self.long_shapes.clear()
        self._save()

    def _load(self):
        if not self._path.exists():
            return
        try:
            d = json.loads(self._path.read_text())
            self.long_vocab = {int(k): float(v)
                                  for k, v in d.get("long_vocab", {}).items()}
            self.long_shapes = {str(k): float(v)
                                   for k, v in d.get("long_shapes", {}).items()}
            self.turn_count = int(d.get("turn_count_seen", 0))
        except Exception:
            pass

    def _save(self):
        d = {"user_id": self.user_id,
              "long_vocab": {str(k): v for k, v in self.long_vocab.items()},
              "long_shapes": dict(self.long_shapes),
              "turn_count_seen": self.turn_count,
              "saved_ts": time.time()}
        self._path.write_text(json.dumps(d, indent=2))

    # ----- diagnostics -----------------------------------------------
    def stats(self) -> dict:
        return {
            "user_id": self.user_id,
            "turn_count": self.turn_count,
            "short_len": len(self.short),
            "long_vocab_size": len(self.long_vocab),
            "long_shapes_size": len(self.long_shapes),
            "short_top5_words": [
                (w, self.short_score(w))
                for w in sorted(set(
                    w for entry in self.short
                    for w in entry["user_words"] + entry["aisha_words"]),
                    key=lambda w: -self.short_score(w))[:5]
            ],
        }
