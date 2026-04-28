"""
grammar.py — rule-based grammar layer for phrase-jump output.

Takes a list of (template, words) pairs and applies English grammar
rules to produce a more readable surface string.  Operations:

  - 3sg verb agreement when subject-pronoun is identifiable
  - a/an choice based on next word's leading sound
  - imperative bare-verb when phrase is V-initial without subject
  - pronoun case (nominative subject, accusative object)
  - capitalise + select terminator (?/!/.)
  - safe insertions only: function words (a/an/the/is/are/was/were/
    has/have/does/do/will/can) — these are CHOICES from a fixed
    allowlist, never new content
  - never adds new content nouns/verbs/adjs — those have to come from
    Aisha's manifold

This is grammar enforcement, not content generation.  Word-salad in →
grammatical word-salad out (still salad meaning-wise, but English-shaped).
"""
from __future__ import annotations

import re
from typing import Iterable

# ---------------------------------------------------------------------------
# Inflection rules — tiny but cover common cases.
# ---------------------------------------------------------------------------

# Words that start with a vowel letter but a consonant sound — take "a".
_CONSONANT_SOUND_VOWEL_START = re.compile(
    r"^(?:university|use[rds]?|usual|usually|"
    r"european|eulogy|euphor|union|unique|unit|unicycle|"
    r"one|once|onerous)", re.I)

# Words that start with a consonant letter but a vowel sound — take "an".
_VOWEL_SOUND_CONSONANT_START = re.compile(
    r"^(?:hour|honest|honor|honour|heir)", re.I)


def a_or_an(next_word: str) -> str:
    """Return 'a' or 'an' for the next word, by leading sound."""
    if not next_word:
        return "a"
    if _CONSONANT_SOUND_VOWEL_START.match(next_word):
        return "a"
    if _VOWEL_SOUND_CONSONANT_START.match(next_word):
        return "an"
    if next_word[:1].lower() in "aeiou":
        return "an"
    return "a"


# 3sg verb conjugation.  Irregulars first, then suffix rules.
_3SG_IRREG = {
    "be": "is", "have": "has", "do": "does", "go": "goes",
    "say": "says", "see": "sees",
}

def to_3sg(verb: str) -> str:
    """Convert a base-form verb to its 3sg present form."""
    v = verb.lower()
    if v in _3SG_IRREG:
        return _3SG_IRREG[v]
    if re.search(r"(s|x|ch|sh|z)$", v):
        return v + "es"
    if re.search(r"[^aeiou]y$", v):
        return v[:-1] + "ies"
    if v.endswith("o"):
        return v + "es"
    return v + "s"


# Plural noun: rough rules.
_PLURAL_IRREG = {
    "child": "children", "person": "people", "man": "men", "woman": "women",
    "tooth": "teeth", "foot": "feet", "mouse": "mice", "goose": "geese",
    "sheep": "sheep", "fish": "fish", "deer": "deer",
}


def to_plural(noun: str) -> str:
    n = noun.lower()
    if n in _PLURAL_IRREG:
        return _PLURAL_IRREG[n]
    if re.search(r"(s|x|z|ch|sh)$", n):
        return n + "es"
    if re.search(r"[^aeiou]y$", n):
        return n[:-1] + "ies"
    if re.search(r"(f|fe)$", n):
        return re.sub(r"f?e?$", "ves", n)
    return n + "s"


# Pronoun case map: nominative ↔ accusative.
_NOM = {"i", "we", "you", "he", "she", "it", "they"}
_ACC = {"me", "us", "him", "her", "them"}
_TO_NOM = {"me": "i", "us": "we", "him": "he", "her": "she", "them": "they"}
_TO_ACC = {"i": "me", "we": "us", "he": "him", "she": "her", "they": "them"}

# Pronouns that take 3sg verb agreement.
_3SG_PRO = {"he", "she", "it"}


# Articles considered "definite-ish" already.
_DET_KEEP = {"the", "this", "that", "these", "those",
              "his", "her", "its", "their", "our", "my", "your"}


# ---------------------------------------------------------------------------
# Phrase-level rules
# ---------------------------------------------------------------------------

def _is_3sg_subject(word: str) -> bool:
    return word.lower() in _3SG_PRO


def _capitalise_first(words: list[str]) -> list[str]:
    if not words:
        return words
    out = list(words)
    out[0] = out[0][:1].upper() + out[0][1:]
    return out


def grammar_phrase(template: tuple, words: list[str],
                    roles: list[str] | None = None) -> list[str]:
    """Apply phrase-level rules to one (template, words) pair.

    If `roles` is provided (same length as template), the rules use
    role information to enforce subject-verb agreement on any
    (subj, root) pair, not just (PRON-VERB) prefix.  Pronoun case is
    enforced strictly: subj → nominative, obj/pobj → accusative.
    """
    if len(template) != len(words):
        return list(words)
    out = list(words)
    pos = list(template)
    rl  = list(roles) if roles else [None] * len(pos)

    # Pass 1: a/an choice for any DET that's "a"/"an".
    for i, (p, w) in enumerate(zip(pos, out)):
        if p == "DET" and w.lower() in {"a", "an"}:
            j = i + 1
            while j < len(out) and not out[j]:
                j += 1
            if j < len(out):
                out[i] = a_or_an(out[j])

    # Pass 2: subject-verb agreement.
    # With roles: find the (subj, root) pair (subj position, root position)
    # and inflect root based on subj.
    # Without roles: fall back to PRON-VERB adjacency rule.
    if any(r == "subj" for r in rl) and any(r == "root" for r in rl):
        try:
            si = rl.index("subj")
            ri = rl.index("root")
            if 0 <= si < len(out) and 0 <= ri < len(out):
                if _is_3sg_subject(out[si]):
                    out[ri] = to_3sg(out[ri])
        except ValueError:
            pass
    else:
        for i in range(len(pos) - 1):
            if pos[i] == "PRON" and pos[i + 1] == "VERB":
                if _is_3sg_subject(out[i]):
                    out[i + 1] = to_3sg(out[i + 1])

    # Pass 3: pronoun case using roles when available.
    for i, p in enumerate(pos):
        if p != "PRON":
            continue
        w = out[i].lower()
        role = rl[i] if i < len(rl) else None
        if role == "subj":
            # Force nominative.
            if w in _ACC and w in _TO_NOM:
                out[i] = _TO_NOM[w]
        elif role in ("obj", "pobj"):
            # Force accusative.
            if w in _NOM and w in _TO_ACC:
                out[i] = _TO_ACC[w]
        else:
            # Fall back to position-based heuristic.
            prev_pos = pos[i - 1] if i > 0 else None
            if prev_pos in ("VERB", "ADP") and w in _NOM and w in _TO_ACC:
                out[i] = _TO_ACC[w]
            if i == 0 and w in _ACC and w in _TO_NOM:
                out[i] = _TO_NOM[w]

    return out


# ---------------------------------------------------------------------------
# Sentence-level rules
# ---------------------------------------------------------------------------

# Light interrogative cues (when output should end with "?").  Match the
# query, not the response — so only used by the caller who knows the query.
_QUESTION_QUERY_PREFIX = re.compile(
    r"^\s*(what|how|why|when|where|which|who|are|is|do|does|did|"
    r"could|would|should|may|can)\b", re.I)


def _select_terminator(text: str, query: str | None) -> str:
    """Select '?' if query was a question and output also looks like one;
    otherwise '.'.  Default '.'."""
    if not text:
        return "."
    if text.endswith(("?", "!", ".")):
        return ""
    return "."


def grammar_compose(phrases: list[dict],
                     query: str | None = None) -> str:
    """Compose phrase-level outputs into a sentence with grammar rules
    applied.

    `phrases` is a list of dicts: {"template": tuple, "words": list[str]}
    """
    all_words: list[str] = []
    for ph in phrases:
        out = grammar_phrase(tuple(ph["template"]), list(ph["words"]))
        all_words.extend(out)

    # Drop empty / whitespace-only entries.
    all_words = [w for w in all_words if w and w.strip()]

    # Standalone "i" → "I" (always-capital subject pronoun).
    all_words = ["I" if w.lower() == "i" else w for w in all_words]

    # Capitalise first word.
    all_words = _capitalise_first(all_words)

    # Drop adjacent duplicate function-word tokens (e.g. "the the").
    deduped: list[str] = []
    for w in all_words:
        if deduped and deduped[-1].lower() == w.lower() and \
                w.lower() in {"a", "an", "the", "is", "are", "was", "were",
                                "of", "in", "on", "at", "to", "for", "and"}:
            continue
        deduped.append(w)

    text = " ".join(deduped)
    text = text + _select_terminator(text, query)
    return text


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------

def _cli():
    examples = [
        # (query, [(template, words)], expected-flavor-comment)
        ("what is entropy",
         [(("PRON", "VERB"), ["he", "run"])],
         "→ He runs."),
        ("I miss her",
         [(("PRON", "VERB", "PRON"), ["i", "see", "she"])],
         "→ I see her."),
        ("what is the capital of France",
         [(("DET", "NOUN"), ["a", "apple"])],
         "→ An apple."),
        ("describe a transformer engine",
         [(("DET", "ADJ", "NOUN"), ["a", "old", "car"])],
         "→ An old car."),
        ("how do you feel",
         [(("PRON",), ["me"])],
         "→ I."),
        ("test 3sg",
         [(("PRON", "VERB"), ["she", "go"])],
         "→ She goes."),
    ]
    for q, phrases, want in examples:
        ph_dicts = [{"template": list(t), "words": w} for t, w in phrases]
        got = grammar_compose(ph_dicts, query=q)
        print(f"{q!r:>40s}  -> {got!r}   ({want})")


if __name__ == "__main__":
    _cli()
