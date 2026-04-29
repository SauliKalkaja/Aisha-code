"""
aisha_lm_helpers.py - LM-polish helpers for the upcoming Phase B/C work.

This module exposes three public functions that Phase B/C (Kotlin/llama.cpp
LM layer) will consume via aisha_bridge.respond():

    is_reflective_question(text)
        Regex classifier for past-self-state questions ("Was my breakfast
        healthy?", "Did I make the right call?"). Validated 17/17 on the
        D15 probe set including edge cases ("What did you do today?",
        "When did the war start?"). Used by Phase C to set the LM
        boundary-bias lambda to 0 (and let Qwen rely on verbatim history)
        for reflective questions.

    aisha_structure(responder, text)
        Compute a 16-D centroid + POS profile + sentence count for any
        text. Returns a JSON-friendly dict (no numpy arrays). This is the
        running structural fingerprint of a conversation, used by
        Phase B/C to track conversation register without crossing into
        Qwen's text channel.

    boundary_with_structural_memory(responder, current_turn, prior_turns,
                                     memory_length)
        Combine current-turn manifold seeds with K=30 manifold neighbours
        of the running centroid (over the last `memory_length` prior
        user turns). Returns a list of words that Phase B/C will use to
        build the LM logit-bias mask on Qwen2.5-0.5B-Instruct.

Implementation is pure Python + numpy. No Chaquopy-specific code; module
runs identically on a laptop for tests and on-device on Android.
"""
from __future__ import annotations

import re
from collections import Counter

import numpy as np


WORD_RE = re.compile(r"[a-zA-Z']+")
SENT_RE = re.compile(r"(?<=[.!?])\s+")
POS_NAMES = ["NOUN", "VERB", "ADJ", "ADV", "PRON_DET", "PREP", "CONJ", "INTJ"]


# ---------------------------------------------------------------------------
# Intent classifier (D15 validated, 17/17 on probe set).
# ---------------------------------------------------------------------------

REFLECTIVE_RE = re.compile(
    r"\b("
    # 1. past-aux + I/my   ("was my", "were my", "did I", "had I", "have I")
    r"(was|were|did|had|have|has)\s+(my|i|i\s+been|i've|ive)"
    # 2. evaluative patterns: "was X (healthy|right|wrong|...)?"
    r"|(was|were|are|am|is)\s+(my|that|this|those|these|the)\s+\w+"
    r"(\s+\w+)?\s+(healthy|unhealthy|good|bad|right|wrong|correct|incorrect|"
    r"safe|wise|unwise|rude|mean|polite|appropriate|fair|reasonable|"
    r"enough|too\s+much|too\s+little)"
    # 3. "did I make/do/say/choose/...", "should I have ..."
    r"|did\s+i\s+(make|do|say|choose|pick|handle|answer|respond)"
    r"|should\s+i\s+have"
    r")\b",
    re.IGNORECASE,
)


def is_reflective_question(text):
    """True if the text reads as a past-self-state reflective question.

    See module docstring for use case. Conservative by design - only fires
    when there's a clear past-tense self-referential pattern; advisory
    questions ("What should I read?") and factual questions ("When did
    the war start?") return False."""
    if not text:
        return False
    return bool(REFLECTIVE_RE.search(text))


# ---------------------------------------------------------------------------
# Structural fingerprinting on the POS-Kahler manifold.
# ---------------------------------------------------------------------------

def aisha_structure(responder, text):
    """Compute structural fingerprint of `text`.

    Returns dict (JSON-friendly, no numpy arrays):
      doc_centroid : list[16] of float    -- mean position on manifold
      pos_profile  : dict[str, float]     -- NOUN/VERB/ADJ/ADV fractions
      n_seeds      : int                  -- count of content words on manifold
      n_sents      : int                  -- sentence count
      mean_step    : float                -- mean inter-sentence centroid step

    Returns None if `text` has no manifold-known content words.
    """
    if not text or not text.strip():
        return None

    sents = [s.strip() for s in SENT_RE.split(text.replace("\n", " ").strip())
              if s.strip()]
    if not sents:
        return None

    Q = responder._main_q
    pos_arg = responder.wm.pi.argmax(axis=1)
    sent_centroids = []
    pos_count = Counter()
    total_seeds = 0

    for s in sents:
        sids = []
        for w in WORD_RE.findall(s.lower()):
            old_i = responder.wm.idx.get(w)
            if old_i is None:
                continue
            p = int(pos_arg[old_i])
            if 0 <= p < len(POS_NAMES):
                pos_count[POS_NAMES[p]] += 1
            new_i = int(responder._old_to_new[old_i])
            if new_i >= 0 and not responder.R.is_stopword[old_i]:
                sids.append(new_i)
        if sids:
            sent_centroids.append(Q[sids].mean(axis=0))
            total_seeds += len(sids)

    if not sent_centroids:
        return None

    sent_centroids = np.stack(sent_centroids)
    doc_centroid = sent_centroids.mean(axis=0)
    if len(sent_centroids) >= 2:
        steps = np.linalg.norm(np.diff(sent_centroids, axis=0), axis=-1)
        mean_step = float(steps.mean())
    else:
        mean_step = 0.0

    total = sum(pos_count.values())
    pos_profile = {p: pos_count[p] / max(total, 1) for p in POS_NAMES[:4]}

    return {
        "doc_centroid": [float(x) for x in doc_centroid],
        "pos_profile":  pos_profile,
        "n_seeds":      int(total_seeds),
        "n_sents":      int(len(sents)),
        "mean_step":    float(mean_step),
    }


# ---------------------------------------------------------------------------
# Boundary expansion with structural memory.
# ---------------------------------------------------------------------------

def _aisha_seeds(responder, text):
    seeds = []
    for w in WORD_RE.findall(text.lower()):
        old_i = responder.wm.idx.get(w)
        if old_i is None or responder.R.is_stopword[old_i]:
            continue
        new_i = int(responder._old_to_new[old_i])
        if new_i >= 0:
            seeds.append(new_i)
    return seeds


def _expand_from_seeds(responder, seeds, k_per_seed=20):
    if not seeds:
        return set()
    Q = responder._main_q
    boundary = set(seeds)
    for s_i in seeds:
        d2 = responder.kahler.mahalanobis_to_seed(Q[s_i], Q)
        d2[s_i] = np.inf
        top = np.argpartition(d2, k_per_seed)[:k_per_seed]
        boundary.update(int(j) for j in top)
    return boundary


def _neighbors_of_centroid(responder, centroid, k=30):
    Q = responder._main_q
    centroid_arr = np.asarray(centroid, dtype=Q.dtype)
    d2 = responder.kahler.mahalanobis_to_seed(centroid_arr, Q)
    return set(int(j) for j in np.argpartition(d2, k)[:k])


def boundary_with_structural_memory(responder, current_turn, prior_turns,
                                      memory_length=5):
    """Build the boundary word list for the LM logit bias.

    Combines current-turn seeds with K=30 manifold neighbours of the
    running centroid taken over the last `memory_length` prior turns.
    Aisha keeps the structural information internal; the function
    returns only words, ready for the LM tokenizer.

    Args:
      responder      : POSResponder instance (already initialised)
      current_turn   : str                -- the user's current message
      prior_turns    : list[str]           -- prior user messages, oldest first
      memory_length  : int                 -- how many prior turns inform memory

    Returns:
      list[str]                            -- boundary words (deduplicated)
    """
    cur_seeds = _aisha_seeds(responder, current_turn)
    boundary = _expand_from_seeds(responder, cur_seeds)

    if memory_length > 0 and prior_turns:
        prior_text = " ".join(prior_turns[-memory_length:])
        struct = aisha_structure(responder, prior_text)
        if struct is not None:
            mem_neighbors = _neighbors_of_centroid(
                responder, struct["doc_centroid"])
            boundary.update(mem_neighbors)

    out = []
    for new_i in boundary:
        try:
            w = responder.kahler.lemmas[int(new_i)]
        except Exception:
            continue
        if isinstance(w, str) and w.isalpha():
            out.append(w)
    return out


# ---------------------------------------------------------------------------
# Module self-test (runs only when invoked directly).
# ---------------------------------------------------------------------------

def _self_test_classifier():
    probes = [
        ("Was my breakfast healthy?",                 True),
        ("Did I make the right call?",                True),
        ("Were my calculations correct?",             True),
        ("Have I been getting enough sleep?",         True),
        ("Was my answer right?",                      True),
        ("Did I sound rude?",                         True),
        ("Should I have done it differently?",        True),
        ("What clothes should I bring?",              False),
        ("How can I improve my morning routine?",     False),
        ("What gift should I get her?",               False),
        ("What technical book would you recommend?",  False),
        ("How should I train from now until the race?", False),
        ("What did you do today?",                    False),
        ("When did the war start?",                   False),
        ("Did the meeting go well?",                  False),
        ("Why is the sky blue?",                      False),
        ("Tell me about Paris.",                      False),
    ]
    correct = sum(1 for q, e in probes if is_reflective_question(q) == e)
    print(f"classifier self-test: {correct}/{len(probes)}")
    for q, e in probes:
        got = is_reflective_question(q)
        if got != e:
            print(f"  MISMATCH: {q!r}  expected={e}  got={got}")
    return correct == len(probes)


if __name__ == "__main__":
    ok = _self_test_classifier()
    raise SystemExit(0 if ok else 1)
