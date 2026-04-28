"""
pipeline.py — the full Aisha conversation pipeline.

Flow:
  1. User input
  2. Harper-in : normalise typos/grammar in user's text
  3. Aisha stack (layers 1 + 2 + 3): build structured response template
  4. LM : fill the template with coherent English
  5. Harper-out : polish LM's output
  6. Return

The LM step is a pluggable callable.  This file provides:
  - `structure_only(a_text, style)`  — stops after step 3, returns template
  - `respond(a_text, style, llm_fn)`  — full pipeline
  - `MockLLM`  — deterministic placeholder; just concatenates phrase words
                 for smoke-testing without a real LM attached
  - CLI:  `python pipeline.py --input "..." --style heated --structure-only`
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import harper_polish as hp                            # noqa: E402
from pos_select import POSSelector, CHANNEL_NAMES    # noqa: E402
from layer2 import JumpLayer, OCTANT_SHORT           # noqa: E402
from layer3 import ClauseLayer                       # noqa: E402
from memory import Memory                            # noqa: E402


# ----------------------------------------------------------------------
# Faithfulness gate.  LM output may not introduce content words that
# weren't in Aisha's raw word stream.  Function words (articles, preps,
# auxiliaries, conjunctions, negation) may be added freely for grammar.
# One invented content word → reject the polish, emit raw stream.
# ----------------------------------------------------------------------
import re as _re

# Function words the LM may introduce freely.
_FUNCTION_WORDS = frozenset({
    # articles & determiners
    "a", "an", "the",
    # prepositions
    "to", "of", "in", "on", "at", "with", "for", "from", "by", "about",
    "into", "onto", "out", "off", "over", "under", "up", "down",
    "through", "between", "as", "than", "like", "near",
    # auxiliaries / copula
    "is", "are", "was", "were", "be", "been", "being",
    "am", "'s", "'re", "'m", "'ve", "'ll", "'d",
    "have", "has", "had", "having",
    "do", "does", "did", "doing",
    "will", "would", "shall", "should", "can", "could", "may", "might",
    "must",
    # conjunctions
    "and", "or", "but", "so", "yet", "because", "if", "when", "while",
    "though", "although", "since", "unless", "until",
    # negation / quantifiers
    "not", "no", "n't", "never", "all", "some", "any", "none", "every",
    # basic anaphora that polish may need to insert
    "it", "its", "that", "this", "these", "those",
    # filler punctuation tokens we tokenise as words
    "ll", "re", "s", "t", "d", "m", "ve",
})

_WORD_RE = _re.compile(r"[a-zA-Z']+")


def _stem(w: str) -> str:
    """Tiny rule-based stemmer.  Strips common English suffixes so
    e.g. 'getting'/'gets'/'got' all collapse to roughly 'get'.  Crude
    but good enough for the gate.  Never returns empty string."""
    w = w.lower().rstrip("'")
    for suf in ("ingly", "edly", "ing", "ied", "ies", "ied",
                  "ed", "es", "er", "est", "ly", "s"):
        if len(w) > len(suf) + 2 and w.endswith(suf):
            return w[: -len(suf)]
    return w


def faithfulness_gate(polished: str, raw_stream_words: list[str]
                       ) -> tuple[bool, str]:
    """Return (ok, reason).  ok=True iff every content word in `polished`
    is traceable to a word in `raw_stream_words` (after stemming) OR is
    in the function-word allowlist."""
    if not polished or not polished.strip():
        return False, "empty polish"

    aisha_stems = {_stem(w) for w in raw_stream_words if w}
    polished_toks = [t.lower() for t in _WORD_RE.findall(polished)]
    if not polished_toks:
        return False, "polish had no word tokens"

    invented: list[str] = []
    for t in polished_toks:
        if t in _FUNCTION_WORDS:
            continue
        if _stem(t) in aisha_stems:
            continue
        # Allow short numerals / single letters that aren't real content.
        if len(t) <= 1:
            continue
        invented.append(t)

    if invented:
        return False, f"invented content words: {invented[:6]}"
    return True, "ok"


def _raw_fallback(raw_stream_words: list[str]) -> str:
    """When the gate rejects the polish, ship Aisha's raw stream itself."""
    if not raw_stream_words:
        return ""
    s = " ".join(raw_stream_words)
    s = s[0].upper() + s[1:]
    if not s.endswith(("?", "!", ".")):
        s = s + "."
    return s


# ======================================================================
class Pipeline:
    def __init__(self, memory_log: Path | None = None):
        print("[pipeline] loading Aisha stack…", flush=True)
        self.pos_sel = POSSelector()
        self.layer2 = self.pos_sel.layer
        self.layer3 = ClauseLayer(self.layer2)

        R = self.pos_sel.R
        self.axes = {
            "M": R.M_n, "chi": R.chi_n, "s": R.spin_n,
            "v": R.v,   "a": R.a,
            "oct8": R.oct8, "mask": R.mask10k,
        }
        self.memory = Memory(R.wm, log_path=memory_log)
        if self.memory.turns:
            print(f"[pipeline] loaded {len(self.memory.turns)} prior turns "
                  f"from {memory_log}")

    # ------------------------------------------------------------------
    # SBERT-based style router — replaces the (v,a) centroid method (36%)
    # with a sentence-embedding classifier (75% on held-out conversations).
    # Lazy-loaded so importing pipeline.py stays fast for non-routing use.
    _SBERT_ROUTER = None

    def _get_sbert_router(self):
        if Pipeline._SBERT_ROUTER is None:
            try:
                from style_router import StyleRouter
                Pipeline._SBERT_ROUTER = StyleRouter()
            except Exception as e:
                print(f"[pipeline] SBERT router unavailable, falling back: {e}",
                      flush=True)
                Pipeline._SBERT_ROUTER = False
        return Pipeline._SBERT_ROUTER

    def _memory_perturbation(self,
                                n_recent: int = 5,
                                strength: float = 0.6
                                ) -> tuple[float, float]:
        """Compute (dv, da) perturbation from recent memory turns.

        Exponentially-weighted mean of recent turn signatures' (v, a),
        scaled by `strength`.  Empty memory → (0, 0), so behavior is
        identical to no-memory pipeline.
        """
        recent = self.memory.recent(n=n_recent)
        if not recent:
            return 0.0, 0.0
        import numpy as _np
        weights = _np.array([0.5 ** (len(recent) - 1 - i)
                             for i in range(len(recent))], dtype=float)
        weights = weights / weights.sum()
        v_arr = _np.array([t.signature["v"] for t in recent], dtype=float)
        a_arr = _np.array([t.signature["a"] for t in recent], dtype=float)
        dv = float((v_arr * weights).sum()) * strength
        da = float((a_arr * weights).sum()) * strength
        return dv, da

    def _infer_style_from_text(self, text: str) -> str:
        """SBERT classifier first (75% on held-out); centroid fallback."""
        router = self._get_sbert_router()
        if router:
            try:
                return router.predict(text)
            except Exception:
                pass

        # Fallback: legacy (v,a) centroid method (36% accuracy).
        import re as _re
        toks = [t.lower() for t in _re.findall(r"[a-zA-Z']+", text)]
        idx = [self.pos_sel.wm.idx[t] for t in toks
                if t in self.pos_sel.wm.idx and self.pos_sel.R.mask10k[
                    self.pos_sel.wm.idx[t]]]
        if not idx: return "casual"
        import numpy as _np
        v = float(self.pos_sel.R.v[_np.asarray(idx)].mean())
        a = float(self.pos_sel.R.a[_np.asarray(idx)].mean())
        return self.pos_sel.R._infer_style(v, a)

    # ------------------------------------------------------------------
    def build_structure(self, a_text_clean: str, style: str | None,
                          n_samples: int = 20) -> dict:
        """Run layer 1 → 2 → 3 and emit a structured response template."""
        out = self.pos_sel.respond(a_text_clean, style=style,
                                      n_samples=n_samples)
        b_idx = out["b_idx"]
        if len(b_idx) < 3:
            return {"ok": False, "reason": "empty or too-short word stream"}

        groups = self.layer2.group(b_idx, threshold_sigma=1.0)
        phrases = [self.layer2.describe(g) for g in groups]
        clauses = self.layer3.clause(phrases, threshold_sigma=1.0)

        # Topic anchors: content words from A's OWN text (what the LM
        # should probably reference).  Filter via wm.pi POS distribution:
        # keep Noun/Verb/Adj/Adv (POS channels 0..3), drop function words.
        import re as _re
        word_re = _re.compile(r"[a-zA-Z']+")
        a_tokens = [t.lower() for t in word_re.findall(a_text_clean)]
        content_pos = {0, 1, 2, 3}  # Noun, Verb, Adj, Adv
        seen = set()
        a_content_anchors = []
        for t in a_tokens:
            if t in seen or t not in self.pos_sel.wm.idx: continue
            i = self.pos_sel.wm.idx[t]
            pos_arg = int(self.pos_sel.wm.pi[i].argmax())
            if pos_arg in content_pos and len(t) >= 3:
                a_content_anchors.append(t); seen.add(t)

        # Also: Aisha-level reused anchors (if any fired in this response)
        aisha_words = [self.pos_sel.wm.lemmas[i] for i in b_idx]
        a_vocab = {t for t in a_tokens}
        aisha_reused = [w for w in aisha_words if w in a_vocab and w not in seen]

        structure = {
            "style": style or out.get("style", "?"),
            "target_register": {
                "v": round(out.get("target_v", 0.0), 3),
                "a": round(out.get("target_a", 0.0), 3)},
            "length": len(b_idx),
            "a_anchors": a_content_anchors,
            "aisha_reused_anchors": aisha_reused,
            "raw_word_stream": aisha_words,
            "pos_sequence": [CHANNEL_NAMES[self.pos_sel.wm.pi[i].argmax()]
                              for i in b_idx],
            "octant_sequence": [OCTANT_SHORT[int(self.layer2.oct8[i])]
                                 for i in b_idx],
            "phrases": [
                {"words": p.words,
                 "pos":    [CHANNEL_NAMES[self.pos_sel.wm.pi[i].argmax()]
                             for i in p.idx],
                 "octant": OCTANT_SHORT[p.dominant_octant],
                 "size":   p.size,
                 "register": {"v": round(p.mean_coord["v"], 2),
                                "a": round(p.mean_coord["a"], 2)}}
                for p in phrases
            ],
            "clauses": [
                {"phrases": [phrases.index(p) for p in c.phrases],
                 "total_words": c.total_words,
                 "coherence":   round(c.coherence, 2),
                 "dominant_octant":
                     OCTANT_SHORT[int(c.octant_hist.argmax())]}
                for c in clauses
            ],
            "ok": True,
        }
        return structure

    # ------------------------------------------------------------------
    def respond(self, user_input: str, style: str | None = None,
                 llm_fn: Callable[[str, dict], str] | None = None,
                 n_samples: int = 20, verbose: bool = False) -> dict:
        timings = {}
        t0 = time.time()
        # 1. Harper-in
        cleaned = hp.polish(user_input)
        timings["harper_in"] = time.time() - t0

        # 1a. Record user turn BEFORE building structure — so memory
        # summary reflects everything up to this turn.
        t_mem = time.time()
        user_style = style or self._infer_style_from_text(cleaned)
        self.memory.record("user", cleaned, user_style, self.axes)
        memory_summary = self.memory.summary_for_llm(n=6)
        timings["memory"] = time.time() - t_mem

        # 1b. Activate per-style POS bigram on the Responder so word-
        # selection uses style-specific syntax.  Set before build_structure
        # so the entire generation respects the routed style.
        self.pos_sel.R.set_style(user_style)

        # 1c. Memory-as-α perturbation: pull next response's (v,a) target
        # toward the conversation's recent trajectory.  Empty memory →
        # zero perturbation, falls back to standard style-centroid behavior.
        dv, da = self._memory_perturbation()
        self.pos_sel.R.set_memory_perturbation(dv, da)

        # 2. Aisha stack — use the SBERT-routed style (user_style), not
        # `style` arg.  Previously this just passed `style` which is None
        # for default calls, causing the Responder to fall back to its
        # internal (v,a)-centroid inference and ignore SBERT's verdict.
        t1 = time.time()
        structure = self.build_structure(cleaned, style=user_style,
                                            n_samples=n_samples)
        timings["aisha_stack"] = time.time() - t1

        # 2a. Clear memory perturbation so the NEXT respond() call computes
        # a fresh one from updated memory rather than reusing this one.
        self.pos_sel.R.set_memory_perturbation(0.0, 0.0)

        if not structure.get("ok"):
            return {"response": "", "structure": structure,
                     "timings": timings, "error": structure.get("reason")}

        structure["memory_summary"] = memory_summary
        structure["prior_turns"]    = len(self.memory.turns) - 1

        # 3. LM polish
        t2 = time.time()
        if llm_fn is None:
            llm_fn = MockLLM().__call__
        raw_response = llm_fn(cleaned, structure)
        timings["llm"] = time.time() - t2

        # 3a. Faithfulness gate.  If the LM invented content words that
        # weren't in Aisha's stream, throw the polish away and ship the
        # raw stream — never let LM-fabricated content reach the user.
        gate_ok, gate_reason = faithfulness_gate(
            raw_response, structure.get("raw_word_stream", []))
        if gate_ok:
            polish_input = raw_response
            gate_status = "ok"
        else:
            polish_input = _raw_fallback(structure.get("raw_word_stream", []))
            gate_status = f"rejected: {gate_reason}"

        # 4. Harper-out (cosmetic spelling/grammar polish — does not
        # introduce new content; safe to run on either branch).
        t3 = time.time()
        polished = hp.polish(polish_input)
        timings["harper_out"] = time.time() - t3

        # 5. Record aisha turn into memory
        self.memory.record("aisha", polished, structure["style"], self.axes)

        timings["total"] = time.time() - t0

        out = {
            "user_input":     user_input,
            "user_cleaned":   cleaned,
            "structure":      structure,
            "llm_raw":        raw_response,
            "gate":           gate_status,
            "response":       polished,
            "timings":        timings,
            "memory_summary": memory_summary,
        }
        if verbose:
            print(f"[pipeline] user_input: {user_input!r}")
            print(f"[pipeline] user_cleaned: {cleaned!r}")
            print(f"[pipeline] structure (summary):")
            print(f"  style={structure['style']}  length={structure['length']}")
            print(f"  anchors={structure['a_anchors']}")
            print(f"  phrases: {len(structure['phrases'])}  "
                  f"clauses: {len(structure['clauses'])}")
            print(f"[pipeline] llm_raw: {raw_response!r}")
            print(f"[pipeline] response: {polished!r}")
            print(f"[pipeline] timings: " + "  ".join(
                f"{k}={v:.3f}s" for k, v in timings.items()))
        return out


# ======================================================================
class MockLLM:
    """Placeholder — concatenates raw word stream w/ sentence capitalisation.
    Serves as a baseline: whatever LM we plug in should beat this trivially."""

    def __call__(self, user_input: str, structure: dict) -> str:
        words = structure.get("raw_word_stream", [])
        if not words: return ""
        words[0] = words[0].capitalize()
        return " ".join(words) + "."


class OllamaLLM:
    """Call a local Ollama server with a structured-prompt template.
    Default model is qwen2.5:0.5b — small enough for phone deployment
    while still chat-tuned.  Requires `ollama serve` running on :11434."""

    # Map our 5 styles to tone descriptors small LMs understand.
    # These describe the REPLY style, not the user's mood.  Heated =
    # firm-and-direct (standing ground), not dismissive — we're
    # responding to a distressed user, not starting a fight.
    STYLE_TONE = {
        "casual":     "friendly, relaxed, informal, warm",
        "civilized":  "polite, measured, thoughtful",
        "emotional":  "warm, empathetic, gentle, caring",
        "heated":     "firm and direct, standing your ground, honest — "
                      "but not mean or dismissive",
        "scientific": "precise, technical, matter-of-fact",
    }

    def __init__(self, model: str = "llama3.2:1b",
                  host: str = "http://localhost:11434",
                  temperature: float = 0.7,
                  max_tokens: int = 25):  # corpus median ~11 words
        self.model = model
        self.host = host
        self.temperature = temperature
        self.max_tokens = max_tokens

    def _build_prompt(self, user_input: str, structure: dict) -> str:
        """Polish prompt — the LM smooths Aisha's word stream into grammar.

        The LM is a polisher, not a generator: it must keep Aisha's words
        and order, fix grammar (articles, tense, auxiliaries, punctuation),
        and add only minimal connective tissue.  It must NOT invent new
        content nouns/verbs the manifold didn't emit.
        """
        words = structure.get("raw_word_stream", [])
        if not words:
            return ""
        style = structure.get("style", "casual")
        tone  = self.STYLE_TONE.get(style, "natural")
        word_stream = " ".join(words)

        prompt = (
            "You are a grammar polisher.  Aisha (a chat AI) emitted the "
            "following word stream as her reply:\n\n"
            f"    {word_stream}\n\n"
            "Polish this into ONE grammatical English sentence.  Rules:\n"
            "- Keep Aisha's words and their order as much as possible.\n"
            "- You may add small connective words (articles, 'to', 'and', "
            "auxiliaries) and adjust word forms (tense, plural).\n"
            "- You may drop a word if it breaks grammar; you may NOT add "
            "new content nouns or verbs that weren't in the stream.\n"
            "- Do not invent new topics, do not 'reply to the user' from "
            "scratch, do not add commentary.\n"
            f"- Tone: {tone}.\n\n"
            "Polished sentence:"
        )
        return prompt

    def __call__(self, user_input: str, structure: dict) -> str:
        import requests
        prompt = self._build_prompt(user_input, structure)
        try:
            r = requests.post(
                f"{self.host}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": self.temperature,
                        "num_predict": self.max_tokens,
                    },
                },
                timeout=60,
            )
            r.raise_for_status()
            response = r.json().get("response", "").strip()
            # Strip common trailing artefacts
            if response.startswith('"') and response.endswith('"'):
                response = response[1:-1]
            return response
        except Exception as e:
            return f"[LLM error: {e}]"


# ======================================================================
def _cli() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, required=True)
    ap.add_argument("--style", type=str, default=None,
                     choices=["casual","civilized","emotional","heated","scientific"])
    ap.add_argument("--samples", type=int, default=20)
    ap.add_argument("--structure-only", action="store_true")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--llm", type=str, default="ollama",
                     choices=["mock", "ollama"],
                     help="which LM to use")
    ap.add_argument("--model", type=str, default="llama3.2:1b",
                     help="Ollama model name")
    ap.add_argument("--memory-log", type=str, default=None,
                     help="persistent memory log path")
    args = ap.parse_args()

    memory_log = Path(args.memory_log) if args.memory_log else None
    p = Pipeline(memory_log=memory_log)
    if args.structure_only:
        cleaned = hp.polish(args.input)
        s = p.build_structure(cleaned, style=args.style,
                                n_samples=args.samples)
        print(json.dumps(s, indent=2, default=str))
    else:
        llm_fn = OllamaLLM(model=args.model) if args.llm == "ollama" else MockLLM()
        out = p.respond(args.input, style=args.style,
                         n_samples=args.samples, llm_fn=llm_fn, verbose=True)
        if args.json:
            print(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    _cli()
