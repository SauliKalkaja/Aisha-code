"""
chat_demo.py — a scripted multi-turn chat to show memory in action.

Sends a sequence of user turns through the pipeline, each using the
shared memory instance.  Latency and memory summary are reported at
each step so we can verify:

  1. Latency stays flat across turns (memory summary is bounded-length).
  2. Memory summary captures mood trajectory and recurring topics.
  3. Aisha's responses reflect the accumulating context.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from pipeline import Pipeline, OllamaLLM    # noqa: E402


def main() -> None:
    # Fresh memory each run
    mem_log = Path("/tmp/aisha_chat_demo.jsonl")
    if mem_log.exists(): mem_log.unlink()

    pipe = Pipeline(memory_log=mem_log)
    llm = OllamaLLM(model="llama3.2:1b", temperature=0.6, max_tokens=25)

    script = [
        ("My dad just told me we're moving. Out of state.", "emotional"),
        ("I don't even know where to start packing.",        "emotional"),
        ("It's not fair. I didn't ask for any of this.",     "heated"),
        ("Sorry. I know I'm just venting. I'll be okay.",    "emotional"),
        ("Any advice for a long-distance move?",              "casual"),
    ]

    print()
    print("=" * 100)
    print("Multi-turn chat — memory flows across turns")
    print("=" * 100)

    for turn_i, (user_text, style) in enumerate(script):
        out = pipe.respond(user_text, style=style, llm_fn=llm,
                             n_samples=20, verbose=False)
        t = out["timings"]
        print(f"\n── Turn {turn_i + 1} ──────────────────────────────────────────────")
        print(f"USER:   {user_text}")
        print(f"AISHA:  {out['response']}")
        mem = out.get("memory_summary", "")
        if mem:
            print(f"(memory: {mem})")
        print(f"(t: total={t['total']:.2f}s  LM={t['llm']:.2f}s  "
              f"mem={t.get('memory', 0):.3f}s)")

    # Final memory inspection
    print()
    print("=" * 100)
    print("Memory at end of chat")
    print("=" * 100)
    for i, t in enumerate(pipe.memory.turns):
        print(f"  [{i:2d}] {t.speaker:<5s} {t.style:<10s} "
              f"v={t.signature['v']:+.2f}  a={t.signature['a']:+.2f}   "
              f"{t.text}")
    print(f"\nPhase jumps between turns:")
    for i, j in enumerate(pipe.memory.phase_jumps()):
        print(f"  turn {i} → {i+1}:  {j:.3f}")


if __name__ == "__main__":
    main()
