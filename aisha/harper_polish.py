"""
Aisha's Harper wrapper — rule-based grammar/spelling polish via
Automattic's harper-cli.

Binary lives in /home/sale/Aisha_backup/Aisha_syntax/tools/harper/ —
we just point at it directly.  If missing / unreachable, polish() is a
no-op.

Uses span-based replacement of Harper's suggested fixes: leftmost-first,
non-overlapping, right-to-left edit application.
"""
from __future__ import annotations

import json
import pathlib
import re
import subprocess

HARPER = pathlib.Path(
    "/home/sale/Aisha_backup/Aisha_syntax/tools/harper/harper-cli")

_REPLACE_RE = re.compile(r"Replace with:\s*[“\"](.+?)[”\"]")
_REMOVE_RE  = re.compile(r"Remove\b", re.I)


def _parse_suggestion(text: str):
    m = _REPLACE_RE.search(text)
    if m: return ("replace", m.group(1))
    if _REMOVE_RE.match(text.strip()): return ("remove", "")
    m = re.match(r"Insert\s*[“\"](.+?)[”\"]", text.strip())
    if m: return ("insert", m.group(1))
    return None


def polish(text: str, timeout: float = 5.0,
            only_rules: set[str] | None = None) -> str:
    """Run `text` through Harper and apply all suggested fixes.
    Returns polished text.  Safe: returns input unchanged on any error."""
    if not text or not text.strip():
        return text
    if not HARPER.exists():
        return text
    try:
        proc = subprocess.run(
            [str(HARPER), "lint", "--format", "json", "--no-color"],
            input=text, capture_output=True, text=True, timeout=timeout)
    except Exception:
        return text
    out = proc.stdout
    if not out.strip():
        return text
    try:
        data = json.loads(out)
    except Exception:
        start = out.find("[")
        if start < 0: return text
        try:
            data = json.loads(out[start:])
        except Exception:
            return text

    lints = []
    for f in data:
        for l in f.get("lints", []):
            if only_rules and l.get("rule") not in only_rules: continue
            if not l.get("suggestions"): continue
            lints.append(l)
    if not lints:
        return text

    lints.sort(key=lambda l: l["span"]["char_start"])
    non_overlap = []; last_end = -1
    for l in lints:
        s = l["span"]["char_start"]
        if s < last_end: continue
        non_overlap.append(l); last_end = l["span"]["char_end"]

    new_text = text
    for l in sorted(non_overlap, key=lambda l: l["span"]["char_start"],
                     reverse=True):
        parsed = _parse_suggestion(l["suggestions"][0])
        if parsed is None: continue
        action, repl = parsed
        s, e = l["span"]["char_start"], l["span"]["char_end"]
        if action == "replace":
            new_text = new_text[:s] + repl + new_text[e:]
        elif action == "remove":
            end = e
            if end < len(new_text) and new_text[end] == " ":
                end += 1
            new_text = new_text[:s] + new_text[end:]
        elif action == "insert":
            new_text = new_text[:s] + repl + new_text[s:]
    return new_text


def lints(text: str, timeout: float = 5.0) -> list[dict]:
    """Return the raw list of Harper lint records for `text`.  Empty on error."""
    if not text or not HARPER.exists(): return []
    try:
        proc = subprocess.run(
            [str(HARPER), "lint", "--format", "json", "--no-color"],
            input=text, capture_output=True, text=True, timeout=timeout)
    except Exception:
        return []
    out = proc.stdout
    if not out.strip(): return []
    try:
        data = json.loads(out)
    except Exception:
        start = out.find("[")
        if start < 0: return []
        try: data = json.loads(out[start:])
        except Exception: return []
    res = []
    for f in data:
        for l in f.get("lints", []):
            res.append(l)
    return res


if __name__ == "__main__":
    tests = [
        "She go to the store.",
        "He eat apples every day.",
        "I are tired.",
        "The cat is on mat.",          # "on the mat" — Harper catches some
        "We watch TV together.",        # clean
    ]
    for t in tests:
        out = polish(t)
        print(f"IN:   {t!r}")
        print(f"OUT:  {out!r}")
        ls = lints(t)
        print(f"      rules: {[l['rule'] for l in ls]}")
        print()
