"""
phase_A_shape_diagnostic.py — categorize cumulative-torsion shapes
for sentences across 3 corpora, per axis.  Phase A of the shape-
analysis roadmap.

Per sentence (content tokens only, len≥4):
  - cumulative trajectory of M_n, chi_n, spin_n
  - detrend (subtract linear fit)
  - categorize residual shape:
        flat       — range too small
        linear_up  — monotone rising (residual near zero)
        linear_dn  — monotone falling
        U          — single dip (concave-up residual)
        inv_U      — single peak (concave-down residual)
        W          — two dips
        M          — two peaks
        other      — multiple inflections, no clear shape

Output to /tmp/phase_A_shapes.txt.  Stats per (corpus × axis × shape).
"""
from __future__ import annotations

import csv
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from aisha_respond import Responder
from corpus_deep import tokenize

DATA = ROOT / "data"

SHAPES = ("flat", "linear_up", "linear_dn", "U", "inv_U", "W", "M", "other")
MIN_CONTENT = 4


def get_content_values(text: str, R: Responder):
    """Return (M_vec, chi_vec, spin_vec) for content tokens in sentence,
    or None if too few content tokens."""
    idx_all = tokenize(text, R.wm)
    idx = [i for i in idx_all if R.mask10k[i]
           and not R.is_stopword[i]]
    if len(idx) < MIN_CONTENT:
        return None
    arr = np.asarray(idx)
    return (R.M_n[arr], R.chi_n[arr], R.spin_n[arr])


def categorize_shape(values):
    """Categorize cumulative-trajectory shape for one axis.
    `values` is the per-word value sequence (we cumsum here)."""
    if len(values) < MIN_CONTENT:
        return "tiny"
    cum = np.cumsum(values)
    n = len(cum)
    x = np.arange(n, dtype=np.float64)

    # Range check
    cum_range = cum.max() - cum.min()
    if cum_range < 0.3:
        return "flat"

    # Detrend (subtract linear fit)
    slope, intercept = np.polyfit(x, cum, 1)
    fitted = slope * x + intercept
    residual = cum - fitted
    res_range = residual.max() - residual.min()

    # Strong monotonicity if residual is small relative to cumulative
    if res_range < cum_range * 0.25:
        # Mostly linear
        return "linear_up" if slope > 0 else "linear_dn"

    # Count sign changes in residual derivative — gives # of bumps
    diffs = np.diff(residual)
    if len(diffs) < 2:
        return "other"
    # Sign changes in derivative
    sign_changes = int(((diffs[:-1] * diffs[1:]) < 0).sum())

    # Use detrended profile to decide concavity at the middle
    mid_val = residual[n // 2]

    if sign_changes == 0:
        # No bump → near-linear
        return "linear_up" if slope > 0 else "linear_dn"
    elif sign_changes == 1:
        # One bump
        return "U" if mid_val < 0 else "inv_U"
    elif sign_changes == 2:
        return "W" if mid_val < 0 else "M"
    else:
        return "other"


def gather_sentences():
    sources = {}
    # conversations.csv
    sents = []
    with (DATA / "conversations.csv").open() as r:
        for row in csv.DictReader(r):
            t = (row.get("text") or "").strip()
            if 5 <= len(t) <= 280:
                sents.append(t)
            if len(sents) >= 1000:
                break
    sources["conversations"] = sents

    # opensubtitles
    sents = []
    with (DATA / "raw" / "opensubtitles_pilot.txt").open(
            encoding="utf-8", errors="ignore") as r:
        for line in r:
            line = line.strip()
            if 5 <= len(line) <= 280:
                sents.append(line)
            if len(sents) >= 1000:
                break
    sources["opensubtitles"] = sents

    # wikipedia paragraph-internal sentences
    sents = []
    SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")
    text = (DATA / "raw" / "wikipedia_pilot.txt").read_text(
        encoding="utf-8", errors="ignore")[:5_000_000]
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        for s in SENT_RE.split(line):
            s = s.strip()
            if 25 <= len(s) <= 280:
                sents.append(s)
                if len(sents) >= 1000:
                    break
        if len(sents) >= 1000:
            break
    sources["wikipedia"] = sents
    return sources


def main():
    out_path = Path("/tmp/phase_A_shapes.txt")
    R = Responder()
    sources = gather_sentences()

    # Counts: corpus → axis → shape → count
    results: dict = {}
    n_used: dict = {}
    n_skip: dict = {}
    for corpus, sents in sources.items():
        results[corpus] = {a: Counter() for a in ("M", "chi", "spin")}
        used = 0
        skipped = 0
        for s in sents:
            vals = get_content_values(s, R)
            if vals is None:
                skipped += 1
                continue
            M, C, S = vals
            results[corpus]["M"][categorize_shape(M)] += 1
            results[corpus]["chi"][categorize_shape(C)] += 1
            results[corpus]["spin"][categorize_shape(S)] += 1
            used += 1
        n_used[corpus] = used
        n_skip[corpus] = skipped

    # Write report
    with out_path.open("w") as f:
        f.write("Phase A — cumulative-torsion shape diagnostic\n")
        f.write("=" * 70 + "\n")
        f.write(f"min content tokens per sentence: {MIN_CONTENT}\n\n")
        for corpus in sources:
            f.write(f"--- {corpus.upper()}  used={n_used[corpus]} "
                    f"skipped={n_skip[corpus]} ---\n")
            for axis in ("M", "chi", "spin"):
                tot = sum(results[corpus][axis].values())
                f.write(f"  {axis}-axis  (n={tot}):\n")
                for shape in SHAPES:
                    c = results[corpus][axis].get(shape, 0)
                    pct = 100 * c / max(tot, 1)
                    bar = "#" * int(pct / 2)
                    f.write(f"    {shape:<11s}: {c:>4d}  {pct:>5.1f}%  {bar}\n")
            f.write("\n")

        # Cross-corpus comparison: what fraction of each shape comes
        # from each corpus?  Useful to see if shapes differ by genre.
        f.write("\n=== CROSS-CORPUS SHAPE FRACTIONS (per axis) ===\n")
        for axis in ("M", "chi", "spin"):
            f.write(f"\n  {axis}-axis:\n")
            f.write(f"    {'shape':<11s}  {'conv %':>7s}  {'opens %':>7s}  {'wiki %':>7s}\n")
            for shape in SHAPES:
                row = []
                for corpus in ("conversations", "opensubtitles", "wikipedia"):
                    tot = sum(results[corpus][axis].values())
                    c = results[corpus][axis].get(shape, 0)
                    row.append(100 * c / max(tot, 1))
                f.write(f"    {shape:<11s}  {row[0]:>6.1f}   {row[1]:>6.1f}   {row[2]:>6.1f}\n")

        # Verdict
        f.write("\n=== VERDICT ===\n")
        for axis in ("M", "chi", "spin"):
            # Use conversations as reference
            counts = results["conversations"][axis]
            tot = sum(counts.values())
            top = counts.most_common(1)
            top_pct = 100 * top[0][1] / max(tot, 1) if top else 0
            n_populated = sum(1 for s in SHAPES
                                if counts.get(s, 0) > tot * 0.05)
            verdict = ""
            if top_pct > 75:
                verdict = "COLLAPSED to one shape"
            elif n_populated < 3:
                verdict = "BIMODAL — only 2 shapes prominent"
            else:
                verdict = "SPREAD — useful signal"
            f.write(f"  {axis}-axis (conversations):  "
                    f"top shape {top_pct:.0f}%, populated shapes {n_populated}/{len(SHAPES)}"
                    f"  → {verdict}\n")

    print(f"done -> {out_path}")


if __name__ == "__main__":
    main()
