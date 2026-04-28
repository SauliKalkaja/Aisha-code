"""
corpus_deep.py — deeper analysis of conversations.csv using the fixed
manifold's torsion axes.

Beyond the 8 octants, we examine:
  (1)  seven scalar AXIS COMBINATIONS per word:
       M, χ, s, M+χ, M+s, χ+s, M+χ+s
  (2)  EIGHT per-word physical quantities:
       α̃ = log|α| (amplitude)
       ε   = 0.5 β² − μ̄/ᾱ (Kepler energy, mean over channels)
       r   = |q|₂ (Keplerian radius)
       e_w = √(M² + χ² + s²) (torsion "eccentricity")
  (3)  JUMP statistics over each conversation:
       |Δε_t| between consecutive words, per style
  (4)  A→B response patterns beyond dominant-octant:
       B's first content-word octant conditioned on A's last content-word

Generates multi-panel plots and a text summary.
"""
from __future__ import annotations

import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from word_manifold import WordManifold             # noqa: E402


OCTANT_SHORT = {0:"cV", 1:"cog", 2:"rel", 3:"coll",
                 4:"abs", 5:"dei", 6:"stat", 7:"cat"}
POS_NAMES = ["NOUN","VERB","ADJ","ADV","DET","ADP","CONJ","INTJ"]
_WORD_RE = re.compile(r"[a-zA-Z']+")
_ALLOWED = re.compile(r"^[a-z]{2,12}(?:[-'][a-z]{1,8})?$")

CONTENT_OCTANTS = {0, 1, 2, 3, 6, 7}   # non-deictic, non-abstract-operator


def tokenize(text: str, wm) -> list[int]:
    return [wm.idx[t] for t in _WORD_RE.findall(text.lower())
             if t in wm.idx]


def build_axes(wm):
    alpha = np.load(ROOT / "alpha_fixed.npz")["alpha"]
    mdat = np.load(ROOT / "m_fixed.npz")
    M = mdat["M"]; spin = mdat["spin"]; H_til = mdat["H_til"]
    beta = mdat["beta"]

    passers = np.array([bool(_ALLOWED.fullmatch(l)) for l in wm.lemmas])
    idx5k = np.where(passers)[0][np.argsort(-wm.m[passers])][:5000]
    mask5k = np.zeros(wm.N, dtype=bool); mask5k[idx5k] = True

    phase = (spin * beta)[:, None] * H_til
    q = alpha.astype(np.complex128) * np.exp(1j * phase)
    q_u = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-30)
    q_common = q_u[mask5k]
    chi = (q_u.conj() @ q_common.T).real.mean(axis=1)

    # Center each axis on its 5k-vocab median for symmetry
    M_c = M - np.median(M[mask5k])
    chi_c = chi - np.median(chi[mask5k])
    spin_c = spin - np.median(spin[mask5k])

    # Normalise to unit std on 5k vocab for fair combination
    M_n   = M_c / max(np.std(M_c[mask5k]), 1e-9)
    chi_n = chi_c / max(np.std(chi_c[mask5k]), 1e-9)
    spin_n = spin_c / max(np.std(spin_c[mask5k]), 1e-9)

    axes = {
        "M":      M_n,
        "chi":    chi_n,
        "s":      spin_n,
        "M+chi":  M_n + chi_n,
        "M+s":    M_n + spin_n,
        "chi+s":  chi_n + spin_n,
        "M+chi+s": M_n + chi_n + spin_n,
    }

    # Physical scalars per word
    mu_mean = wm.mu.mean()
    alpha_mean = alpha.mean(axis=1)
    eps_w = 0.5 * beta * beta - mu_mean / np.maximum(alpha_mean, 1e-30)
    r_w   = np.linalg.norm(q, axis=1)        # |q|  magnitude
    ecc   = np.sqrt(M_n**2 + chi_n**2 + spin_n**2)   # torsion "eccentricity"

    phys = {
        "log_alpha": np.log(np.maximum(alpha_mean, 1e-30)),
        "beta":      beta,
        "eps":       eps_w,
        "r":         r_w,
        "ecc":       ecc,
    }

    return axes, phys, mask5k, q_u, M, chi, spin


def compute_octants(M, chi, spin, mask5k):
    """Same definition as octant_conversations.py."""
    M_med = np.median(M[mask5k])
    chi_med = np.median(chi[mask5k])
    spin_med = np.median(spin[mask5k])
    sM = (M > M_med).astype(int)
    sC = (chi > chi_med).astype(int)
    sS = (spin > spin_med).astype(int)
    return 4 * sM + 2 * sC + sS


def main() -> None:
    print("[deep] loading manifold + fixed params…", flush=True)
    wm = WordManifold.load(str(ROOT / "data" / "processed" / "manifold.pkl"))
    axes, phys, mask5k, q_u, M, chi, spin = build_axes(wm)
    octants = compute_octants(M, chi, spin, mask5k)

    print("[deep] loading conversations…", flush=True)
    rows = []
    with open(ROOT / "data" / "conversations.csv") as f:
        for row in csv.DictReader(f):
            idx = tokenize(row["text"], wm)
            if not idx:
                continue
            rows.append({
                "conv_id": int(row["conv_id"]),
                "turn_id": int(row["turn_id"]),
                "speaker": row["speaker"],
                "style":   row["style"],
                "text":    row["text"],
                "idx":     idx,
            })
    print(f"[deep] {len(rows):,} sentences", flush=True)

    styles = sorted(set(r["style"] for r in rows))

    # =====================================================================
    # 1. Axis-combination means per style
    # =====================================================================
    print()
    print("=" * 84)
    print("1.  per-sentence mean of each axis combination, averaged by style")
    print("=" * 84)
    header = "style        " + "  ".join(f"{k:>8s}" for k in axes.keys())
    print(header)
    for style in styles:
        means = {}
        for k, axis in axes.items():
            vals = []
            for r in rows:
                if r["style"] == style:
                    vals.append(axis[r["idx"]].mean())
            means[k] = float(np.mean(vals))
        line = f"{style:<12s} " + "  ".join(f"{means[k]:+8.3f}" for k in axes.keys())
        print(line)

    # =====================================================================
    # 2. Per-word Energy ε  — per-style distributions + jump statistics
    # =====================================================================
    print()
    print("=" * 84)
    print("2.  ENERGY ε jumps |ε_{t+1} − ε_t| within sentence, per style")
    print("=" * 84)
    eps = phys["eps"]
    jump_stats_by_style: dict[str, list[float]] = defaultdict(list)
    word_eps_by_style: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        idx = r["idx"]
        es = eps[idx]
        word_eps_by_style[r["style"]].extend(es.tolist())
        if len(es) >= 2:
            jumps = np.abs(np.diff(es))
            jump_stats_by_style[r["style"]].extend(jumps.tolist())
    print(f"  {'style':<12s}  {'mean ε':>10s}  {'mean |Δε|':>10s}  "
          f"{'p95 |Δε|':>10s}  {'max |Δε|':>10s}")
    for style in styles:
        es = np.array(word_eps_by_style[style])
        js = np.array(jump_stats_by_style[style])
        print(f"  {style:<12s}  {es.mean():>10.2e}  {js.mean():>10.2e}  "
              f"{np.percentile(js, 95):>10.2e}  {js.max():>10.2e}")

    # =====================================================================
    # 3. Eccentricity (torsion-norm) per word — same style breakdown
    # =====================================================================
    print()
    print("=" * 84)
    print("3.  TORSION ECCENTRICITY √(M²+χ²+s²) jumps, per style")
    print("=" * 84)
    ecc = phys["ecc"]
    e_jump_by_style: dict[str, list[float]] = defaultdict(list)
    word_e_by_style: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        es = ecc[r["idx"]]
        word_e_by_style[r["style"]].extend(es.tolist())
        if len(es) >= 2:
            e_jump_by_style[r["style"]].extend(np.abs(np.diff(es)).tolist())
    print(f"  {'style':<12s}  {'mean e':>10s}  {'mean Δe':>10s}  "
          f"{'p95 Δe':>10s}  {'max Δe':>10s}")
    for style in styles:
        es = np.array(word_e_by_style[style])
        js = np.array(e_jump_by_style[style])
        print(f"  {style:<12s}  {es.mean():>10.3f}  {js.mean():>10.3f}  "
              f"{np.percentile(js, 95):>10.3f}  {js.max():>10.3f}")

    # =====================================================================
    # 4. A→B first content-word octant transition
    # =====================================================================
    print()
    print("=" * 84)
    print("4.  A→B  FIRST CONTENT-WORD octant  (skip deictic/abstract openers)")
    print("=" * 84)
    def first_content(idx):
        for i in idx:
            if int(octants[i]) in CONTENT_OCTANTS:
                return int(octants[i])
        return None
    def last_content(idx):
        for i in reversed(idx):
            if int(octants[i]) in CONTENT_OCTANTS:
                return int(octants[i])
        return None

    by_conv = defaultdict(list)
    for r in rows:
        by_conv[r["conv_id"]].append(r)
    for cid in by_conv:
        by_conv[cid].sort(key=lambda x: x["turn_id"])

    trans = np.zeros((8, 8), dtype=float)
    pair_count = 0
    for cid, turns in by_conv.items():
        for i in range(len(turns) - 1):
            a, b = turns[i], turns[i + 1]
            if a["speaker"] != "A" or b["speaker"] != "B":
                continue
            a_last = last_content(a["idx"])
            b_first = first_content(b["idx"])
            if a_last is not None and b_first is not None:
                trans[a_last, b_first] += 1
                pair_count += 1
    row_sum = trans.sum(axis=1, keepdims=True)
    trans_p = trans / np.maximum(row_sum, 1)
    print(f"  pairs found: {pair_count}")
    print(f"       " + "  ".join(f"{OCTANT_SHORT[k]:>4s}" for k in range(8)))
    for a in range(8):
        if row_sum[a, 0] < 3:
            continue                                 # skip sparse rows
        row = "  ".join(f"{trans_p[a, b]:.2f}" for b in range(8))
        print(f"  {OCTANT_SHORT[a]:<4s}  {row}   (n={int(row_sum[a,0])})")

    # =====================================================================
    # Plots
    # =====================================================================
    fig, axes_fig = plt.subplots(2, 3, figsize=(18, 11))

    # 1. Axis-combo means per style
    ax = axes_fig[0, 0]
    xs = np.arange(len(axes))
    width = 0.15
    for i, style in enumerate(styles):
        vals = []
        for k in axes:
            s_vals = [axes[k][r["idx"]].mean()
                      for r in rows if r["style"] == style]
            vals.append(float(np.mean(s_vals)))
        ax.bar(xs + i * width, vals, width, label=style)
    ax.set_xticks(xs + 2 * width)
    ax.set_xticklabels(list(axes.keys()), rotation=30, fontsize=8)
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.set_title("per-sentence mean of axis combinations, by style")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    # 2. Energy jump histograms per style
    ax = axes_fig[0, 1]
    all_j = np.concatenate([np.array(jump_stats_by_style[s]) for s in styles])
    bins = np.linspace(0, np.percentile(all_j, 99), 60)
    for s in styles:
        ax.hist(jump_stats_by_style[s], bins=bins, alpha=0.4,
                 label=s, density=True)
    ax.set_title(r"|$\Delta\varepsilon$| distribution per style")
    ax.set_xlabel(r"|$\Delta\varepsilon_t$|")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # 3. Eccentricity jump histograms
    ax = axes_fig[0, 2]
    all_e = np.concatenate([np.array(e_jump_by_style[s]) for s in styles])
    bins = np.linspace(0, np.percentile(all_e, 99), 60)
    for s in styles:
        ax.hist(e_jump_by_style[s], bins=bins, alpha=0.4, label=s,
                 density=True)
    ax.set_title(r"$|\Delta e|$  torsion-eccentricity jumps per style")
    ax.set_xlabel(r"|$\Delta e_t$|")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # 4. A→B first-content-word transition matrix (upper-left non-sparse)
    ax = axes_fig[1, 0]
    im = ax.imshow(trans_p, cmap="Blues", vmin=0, vmax=0.6)
    ax.set_xticks(range(8)); ax.set_yticks(range(8))
    ax.set_xticklabels([OCTANT_SHORT[k] for k in range(8)], rotation=45, fontsize=9)
    ax.set_yticklabels([OCTANT_SHORT[k] for k in range(8)], fontsize=9)
    ax.set_xlabel("B first-content octant"); ax.set_ylabel("A last-content")
    ax.set_title("A-last-content → B-first-content (normalised rows)")
    for a in range(8):
        for b in range(8):
            v = trans_p[a, b]
            if v > 0.05:
                ax.text(b, a, f"{v:.2f}", ha="center", va="center",
                        fontsize=7,
                        color="white" if v > 0.3 else "black")
    fig.colorbar(im, ax=ax, fraction=0.045)

    # 5. Energy trajectory example per style
    ax = axes_fig[1, 1]
    for style in styles:
        # Find one longer conversation of this style
        picked = None
        for cid, turns in by_conv.items():
            if turns[0]["style"] == style and len(turns) >= 20:
                picked = turns[:20]
                break
        if not picked:
            continue
        eps_trajectory = []
        for t in picked:
            eps_trajectory.append(eps[t["idx"]].mean())
        ax.plot(range(len(eps_trajectory)), eps_trajectory,
                label=style, marker="o", markersize=3, linewidth=1.0)
    ax.set_title(r"Per-turn mean $\varepsilon$ trajectory (one conv per style)")
    ax.set_xlabel("turn"); ax.set_ylabel(r"$\bar\varepsilon$")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # 6. Torsion eccentricity trajectory per style
    ax = axes_fig[1, 2]
    for style in styles:
        picked = None
        for cid, turns in by_conv.items():
            if turns[0]["style"] == style and len(turns) >= 20:
                picked = turns[:20]; break
        if not picked:
            continue
        e_traj = [ecc[t["idx"]].mean() for t in picked]
        ax.plot(range(len(e_traj)), e_traj, label=style,
                marker="o", markersize=3, linewidth=1.0)
    ax.set_title("Per-turn mean torsion eccentricity, one conv per style")
    ax.set_xlabel("turn"); ax.set_ylabel(r"$\bar e$")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    out = ROOT / "images" / "corpus_deep.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print()
    print(f"[deep] plot → {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
