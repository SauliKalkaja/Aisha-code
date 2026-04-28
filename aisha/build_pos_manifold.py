"""Build the POS-aligned 6D-style manifold (8 real + 8 imaginary).

Real coords:  x_k(w) = normalized pos_counts[w, k]              (POS role)
Imag coords:  X_k(w) = P(next=k|w) − P(prev=k|w)               (POS shift)

Both centered + normalized for Kähler-training stability.  Saves as
data/processed/pos_manifold.pkl with the same skeleton as the existing
manifold so the trainer/runtime can swap cleanly.
"""
import pickle
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
PROC = ROOT / "data" / "processed"


def main():
    print("[build] loading source manifold…", flush=True)
    with open(PROC / "manifold_clean.pkl", "rb") as f:
        M = pickle.load(f)
    lemmas = M["lemmas"]
    pos_counts = np.asarray(M["pos_counts"], dtype=np.float64)         # (N, 8)
    trigrams = np.asarray(M["trigrams"], dtype=np.float64)             # (N, 8, 8)
    m = np.asarray(M["m"], dtype=np.float64).real

    keep = (m > 0)
    valid = np.where(keep)[0]
    N = len(valid)
    print(f"[build] {N} words with m>0", flush=True)

    # ---- raw coordinates ----
    pc = pos_counts[valid]
    pc_total = np.maximum(pc.sum(axis=1, keepdims=True), 1e-12)
    x_raw = pc / pc_total                                                # (N, 8) sums to 1

    tg = trigrams[valid]                                                  # (N, 8, 8)
    incoming = tg.sum(axis=2)
    outgoing = tg.sum(axis=1)
    total = np.maximum(incoming.sum(axis=1, keepdims=True), 1e-12)
    X_raw = (outgoing - incoming) / total                                 # (N, 8) sums to 0

    # ---- center + normalize for Kähler training ----
    # Subtract mean (for x — X already has mean ≈ 0 by construction).
    # Scale each axis to unit standard deviation.
    x_mean = x_raw.mean(axis=0)
    x_std = x_raw.std(axis=0) + 1e-9
    x = (x_raw - x_mean) / x_std

    X_mean = X_raw.mean(axis=0)
    X_std = X_raw.std(axis=0) + 1e-9
    X = (X_raw - X_mean) / X_std

    # Combined real 16-vector for training: q = [x, X]
    q = np.concatenate([x, X], axis=1)                                   # (N, 16)
    z = x + 1j * X                                                        # (N, 8) complex

    print(f"[build] coordinates ready")
    print(f"  x:  μ={x.mean():.3e}  σ={x.std():.3f}  range=[{x.min():.2f},{x.max():.2f}]")
    print(f"  X:  μ={X.mean():.3e}  σ={X.std():.3f}  range=[{X.min():.2f},{X.max():.2f}]")
    print(f"  q:  shape={q.shape}  norm μ={np.linalg.norm(q,axis=1).mean():.2f}")
    print(f"  z:  shape={z.shape}  |z| μ={np.abs(z).mean():.2f}")

    # ---- save manifold pickle ----
    new_M = {
        "lemmas": [lemmas[i] for i in valid],
        "word_idx_orig": valid,           # back-pointer to original manifold
        "m": m[valid],
        "x_raw": x_raw, "X_raw": X_raw,
        "x": x, "X": X,
        "q": q,                           # (N, 16) real, training-ready
        "z": z,                           # (N, 8) complex view
        "x_mean": x_mean, "x_std": x_std,
        "X_mean": X_mean, "X_std": X_std,
        "pos_names": ["NOUN","VERB","ADJ","ADV","PRON_DET","PREP","CONJ","INTJ"],
        # carry through useful pre-computed pieces
        "pos_counts": pc,
        "trigrams": tg,
    }
    out_path = PROC / "pos_manifold.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(new_M, f)
    size_mb = out_path.stat().st_size / 1e6
    print(f"\n[saved] {out_path}  ({size_mb:.1f} MB)")
    print(f"        N={N} words, q_dim=16 real (8 POS + 8 POS-shift)")


if __name__ == "__main__":
    main()
