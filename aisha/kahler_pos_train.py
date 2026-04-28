"""Train Kähler potential on the POS-aligned manifold.

Donaldson form on 8 complex dims:
   z_k = x_k + i·X_k   (k=0..7, POS-axis aligned)
   K(z, z*) = log( s(z)*  H  s(z) )
   s(z) = monomial vector, multi-indices α with |α| ≤ DEG in 8 vars

Default DEG=3 → C(11,3) = 165 monomials.
H is 165×165 Hermitian PSD, parameterized via Cholesky factor L
(softplus on diagonal for positive definiteness).

Loss: MA — match log det g(q_w) to log ρ̂(q_w) (KDE on z weighted by m).

Output: data/processed/pos_kahler/h.npz
"""
import pickle
import time
from itertools import product
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
MANIFOLD = ROOT / "data" / "processed" / "pos_manifold.pkl"
OUTDIR = ROOT / "data" / "processed" / "pos_kahler"
OUTDIR.mkdir(parents=True, exist_ok=True)
LOG = OUTDIR / "train.log"

DEG = 3                  # max monomial degree in 8 complex vars


def log(msg: str):
    print(msg, flush=True)
    with open(LOG, "a") as f:
        f.write(msg + "\n")


# ---------------- data ----------------

def load_data():
    with open(MANIFOLD, "rb") as f:
        M = pickle.load(f)
    q = np.asarray(M["q"], dtype=np.float64)            # (N, 16) real
    m = np.asarray(M["m"], dtype=np.float64)            # (N,)
    return q, m, M


def kde_log_density(q: np.ndarray, m: np.ndarray,
                      bandwidth: float = 0.6, batch=256) -> np.ndarray:
    """Weighted Gaussian-KDE log-density at each point.  Centered to
    mean 0 (only ratios matter for det g matching)."""
    N = q.shape[0]
    w = m / m.sum()
    h2 = bandwidth ** 2
    log_w = np.log(w + 1e-30).astype(np.float32)
    qf = q.astype(np.float32)
    sq = (qf ** 2).sum(axis=1)
    out = np.empty(N, dtype=np.float64)
    for s in range(0, N, batch):
        e = min(s + batch, N)
        d2 = sq[s:e, None] + sq[None, :] - 2.0 * qf[s:e] @ qf.T
        d2 = np.maximum(d2, 0.0)
        lk = log_w[None, :] - d2 / (2 * h2)
        m_max = lk.max(axis=1, keepdims=True)
        out[s:e] = (m_max[:, 0] + np.log(np.exp(lk - m_max).sum(axis=1) + 1e-30)
                       ).astype(np.float64)
    out -= out.mean()
    return out


# ---------------- monomial basis ----------------

def monomial_powers(deg: int, n_complex: int = 8) -> np.ndarray:
    out = []
    for alpha in product(range(deg + 1), repeat=n_complex):
        if sum(alpha) <= deg:
            out.append(alpha)
    return np.asarray(out, dtype=np.int8)                    # (K, 8)


def z_monomials(q: torch.Tensor, powers_np: np.ndarray) -> torch.Tensor:
    """q (B, 16) → z^α (B, K) complex.  z_k = q[:, k] + i·q[:, k+8] for k=0..7."""
    x, y = q[:, :8], q[:, 8:]
    z = torch.complex(x, y)                                  # (B, 8)
    max_p = int(powers_np.max())
    ladders = [torch.ones_like(z)]
    for _ in range(max_p):
        ladders.append(ladders[-1] * z)
    K = powers_np.shape[0]
    out = torch.ones(q.shape[0], K, dtype=z.dtype, device=q.device)
    for c in range(8):
        col = torch.stack([ladders[int(p)][:, c] for p in powers_np[:, c]],
                              dim=0)                          # (K, B)
        out = out * col.transpose(0, 1)                       # (B, K)
    return out


class POSKahler(nn.Module):
    def __init__(self, deg: int = DEG):
        super().__init__()
        self.powers_np = monomial_powers(deg, 8)
        self.K = self.powers_np.shape[0]
        Lr = torch.eye(self.K) * 1.0
        Li = torch.zeros(self.K, self.K)
        self.L_re = nn.Parameter(Lr)
        self.L_im = nn.Parameter(Li)

    def L(self) -> torch.Tensor:
        Lr = self.L_re.tril(-1)
        Li = self.L_im.tril(-1)
        diag = F.softplus(torch.diagonal(self.L_re)) + 1e-3
        return torch.complex(Lr, Li) + torch.diag(
            torch.complex(diag, torch.zeros_like(diag)))

    def h(self) -> torch.Tensor:
        L = self.L()
        return L @ L.conj().transpose(-1, -2)

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        z_a = z_monomials(q, self.powers_np)
        H = self.h()
        Hs = z_a @ H.transpose(-1, -2)
        quad = (Hs * z_a.conj()).sum(dim=-1).real
        return torch.log(quad + 1e-12)


# -------- complex Hessian via autograd, real → (z, z̄) projection --------

def kahler_metric(K_fn, q: torch.Tensor) -> torch.Tensor:
    q = q.detach().requires_grad_(True)
    Kv = K_fn(q)
    grads = torch.autograd.grad(Kv.sum(), q, create_graph=True)[0]
    rows = []
    for i in range(16):
        Hi = torch.autograd.grad(grads[:, i].sum(), q, create_graph=True)[0]
        rows.append(Hi)
    H = torch.stack(rows, dim=1)                              # (B, 16, 16)
    Hxx, Hyy = H[:, :8, :8], H[:, 8:, 8:]
    Hxy, Hyx = H[:, :8, 8:], H[:, 8:, :8]
    g_re = 0.25 * (Hxx + Hyy)
    g_im = 0.25 * (Hxy - Hyx)
    return torch.complex(g_re, g_im)                          # (B, 8, 8)


def hermitize(g):  return 0.5 * (g + g.conj().transpose(-1, -2))


def logdet_complex(g, eps=1e-8):
    sign, logabs = torch.linalg.slogdet(
        g + eps * torch.eye(g.shape[-1], dtype=g.dtype, device=g.device))
    return logabs.real, sign.real


def min_eig(g):
    return torch.linalg.eigvalsh(hermitize(g))[..., 0]


# ---------------- training ----------------

def main():
    LOG.write_text("")
    log(f"[pos-kahler] DEG={DEG} → "
         f"{monomial_powers(DEG).shape[0]} monomials in 8 complex vars")
    q_np, m_np, M = load_data()
    log(f"[pos-kahler] {len(q_np)} words from pos_manifold.pkl")

    log("[pos-kahler] computing KDE log-density target …")
    log_rho = kde_log_density(q_np, m_np, bandwidth=0.6)
    log(f"[pos-kahler] log_rho μ={log_rho.mean():.3f}  σ={log_rho.std():.3f}  "
         f"range=[{log_rho.min():.3f}, {log_rho.max():.3f}]")
    np.savez(OUTDIR / "density_norm.npz", log_rho=log_rho)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"[pos-kahler] device: {device}"
         + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""))
    Q = torch.from_numpy(q_np).float().to(device)
    LR = torch.from_numpy(log_rho).float().to(device)

    model = POSKahler(deg=DEG).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=5e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=6000)

    N = len(Q); BATCH = 384; STEPS = 6000; LOG_EVERY = 200
    best = float("inf")
    log("[pos-kahler] training H …")
    t0 = time.time()
    for step in range(STEPS):
        idx = torch.randint(0, N, (BATCH,), device=device)
        q_b, lr_b = Q[idx], LR[idx]
        g = hermitize(kahler_metric(model, q_b))
        ld, _ = logdet_complex(g)
        loss = ((ld - ld.mean()) - (lr_b - lr_b.mean())).pow(2).mean()
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step(); sched.step()

        if step % LOG_EVERY == 0 or step == STEPS - 1:
            with torch.no_grad():
                me = min_eig(g)
                pd = (me > 0).float().mean().item()
                me_min = me.min().item()
            log(f"  step {step:5d}  loss={loss.item():.4f}  "
                  f"PD frac={pd:.3f}  min_eig_min={me_min:+.2e}  "
                  f"t={time.time()-t0:.1f}s")
            if loss.item() < best:
                best = loss.item()
                with torch.no_grad():
                    h_np = model.h().detach().cpu().numpy()
                np.savez(OUTDIR / "h.npz",
                          h_re=h_np.real, h_im=h_np.imag,
                          powers=model.powers_np,
                          x_mean=M["x_mean"], x_std=M["x_std"],
                          X_mean=M["X_mean"], X_std=M["X_std"],
                          step=step, loss=float(loss.item()))
    log(f"[pos-kahler] done in {time.time()-t0:.1f}s   best={best:.4f}")
    log(f"[pos-kahler] saved h.npz with {model.K}×{model.K} Hermitian h")


if __name__ == "__main__":
    main()
