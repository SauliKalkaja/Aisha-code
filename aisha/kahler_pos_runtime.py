"""Runtime for the POS-aligned Kähler.

Loads h.npz from data/processed/pos_kahler/ and exposes:
  ka.g_at(q_batch)              → (B, 8, 8) complex Hermitian metric
  ka.G_real_at(q_batch)         → (B, 16, 16) real symmetric metric
  ka.grad_K_at(q_batch)         → (B, 16) covariant gradient
  ka.contravariant_grad_K_at    → (B, 16) physical gradient
  ka.hamiltonian_flow_at        → (B, 16) symplectic ξ = J·∇K
  ka.mahalanobis_to_seed(seed, others) → (N,) Kähler distance²
  ka.normalize_q                → applies same x/X centering used in training

Where the manifold lives in q ∈ R^16 with z_k = q_k + i·q_{k+8}, k=0..7.
"""
import pickle
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
PROC = ROOT / "data" / "processed"
KDIR = PROC / "pos_kahler"


class POSKahlerRuntime:
    def __init__(self, manifold_path: Path | None = None):
        a = np.load(KDIR / "h.npz")
        self.h = a["h_re"] + 1j * a["h_im"]                # (K, K) Hermitian
        self.powers = a["powers"]                            # (K, 8) int
        self.x_mean = a["x_mean"]; self.x_std = a["x_std"]
        self.X_mean = a["X_mean"]; self.X_std = a["X_std"]
        self.K_basis = self.h.shape[0]
        self.max_p = int(self.powers.max())

        if manifold_path is None:
            manifold_path = PROC / "pos_manifold.pkl"
        with open(manifold_path, "rb") as f:
            M = pickle.load(f)
        self.manifold = M
        self.q_all = M["q"]                                  # (N, 16) real, training-ready
        self.lemmas = M["lemmas"]
        self.word_idx_orig = M["word_idx_orig"]
        self.m = M["m"]
        self.N = len(self.q_all)
        self.pos_names = M["pos_names"]
        # x and X separately for analysis
        self.x_norm = M["x"]
        self.X_norm = M["X"]

    # --------- monomial s and ds/dz ---------

    def _s_and_ds(self, q_norm: np.ndarray):
        """q_norm: (B, 16) → s (B, K), ds_z (B, K, 8) complex.

        z_k = q_norm[:, k] + i·q_norm[:, k+8].
        """
        z = q_norm[:, :8] + 1j * q_norm[:, 8:]               # (B, 8)
        zp = [np.ones_like(z)]
        for _ in range(self.max_p):
            zp.append(zp[-1] * z)
        B = z.shape[0]; K = self.K_basis
        s = np.ones((B, K), dtype=np.complex128)
        for c in range(8):
            col = np.stack([zp[int(p)][:, c] for p in self.powers[:, c]],
                              axis=0)
            s = s * col.T
        ds_z = np.zeros((B, K, 8), dtype=np.complex128)
        for mu in range(8):
            coefs = self.powers[:, mu].astype(np.float64)
            shifted = np.ones((B, K), dtype=np.complex128)
            for c in range(8):
                if c == mu:
                    cols = []
                    for p in self.powers[:, c]:
                        cols.append(zp[int(p) - 1][:, c] if p > 0
                                       else np.zeros_like(z[:, c]))
                    col = np.stack(cols, axis=0)
                else:
                    col = np.stack([zp[int(p)][:, c]
                                       for p in self.powers[:, c]], axis=0)
                shifted = shifted * col.T
            ds_z[:, :, mu] = coefs[None, :] * shifted
        return s, ds_z

    # --------- metric, gradient, flow ---------

    def g_at(self, q: np.ndarray) -> np.ndarray:
        """q (B, 16) already-normalized → g_αβ̄ (B, 8, 8) Hermitian PSD."""
        s, ds_z = self._s_and_ds(q)
        s_b = s.conj()
        ds_zb = ds_z.conj()
        Fs = np.einsum("nk,kl,nl->n", s, self.h, s_b).real
        F_z  = np.einsum("nkm,kl,nl->nm", ds_z, self.h, s_b)
        F_zb = np.einsum("nk,kl,nlm->nm", s, self.h, ds_zb)
        F_zzb = np.einsum("nkm,kl,nlp->nmp", ds_z, self.h, ds_zb)
        F1 = Fs[:, None, None]
        g = F_zzb / F1 - (F_z[:, :, None] * F_zb[:, None, :]) / (F1 * F1)
        return 0.5 * (g + g.conj().swapaxes(-1, -2))

    def G_real_at(self, q: np.ndarray) -> np.ndarray:
        """Convert g (B,8,8) Hermitian → real symmetric on R^16.
        G_real = ((A, -B), (B, A))  with g = A + iB."""
        g = self.g_at(q)
        A = g.real; Bm = g.imag
        Bn = q.shape[0]
        G = np.zeros((Bn, 16, 16), dtype=np.float64)
        G[:, :8, :8] = A
        G[:, 8:, 8:] = A
        G[:, :8, 8:] = -Bm
        G[:, 8:, :8] = Bm
        return 0.5 * (G + G.swapaxes(-1, -2))

    def grad_K_at(self, q: np.ndarray) -> np.ndarray:
        """∂K/∂q (B, 16) covariant.  K = log(s* H s)."""
        s, ds_z = self._s_and_ds(q)
        s_b = s.conj()
        Fs = np.einsum("nk,kl,nl->n", s, self.h, s_b).real
        F_z = np.einsum("nkm,kl,nl->nm", ds_z, self.h, s_b)         # (B, 8) complex
        dK_dz = F_z / Fs[:, None]
        out = np.empty((q.shape[0], 16), dtype=np.float64)
        out[:, :8] = 2.0 * dK_dz.real
        out[:, 8:] = -2.0 * dK_dz.imag
        return out

    def contravariant_grad_K_at(self, q: np.ndarray) -> np.ndarray:
        cov = self.grad_K_at(q)
        G = self.G_real_at(q)
        Ginv = np.linalg.inv(G)
        return np.einsum("nij,nj->ni", Ginv, cov)

    def hamiltonian_flow_at(self, q: np.ndarray) -> np.ndarray:
        """ξ = J · ∇^a K — symplectic flow, perpendicular to ∇K.
        In (x, y) coords: J · (a, b) = (-b, a)."""
        con = self.contravariant_grad_K_at(q)
        out = np.empty_like(con)
        out[:, :8] = -con[:, 8:]
        out[:, 8:] =  con[:, :8]
        return out

    # --------- distance ---------

    def mahalanobis_to_seed(self, q_seed: np.ndarray,
                                q_others: np.ndarray) -> np.ndarray:
        """d²(q_seed, q_others) under G_real(q_seed)."""
        G = self.G_real_at(q_seed[None, :])[0]
        dq = q_others - q_seed[None, :]
        return np.einsum("ni,ij,nj->n", dq, G, dq)
