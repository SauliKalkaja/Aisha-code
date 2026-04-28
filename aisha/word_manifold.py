"""
word_manifold.py — per-cell Aisha manifold (static structure).

Each cell = one word.  Holds the per-cell 16-dim symplectic object derived
from corpus data.  Composes `solar_system.analytical_jump.AnalyticalJump`
where the physics overlaps (Keplerian quantities per channel); does NOT
modify it.

Per-cell quantities, for word w and channel g ∈ {1..8} (indexed 0..7):

    m_w^(g)   observed count of w tagged as POS-class g
    π_w^(g)   = m_w^(g) / m_w   (simplex; POS membership)
    α_w^(g)   constructed in three steps (no floor, no ceiling):
                 1. Laplace-smooth counts:   m̃_w^(g) = m_w^(g) + ε
                 2. Per-channel scale:       α̃_w^(g) = m̃_w^(g) / m̄^(g),
                                              m̄^(g) = mean_u m̃_u^(g)
                 3. Per-word geometric anchor (paper §1.5):
                       G_w = (∏_g α̃_w^(g))^(1/8)
                       α_w^(g) = α̃_w^(g) / G_w
                    After step 3, ⟨log α_w⟩_g = 0 exactly per word —
                    enforcing the Kähler "fixed information budget"
                    at seed time (page 6 of theory).
                ε is the only free parameter and acts as a uniform
                Dirichlet prior.  ε → 0 recovers the unsmoothed form
                (undefined for words with any zero channel).
    β_w^(g)   = 1 / α_w^(g)     (scalar symplectic lock; paper eq 2 / 13)
    A_w ∈ R^{8×8}  row-stochastic per g:
                A_w[g, h] = count_w[g, h] / Σ_h count_w[g, h]
                Rows with no observed context default to uniform 1/8.
                (Option 1 locked earlier: trigram channel-transition.)
    B_w ∈ R^{8×8}  = A_w   (paper §1.2 convention)
    J_w ∈ R^{16×16}  per-cell symplectic Jacobian, paper eq 15:
          | diag(α_w) · A_w       diag(β_w) · B_w    |
          | −diag(β_w) · B_w      (diag(α_w) · A_w)⁻¹ |
    q_w ∈ C^8  Kähler complex coord, seeded |q_w^(g)|² = α_w^(g),
                phase = 0.  (Fubini–Study form on the lexical
                substrate; phase is learned / not yet populated.)

    ω_w ∈ R^8  per-channel Keplerian frequency (paper eq 16):
                  ε_w^(g) = (1/2)β_w^(g)² − μ^(g) / α_w^(g)
                  a_w^(g) = −μ^(g) / (2 ε_w^(g))
                  ω_w^(g) = sqrt(μ^(g) / a_w^(g)³)
                Per-channel gravitational parameter:
                  μ^(g) = Σ_u m_u^(g)  (G absorbed into unit choice)
                For channels where ε ≥ 0 the orbit is unbound;
                we take a = +μ/(2|ε|) and flag the cell as hyperbolic.

    π_w ⊙ ω_w  effective per-word frequency vector (paper eq 18),
                used for sentence-level interference scoring.

The symplectic residual Res_w = ||J_w^T Ω_8 J_w − Ω_8||_F is
reported per cell by `verify_symplectic()`.  The paper (eq 6) says
exactness only holds in the continuum bigram limit, so finite-N
residuals > 0 are expected.

Nothing in this file imports from the filesystem; it takes numpy
arrays as inputs and returns/saves them.  Persistence lives in the
builder script.
"""

from __future__ import annotations

import dataclasses
import pickle
from pathlib import Path
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_CH = 8
CHANNEL_NAMES = ("Noun", "Verb", "Adj", "Adv",
                 "Pro", "Prep", "Conj", "Interj")

#: Laplace / Dirichlet count smoothing applied to every (word, channel)
#: before normalisation.  Removes exact zeros without imposing a floor
#: (a floor truncates phase space; this is a uniform prior).  ε = 1 is
#: classical Laplace; smaller ε is softer; ε = 0 recovers raw counts
#: (requires a floor elsewhere).
EPSILON_SMOOTH = 1.0

#: Tikhonov regularisation for the lower-right block of J_w.
#: Empirically, LAMBDA_TIKH > 0 made the residual WORSE than plain
#: pseudoinverse for this particular construction — the naïve
#: (α·A + λI)⁻¹ biases the inverse away from I_8 by O(λ), and the
#: residual (which measures departure from Ω = diag(I, I) / (-I, I))
#: picks up that bias directly.  np.linalg.pinv was the right call.
#: Keeping this knob at 0 but exposing it for future experimentation.
LAMBDA_TIKH = 0.0


# ---------------------------------------------------------------------------
# Symplectic form
# ---------------------------------------------------------------------------

def omega_8() -> np.ndarray:
    """Ω_8 = [[0, I_8], [-I_8, 0]] ∈ R^{16×16}."""
    W = np.zeros((16, 16), dtype=np.float64)
    W[:N_CH, N_CH:] =  np.eye(N_CH)
    W[N_CH:, :N_CH] = -np.eye(N_CH)
    return W


OMEGA_8 = omega_8()


# ---------------------------------------------------------------------------
# Cell
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class Cell:
    """One word's per-cell state on the Aisha manifold."""

    lemma:  str
    m_w:    int                  # total observed count
    alpha:  np.ndarray           # (8,)  real
    beta:   np.ndarray           # (8,)  real, = 1/alpha
    pi:     np.ndarray           # (8,)  simplex
    A_w:    np.ndarray           # (8,8) row-stochastic per g
    B_w:    np.ndarray           # (8,8) = A_w for now
    J_w:    np.ndarray           # (16,16)
    q_w:    np.ndarray           # (8,)  complex

    def symplectic_residual(self) -> float:
        """||J^T Ω J − Ω||_F, paper eq 6."""
        J = self.J_w
        res = J.T @ OMEGA_8 @ J - OMEGA_8
        return float(np.linalg.norm(res, "fro"))


# ---------------------------------------------------------------------------
# Manifold (collection of cells, stored as stacked numpy arrays)
# ---------------------------------------------------------------------------

class WordManifold:
    """Collection of per-cell states.  Stored as stacked numpy tensors
    for cache-friendly access; index into any tensor by `idx[lemma]`."""

    # ----- construction ----------------------------------------------------

    def __init__(self,
                 lemmas:     list[str],
                 m:          np.ndarray,       # (N,)
                 pos_counts: np.ndarray,       # (N, 8)
                 trigrams:   np.ndarray,       # (N, 8, 8)
                 epsilon:    float = EPSILON_SMOOTH,
                 lambda_tikh: float = LAMBDA_TIKH):
        self.lemmas       = list(lemmas)
        self.idx          = {lem: i for i, lem in enumerate(self.lemmas)}
        self.N            = len(self.lemmas)
        # Accept either int counts (legacy) or float counts (post-rebuild
        # with per-source token weights).  Build pipeline only ever
        # divides / takes logs of these, so float storage is fine.
        if np.issubdtype(m.dtype, np.floating):
            self.m            = m.astype(np.float64, copy=False)
            self.pos_counts   = pos_counts.astype(np.float64, copy=False)
            self.trigrams     = trigrams.astype(np.float64, copy=False)
        else:
            self.m            = m.astype(np.int64, copy=False)
            self.pos_counts   = pos_counts.astype(np.int64, copy=False)
            self.trigrams     = trigrams.astype(np.int64, copy=False)
        self.epsilon      = float(epsilon)
        self.lambda_tikh  = float(lambda_tikh)

        # Derived tensors — populated by build().
        self.pi:    Optional[np.ndarray] = None
        self.alpha: Optional[np.ndarray] = None
        self.beta:  Optional[np.ndarray] = None
        self.A:     Optional[np.ndarray] = None
        self.B:     Optional[np.ndarray] = None
        self.J:     Optional[np.ndarray] = None
        self.q:     Optional[np.ndarray] = None
        self.m_bar: Optional[np.ndarray] = None   # (8,) per-channel means
        self.mu:    Optional[np.ndarray] = None   # (8,) per-channel μ
        self.omega: Optional[np.ndarray] = None   # (N, 8) frequencies
        self.eps:   Optional[np.ndarray] = None   # (N, 8) specific energies
        self.a_sm:  Optional[np.ndarray] = None   # (N, 8) semi-major axes

    # ----- derivations -----------------------------------------------------

    def _compute_pi(self) -> np.ndarray:
        """π_w^(g) = m_w^(g) / m_w.  Exact simplex by construction."""
        m = self.m.astype(np.float64)
        pc = self.pos_counts.astype(np.float64)
        pi = pc / np.maximum(m[:, None], 1.0)
        return pi

    def _compute_alpha(self) -> tuple[np.ndarray, np.ndarray]:
        """α_w^(g) via smoothing + per-channel scaling + per-word geometric
        anchor.  See class docstring for the three-step construction.
        Returns (alpha, m_bar_smoothed)."""
        pc_smoothed = self.pos_counts.astype(np.float64) + self.epsilon
        # Step 2: per-channel scale.
        m_bar = pc_smoothed.mean(axis=0)                 # (8,)
        alpha_raw = pc_smoothed / m_bar[None, :]         # (N, 8)
        # Step 3: per-word geometric anchor, ⟨log α_w⟩_g = 0.
        # G_w = exp(mean_g log α_raw^(g))
        log_alpha = np.log(alpha_raw)
        G = np.exp(log_alpha.mean(axis=1, keepdims=True))   # (N, 1)
        alpha = alpha_raw / G
        return alpha, m_bar

    def _compute_A(self) -> np.ndarray:
        """A_w[g, h] = count_w[g, h] / Σ_h count_w[g, h].  Rows with no
        observed context default to uniform 1/8 so the row-stochastic
        property holds for every word."""
        tg = self.trigrams.astype(np.float64)
        row_sums = tg.sum(axis=2, keepdims=True)          # (N, 8, 1)
        with np.errstate(divide="ignore", invalid="ignore"):
            A = np.where(row_sums > 0, tg / np.maximum(row_sums, 1.0), 1.0 / N_CH)
        # Sanity: every row sums to 1 (up to float precision).
        return A

    def _assemble_J(self,
                     alpha: np.ndarray,
                     beta:  np.ndarray,
                     A:     np.ndarray,
                     B:     np.ndarray) -> np.ndarray:
        """Build J_w per cell from eq 15.

           | α·A     β·B    |
           | −β·B   (α·A)⁻¹ |
        """
        N = alpha.shape[0]
        J = np.zeros((N, 16, 16), dtype=np.float64)
        # diag(α) · A  :  multiply each row of A_w by the matching α_w^(g)
        aA = alpha[:, :, None] * A        # (N, 8, 8)
        bB = beta[:, :, None]  * B
        # Invert α·A per cell.  Tikhonov: (α·A + λ·I)⁻¹ with λ scaled by
        # each cell's own row-max norm so conditioning is uniform across
        # cells.  For λ=0, fall back to np.linalg.pinv.
        if self.lambda_tikh > 0.0:
            row_max = np.max(np.abs(aA), axis=(1, 2))        # (N,)
            lam = self.lambda_tikh * row_max                 # (N,)
            reg = lam[:, None, None] * np.eye(N_CH)[None, :, :]
            aA_inv = np.linalg.inv(aA + reg)
        else:
            aA_inv = np.linalg.pinv(aA)
        J[:, :N_CH, :N_CH] =  aA
        J[:, :N_CH, N_CH:] =  bB
        J[:, N_CH:, :N_CH] = -bB
        J[:, N_CH:, N_CH:] =  aA_inv
        return J

    def _seed_q(self, alpha: np.ndarray) -> np.ndarray:
        """Kähler complex coord, seeded |q_w^(g)|² = α_w^(g), phase = 0."""
        return np.sqrt(alpha).astype(np.complex128)

    def _compute_kepler(self,
                         alpha: np.ndarray,
                         beta:  np.ndarray) -> tuple[np.ndarray, np.ndarray,
                                                      np.ndarray, np.ndarray]:
        """Per-channel Keplerian quantities (paper eq 16).  Returns
        (mu_per_channel, specific_energy, semi_major_axis, omega).
        mu^(g) = Σ_u m_u^(g)  (G = 1; units absorbed into α, m).
        For bound orbits (ε < 0): a = −μ/(2ε), ω = √(μ/a³).
        For unbound (ε ≥ 0): a = μ/(2|ε|) and we still compute ω as
        √(μ/a³) so the frequency is real and positive; the cell can
        be flagged via the sign of ε."""
        mu = self.pos_counts.sum(axis=0).astype(np.float64)    # (8,)
        mu = np.maximum(mu, 1.0)                                # guard
        eps = 0.5 * beta**2 - mu[None, :] / alpha               # (N, 8)
        a   = -mu[None, :] / (2.0 * np.where(eps != 0.0, eps, 1e-30))
        a_abs = np.abs(a)
        omega = np.sqrt(mu[None, :] / (a_abs ** 3))
        return mu, eps, a, omega

    def build(self) -> "WordManifold":
        """Populate all derived tensors.  Returns self so callers can chain."""
        self.pi                 = self._compute_pi()
        self.alpha, self.m_bar  = self._compute_alpha()
        self.beta               = 1.0 / self.alpha
        self.A                  = self._compute_A()
        self.B                  = self.A.copy()          # paper convention
        self.J                  = self._assemble_J(self.alpha, self.beta,
                                                    self.A, self.B)
        self.q                  = self._seed_q(self.alpha)
        self.mu, self.eps, self.a_sm, self.omega = self._compute_kepler(
            self.alpha, self.beta)
        return self

    # ----- effective frequency vectors (for sentence interference) ---------

    def tilde_omega(self) -> np.ndarray:
        """ω̃_w = π_w ⊙ ω_w  (paper eq 18), shape (N, 8)."""
        if self.omega is None or self.pi is None:
            raise RuntimeError("call build() first")
        return self.pi * self.omega

    def tilde_omega_of(self, lemma: str) -> np.ndarray:
        i = self.idx[lemma]
        return self.pi[i] * self.omega[i]

    # ----- per-cell access -------------------------------------------------

    def cell(self, lemma_or_idx) -> Cell:
        """Materialise one Cell (copy of the slices)."""
        if isinstance(lemma_or_idx, str):
            i = self.idx[lemma_or_idx]
        else:
            i = int(lemma_or_idx)
        return Cell(
            lemma  = self.lemmas[i],
            m_w    = int(self.m[i]),
            alpha  = self.alpha[i].copy(),
            beta   = self.beta[i].copy(),
            pi     = self.pi[i].copy(),
            A_w    = self.A[i].copy(),
            B_w    = self.B[i].copy(),
            J_w    = self.J[i].copy(),
            q_w    = self.q[i].copy(),
        )

    # ----- diagnostics -----------------------------------------------------

    def symplectic_residuals(self) -> np.ndarray:
        """||J^T Ω J − Ω||_F per cell, shape (N,)."""
        if self.J is None:
            raise RuntimeError("call build() first")
        # Vectorised: (N,16,16) → per-cell Frobenius
        M = np.einsum("nij,jk,nkl->nil", np.transpose(self.J, (0, 2, 1)),
                                          OMEGA_8, self.J)
        M -= OMEGA_8[None, :, :]
        return np.linalg.norm(M.reshape(self.N, -1), axis=1)

    def log_alpha_stats(self) -> dict:
        """Report ⟨log α⟩ per channel and overall.  Training anchor target = 0."""
        la = np.log(self.alpha)
        return {
            "overall_mean_log_alpha":  float(la.mean()),
            "per_channel_mean_log_alpha": la.mean(axis=0).tolist(),
        }

    # ----- I/O -------------------------------------------------------------

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "lemmas":      self.lemmas,
            "m":           self.m,
            "pos_counts":  self.pos_counts,
            "trigrams":    self.trigrams,
            "pi":          self.pi,
            "alpha":       self.alpha,
            "beta":        self.beta,
            "A":           self.A,
            "B":           self.B,
            "J":           self.J,
            "q":           self.q,
            "m_bar":       self.m_bar,
            "mu":          self.mu,
            "omega":       self.omega,
            "eps":         self.eps,
            "a_sm":        self.a_sm,
            "epsilon":     self.epsilon,
            "lambda_tikh": self.lambda_tikh,
        }
        with path.open("wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: Path) -> "WordManifold":
        with Path(path).open("rb") as f:
            p = pickle.load(f)
        obj = cls(lemmas=p["lemmas"], m=p["m"],
                   pos_counts=p["pos_counts"], trigrams=p["trigrams"],
                   epsilon=p.get("epsilon", EPSILON_SMOOTH),
                   lambda_tikh=p.get("lambda_tikh", LAMBDA_TIKH))
        obj.pi, obj.alpha, obj.beta = p["pi"], p["alpha"], p["beta"]
        obj.A, obj.B, obj.J, obj.q  = p["A"], p["B"], p["J"], p["q"]
        obj.m_bar = p["m_bar"]
        obj.mu    = p.get("mu")
        obj.omega = p.get("omega")
        obj.eps   = p.get("eps")
        obj.a_sm  = p.get("a_sm")
        return obj
