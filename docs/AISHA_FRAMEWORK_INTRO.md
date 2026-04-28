# Aisha — A Symplectic Framework for Language Generation

## A short, accessible introduction to the math behind a non-LLM language engine

Target reader: undergraduate physics or applied math.  No prior NLP
background needed.  Goal: convince you the foundation is *simple* —
just classical mechanics in disguise — and worth your attention.

---

## 0. The one-line claim

We treat each English word as a point on a complex symplectic manifold
and a sentence as a path on that manifold.  All generation runs in
closed analytical form on the resulting phase space — no neural
network at inference.  Compute cost is roughly that of a polynomial
evaluation per word.

---

## 1. The line element gets doubled

Newtonian mechanics has a real metric.  In 3D space:
```
ds² = (E - Φ) Σᵢ mᵢ (dxᵢ² + dyᵢ² + dzᵢ²)
```
The bodies move on geodesics of this metric.  As bodies fall toward
a singularity (`r → 0`, the potential `Φ → -∞`), the geometry tears.

Our move: introduce an **imaginary buffer space** orthogonal to the real
spatial coordinates.  For each real axis `xᵢ` we add an imaginary
companion `iXᵢ`.  The line element becomes:
```
ds²_6D = (E - Φ) Σᵢ mᵢ [ (dx²ᵢ + dy²ᵢ + dz²ᵢ)  +  ((i dXᵢ)² + (i dYᵢ)² + (i dZᵢ)²) ]
```
The imaginary part *absorbs* the geometric stress that the real part
cannot hold.  When the real metric condenses (gravity well), the
imaginary metric expands.  Total volume is conserved.

This is a **6D complex manifold** — equivalently, a 12D real phase
space.  The "doubling" is not a trick; it's a coordinate choice that
keeps the metric finite where Newton's diverges.

---

## 2. Jacobian becomes block-structured

The transformation between Aisha's local frame and the global frame
on this manifold is a 6×6 Jacobian, partitioned into 3×3 blocks:
```
            ⎡  α A      β B      ⎤
J_6D   =    ⎣ −β B    (α A)⁻¹    ⎦
```
- `A` and `B` are 3×3 matrices encoding the geometric mapping.
- `α` and `β` are **diagonal scaling operators** — one per real axis.
- `α` is "spatial condensation" (real space compressing).
- `β` is "imaginary expansion" (buffer absorbing).

These are not free parameters.  They obey two laws.

---

## 3. Linear-algebra constraints make it work

### The symplectic lock
The manifold is symplectic: phase-space volume is preserved
(Liouville's theorem).  Translating to determinants:
```
det( J_6D ) ≡ 1
```
This forces the per-axis lock:
```
α · β = 1
```
Real condensation is exactly inverse to imaginary expansion.  The
universe is permitted to bend; it is not permitted to crush volume.

### The hyperbolic invariant
With the lock plus an antisymmetric **torsion tensor** `M` capturing
cross-axis coupling, simple algebra yields:
```
( α + β )² − M² = 4
```
This is the conservation law of metric deformation under torsion.
For zero torsion (`M = 0`), `α = β = 1` and we recover flat Newtonian
geometry.  As torsion grows, `α` and `β` separate to absorb it.

### Golden-ratio side-effect
Solving the residual equation `α² λ² − α λ − 1 = 0` for the equilibrium
condition gives `α_s = φ / λ` where `φ ≈ 1.618…` is the golden ratio.
The system locks into a known geometric resonance.

---

## 4. Hamiltonian equations + the "jump integral"

The 6D Hamiltonian (with cross-coupling χ from torsion) reads:
```
   H(q, p) = α · pᵣ²/(2μ)  +  pφ²/(2μ r² sin²θ)  +  V(r) + H_twist + χ
```
By Hamilton's equations, the angular shift is:
```
   dφ/dt = ∂H/∂pφ
```
Here is the punchline.  In standard mechanics this becomes a sum of
small differential time steps `dt`.  But because the symplectic lock
binds `α` and `β`, `dt` *cancels exactly* through the chain rule
`dr/dt = pᵣ/μ`:
```
   M(r) · (pᵣ / μ) · dt  =  M(r) · dr
```
The temporal differential drops out.  What's left is a **pure spatial
integral bounded by the orbit's anchors** (perihelion `r_per` and
aphelion `r_aph`):
```
   Δφ_total  =  ∮ M(r) dr  +  Σ ∂χ_ij/∂θ dθ   (around the orbit)
```
This integral is **fully analytical** — no Runge-Kutta, no time
stepping, no accumulating drift.  We call it the **jump integral**.
It is the engine of an `O(1)` propagator: given anchors, evaluate
the integral, get the trajectory.

For Mercury (`e ≈ 0.21`, the most eccentric inner planet), this
matches NASA JPL Horizons data to **> 99.99 % positional accuracy**
across a 10-year randomized Monte Carlo sweep.

This is not an approximation.  It is a coordinate choice that
**eliminates the differential time variable** by exploiting the
symplectic lock.

---

## 5. Where Aisha lives — Kähler potential on a per-cell language manifold

The framework above is gravity / orbital mechanics.  Now apply it to
language:

- Each English word `w` gets a position on a complex manifold,
  `q_w ∈ R^16` (eight real axes + eight imaginary).
- The eight real axes are POS (parts of speech): `NOUN, VERB, ADJ,
  ADV, PRON_DET, PREP, CONJ, INTJ`.
- The eight imaginary axes are POS-shifts (the conjugate momenta —
  how each word transitions POS context).
- A sentence is a *path* on this manifold.

The metric tensor at each word's position is derived from a **Kähler
potential** `K(z, z̄)`.  We use the **Donaldson form**:
```
   K(z, z̄)  =  log( s(z)*  ·  H  ·  s(z̄) )
```
where:
- `s(z)` = vector of monomials `z^α` for multi-indices `|α| ≤ 4`
  in 8 complex variables (~70 monomials).
- `H` is a `70 × 70` Hermitian positive-definite matrix, learned
  once on a corpus to match the empirical word-density of English
  conversation.

The Hermitian metric `g_{α β̄} = ∂² K / ∂z^α ∂z̄^β` is **structurally
positive-definite** by Cholesky construction.  Curvature, gradient,
and the Hamiltonian flow `ξ = J · ∇^a K` are all closed-form
polynomial expressions in `(q, q̄)`.

### Empirical findings (per-word, 30 824 vocabulary)

- The **gradient direction** `∇K` points toward the corpus density
  basin — high-frequency content nouns / time-words.
- The **Hamiltonian flow** `ξ = J · ∇^a K` (perpendicular to `∇K` by
  construction) points toward function/glue words — articles,
  prepositions, pronouns, conjunctions.
- These two vector fields are *exactly orthogonal* under the Kähler
  metric and the **vocabularies they hit have zero overlap**.

This split — content along `∇K`, grammar along `ξ` — emerged from
the geometry without supervision.  It is a property of the corpus's
own statistics expressed in symplectic coordinates.

---

## 6. Compute footprint

Inference per word:
- One polynomial evaluation of `K` at the word's `q` (~280 ops for
  deg-4 in 8 vars).
- One Hermitian-form inner product (`(70 × 70)` Hermitian).
- One nearest-neighbour lookup on a precomputed dialog vocabulary.

No GPU is required at runtime.  The training of `H` (one Cholesky-
parameterized Hermitian PSD matrix) runs in ~25 minutes on a single
RTX 4090.  The trained artifact is **~250 KB**.

---

## 7. Why we put this out

The math is simple.  The structure is established (symplectic
geometry, Kähler potentials).  The empirical data shows the system
discovers content / grammar separation, locality structure, and
human-vs-Aisha path-geometry differences — *without* a language model.

Aisha generates grammatically valid sentences.  She does not yet
generate semantically coherent ones; that requires further work on
**predication** — the rules governing which word combinations carry
meaning.  Predication is the next frontier and is genuinely open.

If you're a physicist, mathematician, or linguist who sees this and
recognizes a tool you can apply: please.  The framework is small
enough to read in an afternoon, the code is open, and the math is
in textbooks you already own.

Repository entry points:
- `aisha/responder_pos.py` — the runtime
- `aisha/kahler_pos_runtime.py` — analytical metric, gradient,
  Hamiltonian flow
- `aisha/kahler_pos_train.py` — Donaldson-form training
- `docs/6D_Symplectic_Unification.pdf` — the underlying physics paper

The compute is `O(1)` per response.  The framework is symplectic.
The energy is one polynomial evaluation per word.  We think this
matters.
