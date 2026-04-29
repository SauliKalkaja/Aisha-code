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

## 6. The runtime pipeline — manifold as a structural prior on a small LM

The math above is the engine.  In production we use it as a
**structural prior on a small instruction-tuned language model**.
Five boxes wired in series:

```
   user text
       │
       ▼
    HARPER (in)               # rule-based grammar / spelling clean-up
       │
       ▼
    MANIFOLD (Aisha)          # boundary words + intent flag + 16-D centroid
       │           │
       │           ▼
       │       intent classifier → λ ∈ { 0, 1.5 }
       ▼
    LM  +  λ · m              # Qwen2.5-0.5B-Instruct, GGUF Q4_K_M, ~250 MB
       │   (logit-bias mask
       │    from boundary)
       ▼
    HARPER (out)              # polish only — never invents words
       │
       ▼
   reply
```

**Harper (input).**  The user's text first passes through Harper — a
small, open, rule-based grammar / spelling checker.  This fixes typos
and normalises punctuation so the manifold gets a clean sentence.
Harper is not a language model: it is a deterministic ruleset.  No
probabilistic generation, no NN.

**Manifold (Aisha).**  The cleaned text is mapped onto the Kähler
manifold from §5.  Three things come back:

1. A **boundary** — the manifold-neighbour words of the conversation's
   running 16-D centroid (k-NN under POS-Kähler distance).  Typically
   30 – 80 words.  These are the words the conversation has been about,
   geometrically.
2. A **structural fingerprint** — the 16-D centroid itself, the POS-class
   mixture, and the inter-turn step.  Roughly thirty floating-point
   numbers.  This is the conversation's compact memory.
3. An **intent flag** — does the current question read as past-self-
   reflective ("Was my breakfast healthy?") or advisory ("What should I
   read?").  A small regex on the user message.

**Intent classifier picks λ.**  Reflective questions get λ = 0; the
boundary bias is disabled and the LM relies on the verbatim history.
Advisory and exploratory questions get λ = 1.5.  This single switch
turned the boundary from a feature that wins five of eight cases into
one that strictly dominates the words-only baseline on every metric we
tested.  The classifier itself is seventeen lines of regex.

**LM with logit bias.**  The LM (Qwen2.5-0.5B-Instruct, ~250 MB on
disk at int4) sees the prior conversation as plain text and a per-token
bias vector `λ · m`, where `m ∈ {0, 1}^V` is one for every token id of
every word in the boundary.  This bias is the only place where Aisha
"talks" to the LM: the structural numbers themselves never cross over
as text.  We tried that earlier and the LM read words like "centroid"
literally, started talking about AI ethics, and ignored the actual
conversation.  The logit-bias channel is the architectural rule:
**words for the LM, structure for Aisha**.

**Harper (output).**  The LM's reply is fluent but sometimes off by an
article or a tense.  Harper runs a second pass to clean those up.  It
is only allowed to polish — it never invents or substitutes words from
outside the LM's output.  This is a hard rule, the same constraint
Aisha-only operation has always had.

The pipeline runs end-to-end on the phone.  No cloud calls, no server
tier.  The Aisha contribution at runtime is roughly thirty numbers per
turn plus an elementwise add per token; everything else is the small
LM doing its usual job, just with a topical anchor.

---

## 7. Compute footprint

Per response, three components:

**Aisha — the structural prior.**  Per current turn:

- One polynomial evaluation of `K` at each content word's `q` (~280
  ops for deg-4 in 8 vars).
- One Hermitian-form inner product (`(70 × 70)` Hermitian).
- One Mahalanobis-distance pass against the manifold's vocabulary to
  pull the K = 30 nearest neighbours of the running 16-D centroid.
- A single regex evaluation for the intent classifier.

Total Aisha time is on the order of 50 ms on a 2026 mid-range phone.
No GPU.  The trained artifact (the Hermitian matrix `H`) is **~250 KB**.

**The LM — the vocabulary engine.**  Qwen2.5-0.5B-Instruct, GGUF Q4_K_M.
About 494 M parameters, ~250 MB on disk, a few hundred MB resident
during inference.  Roughly 0.5 – 1 second to produce a 50-token reply
on a Snapdragon 8 Gen 3-class phone.

**The bias channel — essentially free.**  At each generation step the
sampler does one elementwise add `ℓ + λ · m` before softmax.  Cost is
microseconds per token.

End-to-end energy per reply, estimated:

| Configuration | J / reply |
|---|---|
| Aisha + Qwen-0.5B (this work) | ~3 J |
| Phi-3-mini 3.8B alone, on the same phone | ~20 J |
| Cloud GPT-class call | ~1000 J |

Roughly seven times cheaper than running a "drop-in" 3.8 B-class chat
LM on the same device, and roughly three hundred times cheaper than a
cloud-served reply (which also adds your phone's radio for the network
round-trip).  These are estimates from public benchmarks; a factor-of-2
error in any direction would not change the story.

The savings come from running a small LM rather than a large one.
Aisha's contribution is making that small LM produce on-topic and
register-matched responses without growing its parameter count.

---

## 8. Why we put this out

The math is simple.  The structure is established (symplectic geometry,
Kähler potentials, sampler chains in llama.cpp).  The empirical data
shows the same framework discovers content / grammar separation,
locality structure, and human-vs-Aisha path-geometry differences in
language *and* matches NASA JPL Horizons on Mercury's orbit to >99.99 %.
The same line element, doubled the same way, in two very different
domains.

The hybrid configuration — Aisha as a logit-bias on a small LM — is
shipping in a phone app and the empirical data is in the
[AishaLLM-experiments](https://github.com/SauliKalkaja/AishaLLM-experiments)
repo.  On Qwen2.5-0.5B-Instruct: average factual recall up from 2.50
to 3.12 keywords per response, stylistic match to the conversation
register up by ~19 %, no measurable cost in fluency.  Numbers small
enough to be honest about; pattern consistent enough across cases to
defend.

What the standalone manifold does *not* do alone is **predication** —
which `(verb, object)` and `(adjective, noun)` combinations carry
meaning.  Aisha-only output is grammatical word-salad because the
manifold knows where words are but not which combinations cohere.  The
hybrid sidesteps this by letting a small LM provide predication and
using Aisha to keep the LM on topic across turns.  The standalone
problem is still genuinely open and a tractable one: it is a question
about constraint satisfaction over the symplectic structure, not about
scaling parameters.

If you're a physicist, mathematician, or linguist who sees this and
recognizes a tool you can apply: please.  The framework is small enough
to read in an afternoon, the code is MIT-licensed, and the math is in
textbooks you already own.  Concretely:

- **Drop it into your existing LLM stack as a logit-bias sampler.**
  See [DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md) — recipes for llama.cpp,
  Hugging Face `transformers`, and vLLM are short.
- **Train your own Kähler matrix on a domain corpus.**  ~25 minutes on
  one RTX 4090.  See `aisha/kahler_pos_train.py`.
- **Push on standalone predication.**  The manifold gives you both
  `∇K` (content) and `ξ = J · ∇^a K` (grammar) as orthogonal vector
  fields.  The combinatorics of which `∇K`-`ξ` interactions form valid
  predicates is geometric, not statistical.  Worth a try.

Repository entry points:

- `aisha/aisha_lm_helpers.py` — public API: `is_reflective_question`,
  `aisha_structure`, `boundary_with_structural_memory`.  This is the
  module you import in your own project.
- `aisha/responder_pos.py` — the standalone runtime; load the
  manifold, run a chat in Aisha-only mode.
- `aisha/kahler_pos_runtime.py` — analytical metric, gradient,
  Hamiltonian flow.
- `aisha/kahler_pos_train.py` — Donaldson-form training.
- `docs/DEVELOPER_GUIDE.md` — practical how-to for the hybrid
  configuration.
- `docs/6D_Symplectic_Unification.pdf` — the underlying physics paper.

The compute is `O(1)` per response from Aisha's side, dominated by the
small LM's per-token cost on the phone side.  The framework is
symplectic.  The energy is roughly three joules per reply.  We think
this matters.
