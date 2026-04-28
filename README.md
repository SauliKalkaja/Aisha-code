# Aisha

A non-LLM language engine built on a **6D symplectic phase space**.
Each word is a point on a complex Kähler manifold; each sentence is
a path on it.  Generation runs in closed analytical form — no neural
network at inference, no token sampling, no LM.

**Compute:** ~one polynomial evaluation per word.  No GPU at runtime.
Trained model artifact is **~250 KB**.

## Status

The framework is mathematically clean and produces grammatically valid
sentences with measurable empirical structure (a discovered split
between content-pulling gradient flow and grammar-pulling Hamiltonian
flow on the manifold).  Semantic coherence remains an open problem —
that is the next frontier and is left as research.

## Read first

- **[docs/AISHA_FRAMEWORK_INTRO.md](docs/AISHA_FRAMEWORK_INTRO.md)** —
  short, undergraduate-level explainer.  Walks through line-element
  doubling, the block Jacobian with α / β scaling, the symplectic
  lock αβ = 1, the hyperbolic invariant, the analytical jump
  integral, and how the Donaldson-form Kähler potential applies
  these to language.
- **[docs/6D_Symplectic_Unification.pdf](docs/6D_Symplectic_Unification.pdf)** —
  the underlying physics paper (Kälkäjä & Gemini, 2026).  Validated
  empirically against NASA JPL Horizons data on Mercury's orbit
  (>99.99 % positional accuracy, single analytical jump, no time
  stepping).

## Repository layout

```
docs/
  AISHA_FRAMEWORK_INTRO.md        Accessible math intro
  6D_Symplectic_Unification.pdf   Underlying physics

aisha/
  responder_pos.py                Main runtime (load + chat)
  kahler_pos_runtime.py           Analytical g, ∇K, Hamiltonian flow ξ
  kahler_pos_train.py             Donaldson-form training (~25 min on
                                  one RTX 4090)
  build_pos_manifold.py           Construct the 16-d POS-aligned coords
  data/
    conversations.csv             Tagged dialog corpus (~4 000 turns)
    processed/
      pos_kahler/h.npz            Trained Kähler matrix (70×70 Hermitian PSD)
      kahler_phase1/h.npz         Earlier 4-complex variant

scripts/
  download_artifacts.sh           Fetch the large pre-trained pickle
                                  from Hugging Face.
                                  https://huggingface.co/datasets/sauli-aisha/aisha-manifold
```

## Quickstart

```bash
# 1. Get the runtime artifact (~33 MB)
./scripts/download_artifacts.sh
# (add --train to also fetch the 128 MB training manifold)

# 2. Install minimal deps
pip install numpy torch scipy

# 3. Run the chat demo
python aisha/chat_demo.py
```

## Training your own Kähler matrix

```bash
# Build POS-aligned coordinates from the manifold
python aisha/build_pos_manifold.py

# Train H via Donaldson form on KDE density target
python aisha/kahler_pos_train.py
```

The trainer fits a `70 × 70` Hermitian positive-definite matrix `H` so
that the induced metric `g_{αβ̄} = ∂² log(s* H s) / ∂z^α ∂z̄^β` matches
an empirical word-density estimate.  Output is `data/processed/pos_kahler/h.npz`
(~250 KB).

## Open problems

The framework is solid; the surface output is grammatical but not yet
semantically coherent.  Specific open directions:

1. **Predication compatibility** — modeling which `(verb, object)` and
   `(adjective, noun)` combinations carry meaning.  Currently the
   manifold knows where words are, but not which combinations form
   sensible predicates.
2. **Sentence-level coherence** without a language model.  We tried
   word-level translation (MUSE alignment, anchored Jacobian
   transforms); none reached coherent output.  Possibly tractable via
   constraint satisfaction over the symplectic structure.
3. **Inverse jump integrals** — extending the analytical
   propagator from orbit dynamics to multi-clause sentence
   composition.

If you see a way in, contributions welcome.

## License

MIT — see [LICENSE](LICENSE).

## Citation

```
Kälkäjä, S. (2026). Aisha: A symplectic framework for language
generation. https://github.com/SauliKalkaja/Aisha-code
```

```
Kälkäjä, S. & Gemini, G. (2026). The 6D Symplectic Phase Space:
Unifying Newtonian Mechanics and Metric Deformation via the α/β
Invariant.
```
