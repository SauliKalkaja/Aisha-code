# Aisha

A **symplectic-manifold structural prior for language models**. Each
English word is a point on a 16-dimensional Kähler manifold; the
running 16-D centroid of a conversation is a compact memory of where
the conversation lives in topical space; the boundary k-NN of that
centroid becomes a logit-bias on any standard language model so a
small LM stays on topic and in register without growing its parameter
count.

**Compute**: ~one polynomial evaluation per word, plus one
elementwise-add per generation step. No GPU at runtime. Trained model
artifact is **~250 KB**. Designed to drop into existing LLM pipelines
(llama.cpp, transformers, vLLM) as an additional sampler.

## Status

The framework is mathematically clean and produces measurable empirical
structure (a discovered split between content-pulling gradient flow and
grammar-pulling Hamiltonian flow on the manifold).

The **standalone framework** (this repository) generates grammatically
valid sentences without a neural network at inference; it does not
produce semantically coherent ones on its own — the manifold knows
where words *are* but not which word combinations carry meaning.

The **hybrid configuration** — Aisha's boundary used as a logit-bias on
top of a small instruction-tuned language model — does produce coherent
output. Empirical validation is in the experiments repo
[AishaLLM-experiments](https://github.com/SauliKalkaja/AishaLLM-experiments)
(D14: +19% stylistic continuity vs words-only baseline at the same factual
recall; D15: 17/17 intent-classifier accuracy with a clean fix for the
one identified failure mode). The Aisha Android app ships this hybrid
configuration with Qwen2.5-0.5B-Instruct as the back-end.

## Read first

- **[docs/AISHA_FRAMEWORK_INTRO.md](docs/AISHA_FRAMEWORK_INTRO.md)** —
  short, undergraduate-level explainer. Walks through line-element
  doubling, the block Jacobian with α / β scaling, the symplectic lock
  αβ = 1, the hyperbolic invariant, the analytical jump integral, and
  how the Donaldson-form Kähler potential applies these to language.
- **[docs/DEVELOPER_GUIDE.md](docs/DEVELOPER_GUIDE.md)** — practical
  guide for dropping Aisha's analytical jump into your own LM project.
  Python API, code example, how to apply as logit bias in llama.cpp /
  Hugging Face transformers / vLLM. **Read this if you want to use
  Aisha in your project, not study its mathematics.**
- **[docs/6D_Symplectic_Unification.pdf](docs/6D_Symplectic_Unification.pdf)** —
  the underlying physics paper (Kälkäjä & Gemini, 2026). Validated
  empirically against NASA JPL Horizons data on Mercury's orbit
  (>99.99% positional accuracy, single analytical jump, no time
  stepping).

## Repository layout

```
docs/
  DEVELOPER_GUIDE.md              How to use Aisha in your LM project
  AISHA_FRAMEWORK_INTRO.md        Accessible math intro
  6D_Symplectic_Unification.pdf   Underlying physics

aisha/
  aisha_lm_helpers.py             Public API: is_reflective_question,
                                  aisha_structure, boundary_with_
                                  structural_memory.  This is the
                                  module to import in your own project.
  responder_pos.py                Loads the manifold + Kähler runtime
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

## Quickstart — using Aisha as a logit bias on your LM

```python
from responder_pos import POSResponder
from aisha_lm_helpers import (
    is_reflective_question,
    aisha_structure,
    boundary_with_structural_memory,
)

# Load the manifold once (~250 KB on disk, ~0.5 s).
r = POSResponder(use_harper=False)

# For every chat turn:
prior_turns = ["I'm planning a trip to Tokyo.",
                "I'll be there in April."]
user_msg    = "What clothes should I bring?"

# 1. Compute the boundary (words near the conversation's 16-D centroid).
words = boundary_with_structural_memory(
    r, current_turn=user_msg,
    prior_turns=prior_turns, memory_length=5)

# 2. Decide bias strength.  Reflective questions ("Was my X ok?") get
#    lambda=0 because the bias hurts them; advisory questions get 1.5.
lam = 0.0 if is_reflective_question(user_msg) else 1.5

# 3. Build a logit-bias mask on YOUR LM's vocabulary, where mask[t]=1
#    for any token id of any word in `words`, and pass `+ lam * mask`
#    to the model's logits at every generation step.  See
#    DEVELOPER_GUIDE.md for llama.cpp / transformers / vLLM glue.
```

## Quickstart — running the standalone framework demo

```bash
# 1. Get the runtime artifact (~33 MB)
./scripts/download_artifacts.sh
# (add --train to also fetch the 128 MB training manifold)

# 2. Install minimal deps
pip install numpy torch scipy

# 3. Run the chat demo (Aisha-only, grammatical but not semantic)
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
an empirical word-density estimate. Output is
`data/processed/pos_kahler/h.npz` (~250 KB).

## Open problems

The standalone framework is solid; the surface output is grammatical
but not yet semantically coherent without an LM in the loop. Specific
open directions:

1. **Predication compatibility** — modeling which `(verb, object)` and
   `(adjective, noun)` combinations carry meaning. Currently the
   manifold knows where words are, but not which combinations form
   sensible predicates.
2. **~~Sentence-level coherence without a language model~~ —
   superseded by the hybrid pipeline.** We now have a working
   configuration (Aisha boundary + small instruction-tuned LM); see
   [AishaLLM-experiments](https://github.com/SauliKalkaja/AishaLLM-experiments)
   for the empirical work and the Aisha phone app for the production
   deployment. The standalone-no-LM version remains a research
   direction; the hybrid is shipping.
3. **Inverse jump integrals** — extending the analytical propagator
   from orbit dynamics to multi-clause sentence composition.
4. **Training-time integration with the LM**. Currently the LM is
   pretrained and Aisha is bolted on at inference. Co-training the LM
   to attend to the boundary signal directly may give larger gains
   than the inference-time bias does.

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
