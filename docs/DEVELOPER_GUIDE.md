# Aisha Developer Guide — Using the Analytical Jump in Your LM Project

This is the practical guide. It assumes you have your own LM pipeline
(llama.cpp, Hugging Face `transformers`, vLLM, MLX, your own
implementation, etc.) and you want to add Aisha's structural prior on
top, with no model retraining and no architecture changes.

If you want to understand the underlying mathematics, read
[AISHA_FRAMEWORK_INTRO.md](AISHA_FRAMEWORK_INTRO.md) and
[6D_Symplectic_Unification.pdf](6D_Symplectic_Unification.pdf) first.

If you want to see the empirical evidence that this actually improves
LM output, read [AishaLLM-experiments](https://github.com/SauliKalkaja/AishaLLM-experiments)
(experiments D1, D6, D9, D11, D14, D15 are the ones that matter).

---

## What Aisha gives your LM

Three operations, all cheap. None require a GPU. None modify your LM:

| Function | Input | Output | Cost |
|---|---|---|---|
| `is_reflective_question(text)` | a single string | `True` / `False` | regex, microseconds |
| `aisha_structure(responder, text)` | text of any length | dict with 16-D centroid + POS profile + sentence count | one Mahalanobis pass per sentence |
| `boundary_with_structural_memory(responder, current_turn, prior_turns, memory_length)` | current message + prior turns | a list of words to apply as logit bias on your LM | k-NN on the manifold, ~50 ms on CPU |

All three are in [`aisha/aisha_lm_helpers.py`](../aisha/aisha_lm_helpers.py).
Pure Python + numpy. Same module runs on a laptop, in a server, in
Chaquopy on Android.

---

## The architectural rule

> **Words for the LM, structure for Aisha.**

Aisha holds a 16-dimensional running fingerprint of the conversation
(POS-class profile, manifold centroid, inter-turn step). That structure
**stays inside Aisha** — it is never serialised to text and shown to
the LM. The bridge between the two is the **logit bias** Aisha
generates from its structural state. The LM only ever sees: (a) the
conversation history as plain text and (b) a per-token bias on its
output distribution.

This rule matters in practice. Empirically, when we tried to feed
Aisha's centroid coordinates to the LM as natural-language preamble
text, the LM read words like "centroid" and "manifold" as topic words
and started producing AI-ethics responses regardless of conversation
content. The mechanism only works if the structural numbers stay on
Aisha's side of the boundary.

---

## Prerequisites

```bash
# Clone this repo and fetch the trained manifold.
git clone https://github.com/SauliKalkaja/Aisha-code.git
cd Aisha-code
./scripts/download_artifacts.sh

pip install numpy torch scipy
```

---

## Step 1 — Get the boundary words for the current turn

```python
from responder_pos import POSResponder
from aisha_lm_helpers import boundary_with_structural_memory

# Load the manifold once at startup (~0.5 s, ~250 KB on disk).
aisha = POSResponder(use_harper=False)

prior_turns = [
    "I'm planning a trip to Tokyo.",
    "I'll be there during cherry blossom season in April.",
]
user_msg = "What clothes should I bring?"

words = boundary_with_structural_memory(
    aisha,
    current_turn  = user_msg,
    prior_turns   = prior_turns,
    memory_length = 5,         # how many prior turns inform memory
)
# `words` is a list[str], usually 30-80 entries, e.g.
#   ['kimono', 'rain', 'jacket', 'spring', 'tokyo', ...]
```

The returned list is the **boundary**: words near both the current
turn's manifold seeds *and* the running 16-D centroid of the prior
turns. They are the words the conversation has been about, geometrically.

---

## Step 2 — Pick the bias strength via the intent classifier

```python
from aisha_lm_helpers import is_reflective_question

LAMBDA_DEFAULT     = 1.5
LAMBDA_REFLECTIVE  = 0.0     # disable bias for past-self-state Qs

lam = (LAMBDA_REFLECTIVE
       if is_reflective_question(user_msg)
       else LAMBDA_DEFAULT)
```

**Why two values?** D15 found that for *reflective* questions ("Was
my breakfast healthy?", "Did I make the right call?") the bias pulls
the LM toward generic-advice register and away from echoing the user's
specific facts. Disabling the bias on those questions restores
verbatim recall. The intent classifier is 17/17 on a probe set that
includes tricky cases like "What did you do today?" (correctly NOT
flagged reflective).

---

## Step 3 — Apply the bias as a logit boost

The math: on every generation step, replace your LM's logits ℓ ∈ ℝ^V
with

    ℓ' = ℓ + λ · m

where m ∈ {0, 1}^V is a sparse mask with `m[t] = 1` for every token id
`t` belonging to any word in `words` (you tokenize each word with your
LM's tokenizer). λ is the per-turn scalar from Step 2.

### llama.cpp

The modern `llama_sampler_chain` API has a built-in logit-bias sampler.
Build a `std::vector<llama_logit_bias>` from your tokenized boundary
words and add it to the chain before your other samplers:

```cpp
std::vector<llama_logit_bias> biases;
for (const auto & w : boundary_words) {
    for (const auto & variant : { w, " " + w, " " + capitalise(w) }) {
        for (auto t : llama_tokenize(vocab, variant, false, true)) {
            biases.push_back({ t, lambda });
        }
    }
}

llama_sampler * smpl = llama_sampler_chain_init(spc);
if (!biases.empty()) {
    llama_sampler_chain_add(smpl, llama_sampler_init_logit_bias(
        llama_n_vocab(vocab), biases.size(), biases.data()));
}
llama_sampler_chain_add(smpl, llama_sampler_init_top_k(40));
llama_sampler_chain_add(smpl, llama_sampler_init_top_p(0.9f, 1));
llama_sampler_chain_add(smpl, llama_sampler_init_temp(0.3f));
llama_sampler_chain_add(smpl, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));
```

This is exactly what the Aisha Android app does in its JNI bridge. See
the corresponding code in the application repo.

### Hugging Face `transformers`

Use a `LogitsProcessor`:

```python
import torch
from transformers import LogitsProcessor

class AishaBoundaryProcessor(LogitsProcessor):
    def __init__(self, tokenizer, words, lam, vocab_size):
        mask = torch.zeros(vocab_size)
        for w in words:
            for variant in (w, " " + w, " " + w.capitalize()):
                ids = tokenizer.encode(variant, add_special_tokens=False)
                for t in ids:
                    if 0 <= t < vocab_size:
                        mask[t] = 1.0
        self.bias = lam * mask

    def __call__(self, input_ids, scores):
        return scores + self.bias.to(scores.device)

# Then in generate():
out = model.generate(
    input_ids,
    logits_processor=[AishaBoundaryProcessor(tokenizer, words, lam, vocab_size)],
    max_new_tokens=80,
    temperature=0.3,
)
```

### vLLM

vLLM accepts `logit_bias` in `SamplingParams` directly (a `dict[int, float]`
mapping token-id → bias). Build the dict from the boundary the same way:

```python
from vllm import SamplingParams

logit_bias = {}
for w in words:
    for variant in (w, " " + w, " " + w.capitalize()):
        for t in tokenizer.encode(variant, add_special_tokens=False):
            logit_bias[t] = lam

sp = SamplingParams(temperature=0.3, max_tokens=80, logit_bias=logit_bias)
out = llm.generate(prompt, sp)
```

### Server-side OpenAI-compatible APIs

Many serving frameworks expose a `logit_bias` field in the chat
completion request, capped at ±100 in some implementations. Aisha's
λ ≈ 1.5 is well within that range. Build the dict per-request:

```json
{
  "model": "your-model",
  "messages": [...],
  "logit_bias": { "12345": 1.5, "67890": 1.5, ... }
}
```

If your serving stack doesn't expose `logit_bias`, you fall back to a
client-side `LogitsProcessor` like the transformers example above.

---

## Step 4 — Track the structural fingerprint across turns (optional)

If you want long-term memory of the conversation's geometric region (a
useful diagnostic, also useful for hallucination/drift detection), keep
the centroid:

```python
from aisha_lm_helpers import aisha_structure

# At any point in the conversation:
struct = aisha_structure(aisha, " ".join(prior_turns))
# struct = {
#   "doc_centroid": [0.12, -0.45, 0.83, ...],   # 16 floats
#   "pos_profile":  {"NOUN": 0.52, "VERB": 0.27, "ADJ": 0.13, "ADV": 0.05},
#   "n_seeds":      28,
#   "n_sents":      3,
#   "mean_step":    1.34,
# }
```

Use cases for the fingerprint:

- **Drift monitoring**: compute the answer's fingerprint after generation;
  compare to the conversation's running centroid. Large distance flags
  potential off-topic drift. Empirical caveat: this signal didn't
  fire on small instruction-tuned models in our tests because they
  follow context too well — so it's most useful with weaker / older
  base models.
- **Conversation memory across sessions**: the 16-D centroid is small
  enough to store per-user without the privacy concerns of storing
  raw transcripts. Re-seed a new session by pre-feeding the stored
  centroid as a virtual seed in `boundary_with_structural_memory`.
- **Style classification**: the POS-class profile is a discriminative
  feature for register (formal / casual / scientific / poetic). See
  experiment D13 in AishaLLM-experiments.

---

## Step 5 — Tune for your model

The defaults (λ=1.5, K=30 centroid neighbours, memory_length=5 turns,
k_per_seed=20 NN per current-turn seed) were tuned for
Qwen2.5-0.5B-Instruct on conversational chat. Other models may want
different values:

- **Larger LMs (≥3B)**: the bias does less because the model has stronger
  internal priors. Empirically, on Qwen-3B-Instruct in factual QA,
  λ=1.5 made effectively no difference (D11b). For larger models the
  *structural prefix* mechanism (a short text preamble describing the
  conversation's POS profile and step) is what helps; see D9.
- **Base / non-instruction-tuned models**: the bias helps **more** because
  these models have weaker conversational priors. D6 saw +30% PPL
  improvement and +200% topic-relevance gain with Pythia-410m. But
  base models often produce non-answers regardless; switch to
  instruction-tuned if you can.
- **Domain-specialised models**: the manifold here was trained on
  ~4 000 conversational dialog turns. For a code-only or
  scientific-only LM, retrain the manifold on your own corpus
  (`aisha/build_pos_manifold.py`, `aisha/kahler_pos_train.py`,
  ~25 minutes on a single RTX 4090).

The single most important configuration choice is the intent classifier
(Step 2). With routing on, our test conversations went from "Aisha
boundary helps in 5/8 cases, hurts in 1/8" to "Aisha boundary strictly
dominates words-only baseline on every metric, in every case."

---

## What to expect empirically

From the [AishaLLM-experiments](https://github.com/SauliKalkaja/AishaLLM-experiments)
data on Qwen2.5-0.5B-Instruct:

| Metric | Words-only baseline | + Aisha (intent-routed) | Improvement |
|---|---|---|---|
| Factual recall (avg keyword hits / response) | 2.50 | 3.12 | +25% |
| Stylistic match (centroid distance to conv) | 1.55 | 1.20 | -23% |
| Register match (POS L1 to conv) | 0.21 | 0.17 | -19% |

Per-turn cost on a 2026 mid-range phone (Snapdragon 8 Gen 3-class):

- Aisha boundary computation: ~0.3 J, ~50 ms
- Logit bias mask construction: <0.01 J, microseconds
- Total Aisha overhead: trivial relative to the LM's own inference

---

## What this is **not**

Be honest with users. Three things Aisha does not do:

1. **Aisha is not a fact memory.** The 16-D fingerprint cannot remember
   that the user said "Tokyo" or "eggs and bacon." Those specifics ride
   on the verbatim history you pass to the LM. If the conversation is
   too long for the LM's context window, you need a separate mechanism
   (summariser, key-value store) — not Aisha's geometry.
2. **Aisha does not detect hallucinations** at modern instruction-tuned
   scale. The fingerprint-distance approach was tested and didn't fire
   because Qwen-3B and similar simply don't hallucinate against
   contradictory sources. The mechanism may still be useful with weaker
   or older models; we have no production data for that.
3. **Aisha does not replace a larger model.** Switching from Pythia-410m
   to Qwen2.5-0.5B-Instruct gave a 40-percentage-point faithfulness
   improvement. Aisha's contribution on top of Qwen-0.5B is a 4-25%
   gain depending on metric. The big win is the backbone choice; Aisha
   is a meaningful additional improvement, not the dominant factor.

---

## Citation

If you use Aisha in research, please cite:

```
Kälkäjä, S. (2026). Aisha: A symplectic framework for language generation.
https://github.com/SauliKalkaja/Aisha-code
```

```
Kälkäjä, S. & Gemini, G. (2026). The 6D Symplectic Phase Space:
Unifying Newtonian Mechanics and Metric Deformation via the α/β Invariant.
```

For the empirical work specifically (the hybrid-LM experiments), cite:

```
Kälkäjä, S. & Anthropic Claude (2026). Empirical evaluation of the
Aisha symplectic prior as logit bias on small language models.
https://github.com/SauliKalkaja/AishaLLM-experiments
```

---

## License

MIT — see [LICENSE](../LICENSE). Use it however you want.
