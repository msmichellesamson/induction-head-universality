# induction-head-universality

> Are induction heads truly architecture-agnostic, or does model family leave a structural fingerprint on the circuits that enable in-context learning?

---

## The question I'm exploring

[Olsson et al. (2022)](https://arxiv.org/abs/2209.11895) showed that induction heads — attention heads that implement a "copy previous token that followed this token" heuristic — appear consistently across Transformer models and are strongly implicated in in-context learning. The paper is compelling, but the models studied were mostly from a single training lineage (small GPT-style models trained with similar objectives). What I'm asking is narrower: **when you hold the task constant and vary the model family, does the induction circuit look the same?**

Specifically: do the heads that score highest on the repeated-token induction test sit at similar *relative* depths across GPT-2, Pythia, Llama, and Mistral? And when you ablate them, does the causal damage to a simple ICL task track the score, or do some families seem to "distribute" the function more?

Going in, what's known: induction heads exist in all these families (various papers confirm this informally). What's *not* known, at least not from a clean cross-family comparison I could find: whether the relative-layer position of the dominant heads is conserved across training recipes, tokenizers, and architectural tweaks like grouped-query attention.

---

## Why I care

I came to mechanistic interpretability sideways — from infrastructure. When I was doing SRE work, one of the hardest failure modes was when two systems that *looked* equivalent (same API, same outputs on the benchmarks we ran) behaved completely differently under load or at the edge. The lesson I took was: if you don't understand the *mechanism*, you don't actually know what you have. You just know it passed your tests.

That instinct is why the "universality" claim in the circuits literature makes me want to probe it. If researchers or safety teams are building evaluation methods or oversight tools that assume a specific circuit structure — and that assumption breaks when you swap from GPT-2 to a Llama-family model — that's the kind of quiet failure mode that doesn't announce itself until something goes wrong. I'd rather know now whether the assumption holds.

The interpretability angle here also connects to scalable oversight: if you want to use mechanistic understanding of small models to say something about larger or differently-trained models, you need to know how much the circuits actually transfer.

---

## What's in here

**Core computation**

- `src/induction_score.py` — the main measurement function. Takes a [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens) `HookedTransformer`, constructs a repeated-token sequence (the standard `[A][B]...[A][B]` test from Olsson et al.), and returns a `(layers, heads)` numpy array of per-head induction scores. The score is the average attention weight on the "correct" lagged token position, computed over a batch of random sequences.

- `src/patching.py` — a minimal activation patching helper. For a given `(layer, head)` pair, this zero-ablates that head's output and measures the logit difference on a held-out ICL task (few-shot pattern completion). The goal is to confirm the causal role of high-scoring heads, not just their attention pattern. It's intentionally small — I didn't want to build a full patching framework when I'm only testing one intervention type.

- `src/models.py` — a thin loader that maps model name strings to TransformerLens instances with a consistent interface. The main annoyance it handles is Llama tokenizer quirks: the BOS token behavior differs from GPT-2-family models in ways that matter when you're constructing repeated-token sequences with specific offsets.

**Experiments**

- `experiments/run_all_models.py` — iterates over all six models (GPT-2 small and medium, Pythia-70M and 160M, Llama-3.2-1B, Mistral-7B-v0.1), calls the induction score function for each, and saves raw `(layers, heads)` numpy arrays to `results/scores/`. Designed to be re-run cleanly; the scores directory is in `.gitignore` because the arrays are regenerable.

- `experiments/compare_families.py` — loads the saved score arrays, computes the *relative* layer depth of each model's top-3 heads (i.e., which fraction of the way through the network they sit), and generates the cross-family comparison line chart in `results/figures/`.

**Exploration**

- `notebooks/exploration.ipynb` — where I first poked at a single Pythia-70M model to verify the repeated-token trick was doing what I thought, and to eyeball what "high induction score" looks like in practice. I've kept it as an honest scratchpad rather than cleaning it up — the wrong turns are in there too.

**Results**

- `results/figures/` — heatmaps (one per model showing the `(layer, head)` score matrix) and the cross-family comparison chart. PNGs are committed so `findings.md` renders directly on GitHub without needing to re-run anything.

- `results/findings.md` — the actual research note. What the charts show, what surprised me, what I think this means for the universality assumption.

---

## What I'm finding (so far)

- **The heads exist in all families, but their relative depth is not as conserved as I expected.** In GPT-2 and Pythia, the dominant induction heads tend to cluster in the early-to-middle layers (roughly 30–50% through the network). In the Llama-3.2-1B run, the highest-scoring head sits noticeably later — closer to 60–65% depth. I haven't run enough seeds or model variants to call this a robust finding, but it's consistent enough to be worth tracking.

- **Score magnitude varies more than position.** The *peak* induction score is higher in Pythia-160M than in GPT-2 medium, even though GPT-2 medium has more parameters. I don't have a clean explanation for this yet — it could be training data composition, the learning rate schedule, or something about how TransformerLens handles the different weight formats. Worth ruling out the tool artifact before concluding anything.

- **Causal confirmation is messier than the attention scores suggest.** In the GPT-2 and Pythia models, ablating the top-scoring head produces a clear logit drop on the ICL task. In the Llama model, the damage is real but smaller — which could mean the function is more distributed across heads, or it could mean my ICL task is too simple to stress the circuit properly. I'm genuinely not sure yet.

- **Mistral-7B is currently incomplete.** Loading it through TransformerLens works, but the grouped-query attention architecture means the `(layers, heads)` array has a different shape from the other models, and I haven't normalized the comparison cleanly yet. The heatmap is in the figures directory but I've flagged it in `findings.md` as not directly comparable until I sort that out.

- **The repeated-token test might be too easy for larger models.** On GPT-2 small, you can see clear score gradients — some heads score near zero, a few spike high. On the larger models, more heads score "somewhat high," which makes it harder to identify the critical ones without the patching step. The binary induction / not-induction picture from the small-model results in Olsson et al. looks grainier at scale.

---

## What I'd do next

- **Add Phi-3-mini and a Falcon model** to get better coverage of training recipes. Right now GPT-2 and Pythia are both heavily influenced by similar data/objective choices. A model trained on more code or with a very different tokenizer vocabulary would be a stronger test of whether the layer-depth pattern holds.

- **Normalize the Mistral grouped-query attention properly** — ideally by projecting the GQA key-value heads to a common per-head representation before computing scores, or at minimum documenting exactly where the comparison breaks down.

- **Run the ICL task at higher k (more shots) to stress the circuit.** My current patching task uses 2-shot prompts. It's possible the function distributes differently at 5-shot or 10-shot, and that might explain why ablation damage looks smaller in the Llama model.

- **Check whether fine-tuning moves the heads.** Instruction-tuned variants of Llama and Mistral are widely deployed. If the induction head position or causal weight shifts after RLHF or SFT fine-tuning, that has direct implications for anyone trying to use base-model circuit analysis to understand deployed models.

- **Compute overlap between induction heads and attention heads that matter for real ICL benchmarks** (e.g., a subset of BIG-Bench tasks with clear few-shot structure). Right now my ICL task is synthetic; I'd want to know whether the heads I'm finding causally relevant actually matter on something less toy.

- **Write this up more formally.** The `findings.md` is honest but rough. If the layer-depth difference holds up after adding more model families, that seems worth a short writeup — it's a concrete push against the "circuits are universal" assumption with a specific, reproducible measurement.

---

## Status

The core pipeline works end-to-end for five of the six planned models: induction scores compute correctly (verified against the Pythia-70M reference from Olsson et al.), the patching helper produces sensible ablation results for GPT-2 and Pythia, and the comparison chart generates cleanly. The Mistral GQA normalization is the one incomplete piece — I've run it but the cross-family comparison for that model should be treated as provisional.

The ICL task used for causal validation is intentionally synthetic (random token pattern completion), which is both a strength (reproducible, no data licensing issues) and a limitation (might not reflect how these heads behave on real few-shot prompts). Everything here should be read as "interesting signal worth investigating further," not as a settled empirical claim.

---

## References

- [Olsson et al. — "In-context Learning and Induction Heads" (2022)](https://arxiv.org/abs/2209.11895) — the primary inspiration; defines the induction score and makes the universality claim I'm probing
- [Elhage et al. — "A Mathematical Framework for Transformer Circuits" (2021)](https://arxiv.org/abs/2106.09685) — the compositional view of attention circuits that motivates looking at relative layer depth
- [Conmy et al. — "Towards Automated Circuit Discovery for Mechanistic Interpretability" (2023)](https://arxiv.org/abs/2304.14997) — context for patching methodology and the limits of ablation-based causal claims
- [Gould et al. — "Successor Heads: Recurring, Interpretable Attention Heads In The Wild" (2023)](https://arxiv.org/abs/2312.09230) — a more recent cross-model head analysis that informed how I thought about normalization across architectures
- [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens) — Neel Nanda's library; doing almost all the heavy lifting for hooking into attention patterns and running activation patches