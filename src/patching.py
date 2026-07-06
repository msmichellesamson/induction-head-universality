src/patching.py

import torch
import numpy as np
from transformer_lens import HookedTransformer, ActivationCache
from typing import Optional
import json
from pathlib import Path
from datetime import datetime

# Activation patching to confirm causal role of specific (layer, head) pairs
# in ICL. The core idea: if a head is truly causally responsible for in-context
# learning, zeroing its output should tank performance on ICL tasks but leave
# clean (non-ICL) tasks relatively intact.
#
# Inspired by Wang et al. (2022) "Interpretability in the Wild" and the
# original Elhage et al. induction heads paper. We want to see if the same
# causal story holds across model families or if Pythia vs. GPT-2 vs. Llama
# have structurally different induction circuits.
#
# NOTE: This is zero-ablation, not mean-ablation. Easier to implement and
# interpret, but noisier — zero is out-of-distribution for most activations.
# May revisit with mean-ablation if the signal is too dirty.


def build_icl_token_sequence(model: HookedTransformer, repeat_n: int = 3) -> torch.Tensor:
    """
    Build a simple repeated-token sequence for probing induction behavior.
    Pattern: [A B C] [A B C] [A B C] ... then query with A -> expect B.

    Using the classic "repeated random tokens" setup from Olsson et al. 2022.
    Not trying to be clever here — just want a clean induction signal.

    repeat_n: how many times to repeat the AB pair before the query
    """
    # Pick a random "A" and "B" token (avoiding special tokens, BOS/EOS, etc.)
    # We'll use token IDs in a safe mid-range to avoid tokenizer weirdness
    vocab_size = model.cfg.d_vocab
    safe_range_start = 100
    safe_range_end = min(vocab_size - 100, 5000)

    rng = np.random.default_rng(42)  # fixed seed so sequences are reproducible
    token_A = int(rng.integers(safe_range_start, safe_range_end))
    token_B = int(rng.integers(safe_range_start, safe_range_end))
    while token_B == token_A:
        token_B = int(rng.integers(safe_range_start, safe_range_end))

    # Sequence: [BOS, A, B, A, B, ..., A] — we'll measure P(B | ... A) at final pos
    tokens = [model.tokenizer.bos_token_id] if model.tokenizer.bos_token_id is not None else []
    for _ in range(repeat_n):
        tokens.extend([token_A, token_B])
    tokens.append(token_A)  # final query token

    print(f"  ICL sequence: token_A={token_A}, token_B={token_B}, length={len(tokens)}")
    print(f"  Sequence: {tokens}")

    return torch.tensor(tokens).unsqueeze(0)  # shape [1, seq_len]


def get_logit_for_target(logits: torch.Tensor, target_token_id: int) -> float:
    """
    Extract the logit for a specific target token at the final sequence position.
    logits shape: [batch, seq_len, vocab]
    """
    final_logits = logits[0, -1, :]  # last position, first (only) batch
    return final_logits[target_token_id].item()


def zero_ablate_head(
    model: HookedTransformer,
    tokens: torch.Tensor,
    layer: int,
    head: int,
    target_token_id: int,
) -> dict:
    """
    Zero-ablate the output of head `head` in layer `layer` and measure the
    change in logit for `target_token_id`.

    We hook into the z activation (attention head output before projection)
    and set it to zero for the specific head. This isolates that head's
    contribution cleanly.

    Returns a dict with baseline and ablated logits + the drop.
    """

    # First: clean forward pass to get baseline
    with torch.no_grad():
        clean_logits = model(tokens)
    baseline_logit = get_logit_for_target(clean_logits, target_token_id)

    # Hook: zero out head `head` in layer `layer`
    # The z tensor at hook_z has shape [batch, seq_len, n_heads, d_head]
    def zero_ablation_hook(z, hook):
        # z: [batch, seq_len, n_heads, d_head]
        z[:, :, head, :] = 0.0
        return z

    hook_name = f"blocks.{layer}.attn.hook_z"

    with torch.no_grad():
        ablated_logits = model.run_with_hooks(
            tokens,
            fwd_hooks=[(hook_name, zero_ablation_hook)],
        )

    ablated_logit = get_logit_for_target(ablated_logits, target_token_id)
    logit_drop = baseline_logit - ablated_logit

    return {
        "layer": layer,
        "head": head,
        "baseline_logit": baseline_logit,
        "ablated_logit": ablated_logit,
        "logit_drop": logit_drop,
        "target_token_id": target_token_id,
    }


def scan_all_heads(
    model: HookedTransformer,
    tokens: torch.Tensor,
    target_token_id: int,
    verbose: bool = True,
) -> list[dict]:
    """
    Run zero-ablation across all (layer, head) pairs and collect logit drops.
    This gives us a causal importance map, similar to the "attention knockout"
    approach in the induction heads paper.

    Slow but simple — iterating over all heads. For a model like Pythia-1.4B
    this is 24 layers * 16 heads = 384 forward passes. Takes a few minutes
    on CPU; manageable on GPU.

    NOTE: Not batching the ablations because the hook closure over (layer, head)
    gets tricky. Could optimize later if this becomes a bottleneck.
    """
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    results = []
    for layer in range(n_layers):
        for head in range(n_heads):
            result = zero_ablate_head(model, tokens, layer, head, target_token_id)
            results.append(result)
            if verbose:
                drop = result["logit_drop"]
                marker = " <-- HIGH" if drop > 2.0 else ""
                print(f"  L{layer:02d}H{head:02d}: drop={drop:+.3f}{marker}")

    return results


def patching_experiment(
    model: HookedTransformer,
    model_name: str,
    repeat_n: int = 3,
    results_dir: str = "results",
) -> dict:
    """
    Full patching experiment for a single model.

    Builds the ICL sequence, runs baseline, scans all heads, saves results.

    Returns the results dict for downstream analysis / plotting.
    """
    print(f"\n{'='*60}")
    print(f"Patching experiment: {model_name}")
    print(f"{'='*60}")

    tokens = build_icl_token_sequence(model, repeat_n=repeat_n)

    # The target is the second token in the AB pair (token_B), which should
    # be predicted at the final position if the model has learned induction.
    # We grab token_B from the sequence: it's at position 2 (0=BOS, 1=A, 2=B, ...)
    # Actually let's just look at what the model predicts first — sanity check.
    with torch.no_grad():
        clean_logits = model(tokens)

    final_pos_logits = clean_logits[0, -1, :]
    top5_ids = final_pos_logits.topk(5).indices.tolist()
    top5_vals = final_pos_logits.topk(5).values.tolist()
    print(f"  Top-5 predictions at final pos: {list(zip(top5_ids, [f'{v:.2f}' for v in top5_vals]))}")

    # token_B is at index 2 in the sequence (BOS=0, A=1, B=2)
    bos_offset = 1 if model.tokenizer.bos_token_id is not None else 0
    target_token_id = tokens[0, bos_offset + 1].item()  # this is token_B
    baseline_logit = get_logit_for_target(clean_logits, target_token_id)

    print(f"  Target token_B id={target_token_id}, baseline logit={baseline_logit:.3f}")
    print(f"  Is token_B in top-5? {target_token_id in top5_ids}")

    # Run the full head scan
    print(f"\n  Scanning all heads...")
    head_results = scan_all_heads(model, tokens, target_token_id, verbose=True)

    # Sort by logit drop (most causal first)
    head_results_sorted = sorted(head_results, key=lambda x: x["logit_drop"], reverse=True)

    print(f"\n  Top 5 most causally important heads:")
    for r in head_results_sorted[:5]:
        print(f"    L{r['layer']:02d}H{r['head']:02d}: logit_drop={r['logit_drop']:+.3f}")

    experiment_result = {
        "model_name": model_name,
        "n_layers": model.cfg.n_layers,
        "n_heads": model.cfg.n_heads,
        "d_model": model.cfg.d_model,
        "repeat_n": repeat_n,
        "target_token_id": target_token_id,
        "baseline_logit": baseline_logit,
        "token_B_in_top5": target_token_id in top5_ids,
        "head_results": head_results,
        "timestamp": datetime.now().isoformat(),
    }

    # Save to disk
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    safe_model_name = model_name.replace("/", "_").replace("-", "_")
    out_path = Path(results_dir) / f"patching_{safe_model_name}.json"
    with open(out_path, "w") as f:
        json.dump(experiment_result, f, indent=2)
    print(f"\n  Results saved to {out_path}")

    return experiment_result


def plot_logit_drop_heatmap(
    results: dict,
    save_path: Optional[str] = None,
):
    """
    Plot a (layer x head) heatmap of logit drops.

    High values = head is causally important for this ICL task.
    Should see hot spots at induction head locations identified by
    induction_score.py — if not, something is off.
    """
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    model_name = results["model_name"]
    n_layers = results["n_layers"]
    n_heads = results["n_heads"]
    head_results = results["head_results"]

    # Build (n_layers, n_heads) matrix
    drop_matrix = np.zeros((n_layers, n_heads))
    for r in head_results:
        drop_matrix[r["layer"], r["head"]] = r["logit_drop"]

    fig, ax = plt.subplots(figsize=(max(8, n_heads * 0.6), max(6, n_layers * 0.4)))

    # Diverging colormap centered at 0 — want to see both helpful and harmful
    # heads. Some heads may have *negative* drop (ablating them actually helps),
    # which is interesting and worth investigating separately.
    vmax = max(abs(drop_matrix.max()), abs(drop_matrix.min()), 1.0)
    im = ax.imshow(
        drop_matrix,
        aspect="auto",
        cmap="RdYlGn_r",  # red = high drop (important), green = low drop / negative
        vmin=-vmax,
        vmax=vmax,
        interpolation="nearest",
    )

    ax.set_xlabel("Head", fontsize=12)
    ax.set_ylabel("Layer", fontsize=12)
    ax.set_title(f"ICL Logit Drop by Head — {model_name}\n(red = head causally important for ICL)", fontsize=11)
    ax.set_xticks(range(n_heads))
    ax.set_yticks(range(n_layers))

    plt.colorbar(im, ax=ax, label="Logit drop (baseline − ablated)")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Heatmap saved to {save_path}")
    else:
        plt.show()

    return fig, ax


def compare_models_patching(results_list: list[dict]) -> None:
    """
    Quick comparison across models: where do the top-k causal heads live?

    The hypothesis: if induction heads are universal, the same (layer/total_layers)
    and (head_idx) patterns should emerge across architectures. If model family
    leaves a fingerprint, we'd expect Pythia heads to cluster differently than
    GPT-2 heads in normalized layer depth.

    This is the interesting question — let's see what the data says.
    """
    print("\n" + "="*60)
    print("Cross-model causal head comparison")
    print("="*60)

    for results in results_list:
        model_name = results["model_name"]
        n_layers = results["n_layers"]
        n_heads = results["n_heads"]
        head_results = results["head_results"]

        sorted_heads = sorted(head_results, key=lambda x: x["logit_drop"], reverse=True)
        top3 = sorted_heads[:3]

        print(f"\n{model_name} ({n_layers}L x {n_heads}H):")
        for r in top3:
            # Normalized layer depth [0, 1]
            norm_depth = r["layer"] / (n_layers - 1) if n_layers > 1 else 0.0
            print(
                f"  L{r['layer']:02d}H{r['head']:02d} "
                f"| norm_depth={norm_depth:.2f} "
                f"| drop={r['logit_drop']:+.3f}"
            )

    # TODO: more rigorous statistical comparison — KS test on depth distributions?
    # The eyeball test is suggestive but not convincing. Want to see if top-k heads
    # cluster at similar normalized depths across model families.


# ─── quick sanity check ───────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    # Use models.py to load — avoids duplicating the loading logic
    # Just test with GPT-2 small here since it loads fast
    print("Loading GPT-2 small for patching sanity check...")
    model = HookedTransformer.from_pretrained("gpt2", center_unembed=True, center_writing_weights=True)
    model.eval()

    if torch.cuda.is_available():
        model = model.cuda()
        print("Using CUDA")
    else:
        print("Using CPU (will be slow for full scan)")

    results = patching_experiment(
        model,
        model_name="gpt2-small",
        repeat_n=3,
        results_dir="results",
    )

    # Spot check: known induction heads in GPT-2 small are around L5H1 and L5H5
    # (from Olsson et al.). See if we recover those.
    sorted_heads = sorted(results["head_results"], key=lambda x: x["logit_drop"], reverse=True)
    print(f"\nTop 5 causal heads recovered:")
    for r in sorted_heads[:5]:
        print(f"  L{r['layer']:02d}H{r['head']:02d}: drop={r['logit_drop']:+.3f}")

    plot_logit_drop_heatmap(results, save_path="results/patching_gpt2_small.png")

    print("\nDone. Check results/ directory for outputs.")