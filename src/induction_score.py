"""
induction_score.py

Core scoring function for the induction head universality probe.

The idea: induction heads are supposedly universal across transformer
architectures (see Olsson et al. 2022 "In-context Learning and Induction Heads").
But are they really? Or does the training setup / architecture family leave
a structural fingerprint on *which* heads do the work and *how strongly*?

This file: given any TransformerLens model, run the standard repeated-token
sequence and compute per-head induction scores. Returns a (n_layers, n_heads)
array that we can then compare across model families.

Induction score definition (following Olsson et al.):
  For a sequence [A, B, ..., A, B'], the induction head should attend
  strongly from B' back to B (i.e., the token immediately after the
  previous occurrence of A). The score is the average attention weight
  to that "induction" position across all token positions.

NOTE: TransformerLens's run_with_cache gives us attention patterns directly,
but the cache key naming differs slightly between models. Handled below.
"""

import json
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import transformer_lens
from transformer_lens import HookedTransformer


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _make_repeated_token_sequence(
    seq_len: int,
    vocab_size: int,
    batch_size: int,
    device: torch.device,
    rng: np.random.Generator,
) -> torch.Tensor:
    """
    Build the canonical repeated-token sequence from Olsson et al.

    Format: [BOS, t_0, t_1, ..., t_{N-1}, t_0, t_1, ..., t_{N-1}]

    We use a random prefix of length seq_len // 2 then repeat it.
    BOS token prepended so positions line up with what the model expects.

    Why random tokens? We want to test whether the model has learned a
    *general* copy mechanism, not just memorised specific bigrams.
    Sampling from the full vocab makes accidental memorisation unlikely.
    """
    half = seq_len // 2

    # sample random tokens, avoiding special tokens (0 = BOS, 1 = EOS in most
    # HF tokenizers — using 2: to be safe, but this is a bit hacky)
    # TODO: make special token exclusion model-aware
    rand_tokens = rng.integers(low=2, high=vocab_size - 1, size=(batch_size, half))
    rand_tokens_t = torch.tensor(rand_tokens, dtype=torch.long, device=device)

    # BOS prepended — shape: (batch, 1 + half + half) = (batch, 1 + seq_len)
    bos = torch.zeros(batch_size, 1, dtype=torch.long, device=device)
    seq = torch.cat([bos, rand_tokens_t, rand_tokens_t], dim=1)

    return seq


def _get_attention_patterns(
    model: HookedTransformer,
    tokens: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """
    Run forward pass and pull attention patterns from the cache.

    Returns dict keyed by 'blocks.{layer}.attn.hook_pattern' with
    tensors of shape (batch, heads, seq, seq).

    TransformerLens sometimes calls these hook_attn_weights depending on
    the model config — checking both to be safe.
    """
    with torch.no_grad():
        _, cache = model.run_with_cache(
            tokens,
            names_filter=lambda name: "hook_pattern" in name or "hook_attn_weights" in name,
            return_type=None,  # don't need logits
        )

    return dict(cache)


# ---------------------------------------------------------------------------
# main scoring function
# ---------------------------------------------------------------------------

def compute_induction_scores(
    model: HookedTransformer,
    seq_len: int = 50,
    batch_size: int = 8,
    seed: int = 42,
    device: Optional[str] = None,
    verbose: bool = True,
) -> np.ndarray:
    """
    Compute per-head induction scores for a TransformerLens model.

    Returns
    -------
    scores : np.ndarray of shape (n_layers, n_heads)
        Each entry is the mean attention weight to the induction position
        averaged over all valid query positions (positions in the second
        half of the repeated sequence).

    Parameters
    ----------
    model       : loaded HookedTransformer (already on device)
    seq_len     : length of the repeated half (total seq = 2*seq_len + 1 with BOS)
    batch_size  : number of random sequences to average over
    seed        : for reproducibility
    device      : if None, inferred from model parameters
    verbose     : print layer/head progress + summary stats

    Implementation notes
    --------------------
    The "induction position" for a query at position p (in the second half)
    is p - seq_len. So if the sequence is:
        [BOS, t0, t1, ..., t_{N-1}, t0, t1, ..., t_{N-1}]
         pos0  1   2        N       N+1 N+2       2N

    And we're computing attention at position q (q > N), the induction target
    is position q - N (the position of the same token in the first half, which
    is immediately after its predecessor — the classic A, B -> A, ? pattern).

    We only score positions in the second half (seq_len+1 ... 2*seq_len)
    because that's where the induction signal can actually fire.
    """

    rng = np.random.default_rng(seed)

    if device is None:
        device = next(model.parameters()).device
    else:
        device = torch.device(device)

    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    vocab_size = model.cfg.d_vocab

    if verbose:
        print(f"Model: {model.cfg.model_name}")
        print(f"Layers: {n_layers}, Heads: {n_heads}, Vocab: {vocab_size}")
        print(f"Sequence length (half): {seq_len}, Batch: {batch_size}, Seed: {seed}")
        print(f"Device: {device}")
        print()

    tokens = _make_repeated_token_sequence(
        seq_len=seq_len,
        vocab_size=vocab_size,
        batch_size=batch_size,
        device=device,
        rng=rng,
    )
    total_seq_len = tokens.shape[1]  # 1 + 2 * seq_len

    if verbose:
        print(f"Token sequence shape: {tokens.shape}")
        print(f"Running forward pass with cache...")

    t0 = time.time()
    attn_patterns = _get_attention_patterns(model, tokens)
    t1 = time.time()

    if verbose:
        print(f"Forward pass done in {t1 - t0:.2f}s")
        print(f"Cache keys found: {len(attn_patterns)}")
        # spot check — make sure we got the right thing
        sample_key = next(iter(attn_patterns))
        print(f"  sample key: {sample_key} | shape: {attn_patterns[sample_key].shape}")
        print()

    # The "second half" positions start at index (seq_len + 1) because of BOS.
    # For query position q, the induction target is at q - seq_len.
    # We average attention[q, q - seq_len] across q in the second half.
    second_half_start = seq_len + 1  # inclusive

    scores = np.zeros((n_layers, n_heads), dtype=np.float32)

    for layer in range(n_layers):
        # Try both naming conventions TransformerLens uses
        key = f"blocks.{layer}.attn.hook_pattern"
        if key not in attn_patterns:
            key = f"blocks.{layer}.attn.hook_attn_weights"
        if key not in attn_patterns:
            # some models use hook_z or other names — log and skip
            print(f"  WARNING: could not find attention pattern for layer {layer}")
            print(f"  Available keys (first 5): {list(attn_patterns.keys())[:5]}")
            continue

        # attn shape: (batch, heads, q_pos, k_pos)
        attn = attn_patterns[key]  # (B, H, S, S)

        layer_scores = []
        for q_pos in range(second_half_start, total_seq_len):
            induction_k_pos = q_pos - seq_len
            # attn[:, :, q_pos, induction_k_pos] → (B, H)
            induction_attn = attn[:, :, q_pos, induction_k_pos]  # (B, H)
            layer_scores.append(induction_attn.cpu().float().numpy())

        # layer_scores: list of (B, H) arrays, one per valid q_pos
        # stack → (n_positions, B, H), then mean over positions and batch
        layer_scores_arr = np.stack(layer_scores, axis=0)  # (P, B, H)
        scores[layer] = layer_scores_arr.mean(axis=(0, 1))  # (H,)

        if verbose:
            top_head = int(np.argmax(scores[layer]))
            print(
                f"  Layer {layer:2d} | "
                f"max score: {scores[layer].max():.4f} (head {top_head}) | "
                f"mean: {scores[layer].mean():.4f}"
            )

    if verbose:
        print()
        global_max_layer, global_max_head = np.unravel_index(
            np.argmax(scores), scores.shape
        )
        print(
            f"Top induction head: layer {global_max_layer}, head {global_max_head} "
            f"(score={scores[global_max_layer, global_max_head]:.4f})"
        )

    return scores


# ---------------------------------------------------------------------------
# save / load utilities
# ---------------------------------------------------------------------------

def save_scores(
    scores: np.ndarray,
    model_name: str,
    output_dir: str | Path = "results",
    metadata: Optional[dict] = None,
) -> Path:
    """
    Save induction scores to disk as both .npy (for fast reloading) and
    a JSON sidecar with metadata.

    Naming convention: {model_name_sanitised}_induction_scores.npy
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # sanitise model name for filesystem
    safe_name = model_name.replace("/", "_").replace(" ", "_").lower()

    npy_path = output_dir / f"{safe_name}_induction_scores.npy"
    json_path = output_dir / f"{safe_name}_induction_scores_meta.json"

    np.save(npy_path, scores)

    meta = {
        "model_name": model_name,
        "scores_shape": list(scores.shape),
        "n_layers": scores.shape[0],
        "n_heads": scores.shape[1],
        "global_max": float(scores.max()),
        "global_mean": float(scores.mean()),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    if metadata:
        meta.update(metadata)

    with open(json_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved scores → {npy_path}")
    print(f"Saved metadata → {json_path}")

    return npy_path


def load_scores(model_name: str, output_dir: str | Path = "results") -> np.ndarray:
    """Load previously saved scores."""
    output_dir = Path(output_dir)
    safe_name = model_name.replace("/", "_").replace(" ", "_").lower()
    npy_path = output_dir / f"{safe_name}_induction_scores.npy"

    if not npy_path.exists():
        raise FileNotFoundError(
            f"No saved scores found for {model_name!r} at {npy_path}. "
            f"Run compute_induction_scores first."
        )

    return np.load(npy_path)


# ---------------------------------------------------------------------------
# quick sanity check when run directly
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    """
    Quick smoke test on GPT-2 small — small enough to run on CPU in
    reasonable time, well-studied so we know roughly what to expect.

    Expected: induction heads should appear in layers 5-6 (based on
    Olsson et al. Figure 1 for GPT-2 small). Scores for those heads
    should be >> 0.1 vs ~0.01-0.03 for non-induction heads.
    """
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="gpt2",
        help="TransformerLens model name (default: gpt2)",
    )
    parser.add_argument("--seq-len", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save", action="store_true", help="Save results to results/")
    args = parser.parse_args()

    print(f"Loading model: {args.model}")
    print("(this may take a moment on first load — HF weights download)")
    print()

    model = HookedTransformer.from_pretrained(
        args.model,
        center_unembed=True,
        center_writing_weights=True,
        fold_ln=True,
        refactor_factored_attn_matrices=False,  # keep W_Q, W_K separate for later analysis
    )
    model.eval()

    scores = compute_induction_scores(
        model,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        seed=args.seed,
        verbose=True,
    )

    print()
    print("Full score matrix (layers x heads):")
    print(np.round(scores, 4))

    if args.save:
        save_scores(
            scores,
            model_name=args.model,
            metadata={
                "seq_len": args.seq_len,
                "batch_size": args.batch_size,
                "seed": args.seed,
            },
        )