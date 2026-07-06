experiments/run_all_models.py
"""
Entry point for the induction head universality probe.

Iterates over all 6 models, computes induction scores via induction_score.py,
and saves raw score arrays to results/scores/.

Usage:
    python experiments/run_all_models.py
    python experiments/run_all_models.py --models pythia-160m pythia-1.4b
    python experiments/run_all_models.py --device cuda --seq_len 512

NOTE: First run will download models (~4-8 GB total). Set HF_HOME to control
where they land.

Hardware used during dev: single A100 40GB. With seq_len=256 all 6 models fit
comfortably. Longer sequences may OOM on smaller GPUs — use --batch_size 1
and --seq_len 128 as a fallback.
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch

# Add project root so imports work regardless of where you run from
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.induction_score import compute_induction_scores
from src.models import load_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model registry
#
# Chosen to give us three "families" with two scales each:
#   - Pythia  (EleutherAI, same arch across scales, great for ablations)
#   - GPT-2   (OpenAI, BPE tokeniser, no RoPE, classic baseline)
#   - Llama   (Meta, RoPE + SwiGLU, modern recipe)
#
# The interesting question is whether induction circuits look the same across
# families when we control for scale, or whether architectural choices
# (positional encoding style, MLP variant, norm placement) leave a fingerprint.
#
# Small scales on purpose — we're doing activation patching and full attention
# score extraction, so we need things that run fast in a loop.
# ---------------------------------------------------------------------------
MODEL_REGISTRY = {
    # family -> list of (friendly_name, hf_model_id)
    "pythia": [
        ("pythia-160m", "EleutherAI/pythia-160m"),
        ("pythia-1.4b", "EleutherAI/pythia-1.4b"),
    ],
    "gpt2": [
        ("gpt2-small",  "gpt2"),
        ("gpt2-medium", "gpt2-medium"),
    ],
    "llama": [
        ("llama-3.2-1b", "meta-llama/Llama-3.2-1B"),
        ("llama-3.2-3b", "meta-llama/Llama-3.2-3B"),
    ],
}

# Flat view used for iteration
ALL_MODELS = [
    entry for family in MODEL_REGISTRY.values() for entry in family
]


def parse_args():
    p = argparse.ArgumentParser(description="Run induction score probes across model families")
    p.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Subset of model friendly-names to run (default: all 6). "
             "Example: --models pythia-160m gpt2-small",
    )
    p.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for inference (default: cuda if available)",
    )
    p.add_argument(
        "--seq_len",
        type=int,
        default=256,
        help="Sequence length for induction score sequences (default: 256). "
             "Must be even — we repeat the first half in the second half.",
    )
    p.add_argument(
        "--n_seqs",
        type=int,
        default=50,
        help="Number of random repeated sequences to average over (default: 50). "
             "More = less noisy scores but slower.",
    )
    p.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Sequences per forward pass (default: 8). Reduce if OOM.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for reproducibility",
    )
    p.add_argument(
        "--results_dir",
        type=Path,
        default=Path("results/scores"),
        help="Directory to write score arrays into (default: results/scores)",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-run even if results already exist for a model",
    )
    return p.parse_args()


def seed_everything(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    # Not using random module directly but good habit
    import random
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_models(requested: Optional[list[str]]) -> list[tuple[str, str]]:
    """Filter the registry down to the requested models, preserving family order."""
    if requested is None:
        return ALL_MODELS

    name_to_entry = {name: (name, hf_id) for name, hf_id in ALL_MODELS}
    missing = [m for m in requested if m not in name_to_entry]
    if missing:
        valid = [n for n, _ in ALL_MODELS]
        log.error("Unknown model names: %s", missing)
        log.error("Valid names: %s", valid)
        sys.exit(1)

    return [name_to_entry[m] for m in requested]


def results_path(results_dir: Path, model_name: str) -> Path:
    """Where we write the per-model score array."""
    return results_dir / f"{model_name}_scores.npz"


def metadata_path(results_dir: Path) -> Path:
    return results_dir / "run_metadata.json"


def run_model(
    model_name: str,
    hf_model_id: str,
    device: str,
    seq_len: int,
    n_seqs: int,
    batch_size: int,
    seed: int,
    out_path: Path,
):
    """
    Load one model, compute induction scores for every (layer, head), save.

    Returns a dict with summary stats so the outer loop can log progress.
    """
    log.info("=" * 60)
    log.info("Model: %s  (%s)", model_name, hf_model_id)
    log.info("=" * 60)

    t0 = time.time()

    # Load via our wrapper so TransformerLens hooked models are handled
    # consistently across families. This is the part that can be slow on
    # first run (model download).
    log.info("Loading model...")
    model = load_model(hf_model_id, device=device)
    load_time = time.time() - t0
    log.info("  Loaded in %.1fs  |  n_layers=%d  n_heads=%d",
             load_time, model.cfg.n_layers, model.cfg.n_heads)

    # Reseed before each model so score arrays are reproducible regardless
    # of which models we ran before this one in the same process.
    seed_everything(seed)

    log.info("Computing induction scores (n_seqs=%d, seq_len=%d, batch=%d)...",
             n_seqs, seq_len, batch_size)
    t1 = time.time()

    # scores shape: (n_layers, n_heads)
    # mean_score: scalar average across all heads (headline number)
    scores, per_seq_scores = compute_induction_scores(
        model=model,
        seq_len=seq_len,
        n_seqs=n_seqs,
        batch_size=batch_size,
        device=device,
    )

    score_time = time.time() - t1
    log.info("  Score computation: %.1fs", score_time)

    # Quick sanity print — useful to eyeball while the run is going
    log.info("  Score matrix shape: %s", scores.shape)
    log.info("  Mean induction score (all heads): %.4f", float(scores.mean()))
    log.info("  Max  induction score (best head): %.4f", float(scores.max()))

    # Find the top-5 heads by score — this is the main thing we'll compare
    # across models in the analysis notebook
    flat_idx = np.argsort(scores.ravel())[::-1][:5]
    top_heads = [(int(i // scores.shape[1]), int(i % scores.shape[1])) for i in flat_idx]
    log.info("  Top 5 heads (layer, head): %s", top_heads)
    log.info("  Top 5 scores: %s",
             [f"{scores[l, h]:.4f}" for l, h in top_heads])

    # Persist raw arrays so the analysis notebook doesn't need to rerun
    np.savez_compressed(
        out_path,
        scores=scores,                       # (n_layers, n_heads)
        per_seq_scores=per_seq_scores,        # (n_seqs, n_layers, n_heads)
        top_heads=np.array(top_heads),        # (5, 2)
    )
    log.info("  Saved to %s", out_path)

    # Free GPU memory before loading the next model
    del model
    if device == "cuda":
        torch.cuda.empty_cache()

    return {
        "model_name": model_name,
        "hf_model_id": hf_model_id,
        "mean_score": float(scores.mean()),
        "max_score": float(scores.max()),
        "top_heads": top_heads,
        "load_time_s": round(load_time, 2),
        "score_time_s": round(score_time, 2),
        "scores_shape": list(scores.shape),
    }


def main():
    args = parse_args()

    log.info("Induction head universality probe")
    log.info("Device: %s", args.device)
    log.info("Seed:   %d", args.seed)

    if args.seq_len % 2 != 0:
        log.error("--seq_len must be even (we split sequence into two halves). Got %d", args.seq_len)
        sys.exit(1)

    models_to_run = select_models(args.models)
    log.info("Models selected: %s", [n for n, _ in models_to_run])

    args.results_dir.mkdir(parents=True, exist_ok=True)
    log.info("Results dir: %s", args.results_dir.resolve())

    # Log CUDA info if available — good to have in run logs for reproducibility
    if args.device == "cuda":
        log.info("CUDA device: %s", torch.cuda.get_device_name(0))
        log.info("CUDA memory: %.1f GB", torch.cuda.get_device_properties(0).total_memory / 1e9)

    run_results = []
    skipped = []
    failed = []

    total_t0 = time.time()

    for model_name, hf_model_id in models_to_run:
        out_path = results_path(args.results_dir, model_name)

        if out_path.exists() and not args.overwrite:
            log.info("Skipping %s — results already exist at %s. "
                     "Use --overwrite to force.", model_name, out_path)
            skipped.append(model_name)
            continue

        try:
            result = run_model(
                model_name=model_name,
                hf_model_id=hf_model_id,
                device=args.device,
                seq_len=args.seq_len,
                n_seqs=args.n_seqs,
                batch_size=args.batch_size,
                seed=args.seed,
                out_path=out_path,
            )
            run_results.append(result)

        except torch.cuda.OutOfMemoryError:
            log.error("OOM on %s. Try --batch_size 1 or --seq_len 128.", model_name)
            failed.append(model_name)
            if args.device == "cuda":
                torch.cuda.empty_cache()

        except Exception as exc:
            log.exception("Unexpected error running %s: %s", model_name, exc)
            failed.append(model_name)

    total_elapsed = time.time() - total_t0

    # ------------------------------------------------------------------
    # Summary table — quick look at scores across models
    # ------------------------------------------------------------------
    if run_results:
        log.info("")
        log.info("=" * 60)
        log.info("SUMMARY")
        log.info("=" * 60)
        log.info("%-20s  %8s  %8s", "Model", "MeanScore", "MaxScore")
        log.info("-" * 42)
        for r in run_results:
            log.info("%-20s  %8.4f  %8.4f",
                     r["model_name"], r["mean_score"], r["max_score"])
        log.info("-" * 42)
        log.info("Total wall time: %.1f min", total_elapsed / 60)

    if skipped:
        log.info("Skipped (already done): %s", skipped)
    if failed:
        log.warning("Failed: %s  — check logs above for details", failed)

    # ------------------------------------------------------------------
    # Write run metadata so the analysis notebook knows what config
    # produced these scores. Especially important once we start varying
    # seq_len and n_seqs to test stability.
    # ------------------------------------------------------------------
    metadata = {
        "run_config": {
            "seq_len": args.seq_len,
            "n_seqs": args.n_seqs,
            "batch_size": args.batch_size,
            "seed": args.seed,
            "device": args.device,
        },
        "model_registry": {
            family: entries
            for family, entries in MODEL_REGISTRY.items()
        },
        "results": run_results,
        "skipped": skipped,
        "failed": failed,
        "total_elapsed_s": round(total_elapsed, 2),
    }
    meta_path = metadata_path(args.results_dir)
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    log.info("Run metadata written to %s", meta_path)

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()