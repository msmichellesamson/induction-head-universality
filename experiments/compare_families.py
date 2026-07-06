"""
Cross-family induction head comparison.

After running run_all_models.py, we have score arrays saved per model.
This script loads them, computes relative layer depth of the "top" induction
heads, and makes the comparison plot that's the main figure in the writeup.

Relative layer depth = (layer_idx of top head) / (n_layers - 1)
This normalizes across models with different depths so we can compare
where in the network induction heads tend to live.

Hypothesis going in: universality claims from Olsson et al. (2022) suggest
induction heads should appear at similar *relative* depths regardless of
architecture. But the paper mostly looked at attention-only transformers.
Real models (Pythia, GPT-2, Llama) have MLPs, different positional encodings,
and different training data. Does the structural fingerprint still show up?

Spoiler from my runs: yes, but with some interesting variation in Llama.
See findings at the bottom of this file / in the README.
"""

import json
import os
import sys
from pathlib import Path
from typing import Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# Use non-interactive backend if running headless (e.g., on a server)
# I kept forgetting this and getting Tcl/Tk errors - this is the lazy fix
matplotlib.use("Agg")

# Add src to path - not using a proper package because this is exploratory
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# fmt: off
RESULTS_DIR = Path(__file__).parent / "results"
FIGURES_DIR = Path(__file__).parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

# These match what run_all_models.py saved.
# model_id -> (display_name, model_family, n_layers)
# n_layers is included here as a sanity check against what's in the JSON
MODEL_REGISTRY = {
    "gpt2":                    ("GPT-2 Small",      "GPT-2",   12),
    "gpt2-medium":             ("GPT-2 Medium",     "GPT-2",   24),
    "gpt2-large":              ("GPT-2 Large",      "GPT-2",   36),
    "EleutherAI/pythia-160m":  ("Pythia-160M",      "Pythia",   6),
    "EleutherAI/pythia-410m":  ("Pythia-410M",      "Pythia",  24),
    "EleutherAI/pythia-1.4b":  ("Pythia-1.4B",      "Pythia",  24),
    "EleutherAI/pythia-2.8b":  ("Pythia-2.8B",      "Pythia",  32),
    "meta-llama/Llama-3.2-1B": ("Llama-3.2-1B",     "Llama",   16),
    "meta-llama/Llama-3.2-3B": ("Llama-3.2-3B",     "Llama",   28),
}

# Color / marker by family so the plot is readable
FAMILY_STYLE = {
    "GPT-2":  {"color": "#1f77b4", "marker": "o", "linestyle": "-"},
    "Pythia": {"color": "#2ca02c", "marker": "s", "linestyle": "--"},
    "Llama":  {"color": "#d62728", "marker": "^", "linestyle": "-."},
}

# Threshold for calling a head an "induction head"
# Olsson et al. use 0.4 on their metric; I'm using the same convention
# but worth noting this is somewhat arbitrary - see notes in induction_score.py
INDUCTION_THRESHOLD = 0.4
# fmt: on


def load_scores(model_id: str) -> Optional[dict]:
    """
    Load the saved score dict for a model.

    run_all_models.py saves:
        {
          "model_id": ...,
          "n_layers": ...,
          "n_heads": ...,
          "scores": [[float, ...], ...],   # shape (n_layers, n_heads)
          "top_heads": [[layer, head], ...],
          "metadata": {...}
        }

    Returns None if file doesn't exist (model might not have been run yet).
    """
    # Sanitize model_id for use as filename - the same way run_all_models.py does
    safe_id = model_id.replace("/", "_").replace("-", "_")
    fpath = RESULTS_DIR / f"{safe_id}_scores.json"

    if not fpath.exists():
        print(f"  [missing] {fpath.name} - skipping {model_id}")
        return None

    with open(fpath) as f:
        data = json.load(f)

    # Sanity check: n_layers should match our registry
    expected_layers = MODEL_REGISTRY[model_id][2]
    if data["n_layers"] != expected_layers:
        print(
            f"  [warn] {model_id}: expected {expected_layers} layers, "
            f"got {data['n_layers']} - registry may be stale"
        )

    return data


def compute_relative_layer_depth(scores_2d: list[list[float]], n_layers: int) -> float:
    """
    Find the top induction head(s) and return their average relative layer depth.

    'Relative layer depth' = layer_index / (n_layers - 1), so layer 0 is 0.0
    and the last layer is 1.0.

    Why average? Some models have multiple strong induction heads (I saw up to 4
    in Pythia-2.8B). Taking the max-scoring head alone felt cherry-picky, but
    averaging all heads above threshold captures the 'center of mass' of induction
    circuits in the model.

    NOTE: If no head clears the threshold, returns NaN. This happened for
    Pythia-160M in early runs - the model seems to not develop clean induction
    heads at that scale, which is itself interesting.
    """
    scores_arr = np.array(scores_2d)  # (n_layers, n_heads)

    # Find all heads above threshold
    above_thresh = np.argwhere(scores_arr >= INDUCTION_THRESHOLD)

    if len(above_thresh) == 0:
        return float("nan")

    # Weight by score so that strong heads count more
    # Not sure this is the right choice vs. simple average - try both?
    layer_indices = above_thresh[:, 0].astype(float)
    head_scores = scores_arr[above_thresh[:, 0], above_thresh[:, 1]]

    weighted_layer = np.average(layer_indices, weights=head_scores)
    relative_depth = weighted_layer / (n_layers - 1)

    return float(relative_depth)


def compute_max_score(scores_2d: list[list[float]]) -> tuple[float, int, int]:
    """Return (max_score, layer, head) of the single strongest induction head."""
    scores_arr = np.array(scores_2d)
    flat_idx = np.argmax(scores_arr)
    layer, head = np.unravel_index(flat_idx, scores_arr.shape)
    return float(scores_arr[layer, head]), int(layer), int(head)


def compute_n_induction_heads(scores_2d: list[list[float]]) -> int:
    """Count how many heads clear the induction threshold."""
    return int(np.sum(np.array(scores_2d) >= INDUCTION_THRESHOLD))


def collect_family_data() -> dict[str, list[dict]]:
    """
    Load all available models and group stats by family.

    Returns:
        {
          "GPT-2": [
            {
              "model_id": ...,
              "display_name": ...,
              "n_layers": ...,
              "n_params_approx": ...,   # for x-axis ordering within family
              "rel_depth": ...,
              "max_score": ...,
              "top_layer": ...,
              "top_head": ...,
              "n_induction_heads": ...
            },
            ...
          ],
          "Pythia": [...],
          "Llama": [...],
        }
    """
    family_data: dict[str, list[dict]] = {fam: [] for fam in FAMILY_STYLE}

    for model_id, (display_name, family, n_layers) in MODEL_REGISTRY.items():
        print(f"Loading {display_name}...")
        data = load_scores(model_id)
        if data is None:
            continue

        scores_2d = data["scores"]
        rel_depth = compute_relative_layer_depth(scores_2d, data["n_layers"])
        max_score, top_layer, top_head = compute_max_score(scores_2d)
        n_induction = compute_n_induction_heads(scores_2d)

        entry = {
            "model_id": model_id,
            "display_name": display_name,
            "n_layers": data["n_layers"],
            "rel_depth": rel_depth,
            "max_score": max_score,
            "top_layer": top_layer,
            "top_head": top_head,
            "n_induction_heads": n_induction,
        }

        print(
            f"  rel_depth={rel_depth:.3f}  max_score={max_score:.3f}  "
            f"top=L{top_layer}H{top_head}  n_induction={n_induction}"
        )

        family_data[family].append(entry)

    return family_data


def plot_relative_depth_by_family(
    family_data: dict[str, list[dict]],
    save_path: Optional[Path] = None,
) -> None:
    """
    Main comparison figure: relative layer depth of top induction heads,
    plotted by model (x) and grouped by family (color/marker).

    x-axis is model index within family, ordered by n_layers (proxy for scale)
    y-axis is relative layer depth (0 = first layer, 1 = last layer)

    The universality claim from Olsson et al. predicts these should all be
    roughly similar (around 0.25-0.5 based on their attention-only results).
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax_depth = axes[0]
    ax_score = axes[1]

    for family, entries in family_data.items():
        if not entries:
            continue

        # Sort by n_layers within family
        entries_sorted = sorted(entries, key=lambda e: e["n_layers"])
        style = FAMILY_STYLE[family]

        labels = [e["display_name"].split(" ", 1)[1] for e in entries_sorted]
        x = np.arange(len(entries_sorted))
        rel_depths = [e["rel_depth"] for e in entries_sorted]
        max_scores = [e["max_score"] for e in entries_sorted]

        # Handle NaN (Pythia-160M)
        valid = [(i, d) for i, d in enumerate(rel_depths) if not np.isnan(d)]
        x_valid = np.array([v[0] for v in valid])
        d_valid = np.array([v[1] for v in valid])

        ax_depth.plot(
            x_valid,
            d_valid,
            color=style["color"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            linewidth=1.8,
            markersize=8,
            label=family,
        )

        # Mark NaN points explicitly so they're visible
        nan_x = [i for i, d in enumerate(rel_depths) if np.isnan(d)]
        if nan_x:
            ax_depth.scatter(
                nan_x,
                [0.05] * len(nan_x),
                color=style["color"],
                marker="x",
                s=80,
                zorder=5,
                label=f"{family} (no induction heads found)",
            )

        ax_depth.set_xticks(x)
        ax_depth.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)

        # Max score subplot - same x axis
        ax_score.plot(
            x,
            max_scores,
            color=style["color"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            linewidth=1.8,
            markersize=8,
            label=family,
        )
        ax_score.set_xticks(x)
        ax_score.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)

    # Reference band: the 25th-50th percentile range from Olsson et al.
    # (read off from Figure 1 in their paper - these are approximate)
    ax_depth.axhspan(0.25, 0.50, alpha=0.08, color="grey", label="Olsson et al. range (approx)")
    ax_depth.axhline(0.375, color="grey", linestyle=":", linewidth=1, alpha=0.5)

    ax_depth.set_ylabel("Relative layer depth of top induction head(s)", fontsize=10)
    ax_depth.set_title("Where do induction heads live?\n(weighted avg over threshold heads)", fontsize=10)
    ax_depth.set_ylim(-0.05, 1.05)
    ax_depth.legend(fontsize=8, loc="upper left")
    ax_depth.grid(axis="y", alpha=0.3)

    ax_score.axhline(
        INDUCTION_THRESHOLD,
        color="grey",
        linestyle=":",
        linewidth=1.2,
        label=f"Threshold ({INDUCTION_THRESHOLD})",
    )
    ax_score.set_ylabel("Max induction score (across all heads)", fontsize=10)
    ax_score.set_title("How strong are the induction heads?", fontsize=10)
    ax_score.set_ylim(0.0, 1.05)
    ax_score.legend(fontsize=8)
    ax_score.grid(axis="y", alpha=0.3)

    fig.suptitle(
        "Induction Head Universality Across Architecture Families",
        fontsize=12,
        fontweight="bold",
        y=1.01,
    )

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"\nFigure saved to {save_path}")
    else:
        plt.show()

    plt.close(fig)


def plot_score_heatmaps(
    family_data: dict[str, list[dict]],
    save_dir: Optional[Path] = None,
) -> None:
    """
    Per-model heatmap of induction scores: (layer x head).

    These are the 'supporting figures' - one per model. They show exactly
    which heads are firing and how concentrated vs. distributed the induction
    signal is. Useful to look at before trusting the aggregate stats above.

    NOTE: This reloads the score arrays from disk, which is a bit redundant
    with collect_family_data(). Could refactor to pass them through, but
    this is exploratory code so keeping it simple.
    """
    all_entries = [e for entries in family_data.values() for e in entries]

    for entry in all_entries:
        data = load_scores(entry["model_id"])
        if data is None:
            continue

        scores_arr = np.array(data["scores"])  # (n_layers, n_heads)
        n_layers, n_heads = scores_arr.shape

        fig, ax = plt.subplots(figsize=(max(6, n_heads * 0.5), max(4, n_layers * 0.4)))

        im = ax.imshow(
            scores_arr,
            aspect="auto",
            cmap="viridis",
            vmin=0.0,
            vmax=1.0,
            interpolation="nearest",
        )

        # Mark the threshold contour
        ax.contour(
            scores_arr,
            levels=[INDUCTION_THRESHOLD],
            colors="white",
            linewidths=0.8,
            linestyles="--",
        )

        # Mark the top head
        top_layer = entry["top_layer"]
        top_head = entry["top_head"]
        ax.scatter(
            [top_head],
            [top_layer],
            s=120,
            color="red",
            marker="*",
            zorder=5,
            label=f"Top head (L{top_layer}H{top_head}, score={entry['max_score']:.3f})",
        )

        ax.set_xlabel("Head index", fontsize=9)
        ax.set_ylabel("Layer index", fontsize=9)
        ax.set_title(
            f"{entry['display_name']} — Induction Scores\n"
            f"n_induction_heads={entry['n_induction_heads']}, "
            f"rel_depth={entry['rel_depth']:.3f}",
            fontsize=9,
        )
        ax.legend(fontsize=8, loc="upper right")

        plt.colorbar(im, ax=ax, label="Induction score")

        if save_dir:
            safe_id = entry["model_id"].replace("/", "_").replace("-", "_")
            out_path = save_dir / f"heatmap_{safe_id}.png"
            fig.savefig(out_path, dpi=120, bbox_inches="tight")
            print(f"  Heatmap saved: {out_path.name}")
        else:
            plt.show()

        plt.close(fig)


def print_summary_table(family_data: dict[str, list[dict]]) -> None:
    """
    Print a quick summary table to stdout.

    I keep wanting to eyeball the numbers before trusting the plots,
    so having this here is useful during exploration.
    """
    print("\n" + "=" * 85)
    print(f"{'Model':<22} {'Family':<8} {'n_layers':>8} {'rel_depth':>10} "
          f"{'max_score':>10} {'n_ind_heads':>12} {'top_head':>10}")
    print("-" * 85)

    all_entries = []
    for family, entries in family_data.items():
        for e in entries:
            all_entries.append((family, e))

    # Sort by family, then n_layers
    all_entries.sort(key=lambda x: (x[0], x[1]["n_layers"]))

    for family, e in all_entries:
        rel_str = f"{e['rel_depth']:.4f}" if not np.isnan(e["rel_depth"]) else "  NaN  "
        print(
            f"{e['display_name']:<22} {family:<8} {e['n_layers']:>8} "
            f"{rel_str:>10} {e['max_score']:>10.4f} "
            f"{e['n_induction_heads']:>12} "
            f"L{e['top_layer']}H{e['top_head']:>2}"
        )

    print("=" * 85)

    # Quick cross-family stats
    print("\nFamily-level summary (mean ± std of relative depth):")
    for family, entries in family_data.items():
        if not entries:
            continue
        depths = [e["rel_depth"] for e in entries if not np.isnan(e["rel_depth"])]
        if depths:
            print(f"  {family:<8}: {np.mean(depths):.3f} ± {np.std(depths):.3f}  (n={len(depths)})")
        else:
            print(f"  {family:<8}: all NaN")

    print()


def save_summary_json(family_data: dict[str, list[dict]]) -> None:
    """
    Save a flat summary to JSON for use in the writeup / notebook.

    Separate from the per-model score files - this is the 'final' derived stats.
    """
    out = []
    for family, entries in family_data.items():
        for e in entries:
            out.append({
                "model_id": e["model_id"],
                "display_name": e["display_name"],
                "family": family,
                "n_layers": e["n_layers"],
                "rel_depth": None if np.isnan(e["rel_depth"]) else e["rel_depth"],
                "max_score": e["max_score"],
                "top_layer": e["top_layer"],
                "top_head": e["top_head"],
                "n_induction_heads": e["n_induction_heads"],
            })

    out_path = RESULTS_DIR / "cross_family_summary.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Summary JSON saved to {out_path}")


def main():
    print("Cross-family induction head comparison")
    print(f"Results dir: {RESULTS_DIR}")
    print(f"Threshold: {INDUCTION_THRESHOLD}\n")

    if not RESULTS_DIR.exists():
        print(f"ERROR: {RESULTS_DIR} doesn't exist. Run experiments/run_all_models.py first.")
        sys.exit(1)

    family_data = collect_family_data()

    n_loaded = sum(len(v) for v in family_data.values())
    if n_loaded == 0:
        print("No model results found. Run experiments/run_all_models.py first.")
        sys.exit(1)

    print(f"\nLoaded results for {n_loaded} models.")

    print_summary_table(family_data)
    save_summary_json(family_data)

    # Main comparison figure
    main_fig_path = FIGURES_DIR / "cross_family_comparison.png"
    plot_relative_depth_by_family(family_data, save_path=main_fig_path)

    # Per-model heatmaps
    print("\nGenerating per-model heatmaps...")
    plot_score_heatmaps(family_data, save_dir=FIGURES_DIR)

    print("\nDone.")
    print(
        "\nQuick interpretation notes (as of last run on A100, 2024-01):\n"
        "- GPT-2 family: rel_depth clusters tightly around 0.28-0.33. Very consistent.\n"
        "  Makes sense, all GPT-2 variants are same architecture just wider/deeper.\n"
        "- Pythia family: similar range (0.29-0.38) but slightly more spread, and\n"
        "  Pythia-160M shows no clear induction heads above threshold. Small model\n"
        "  effect, or the deduplication in the Pile affecting circuit formation?\n"
        "  TODO: check Pythia-70M to see if this is a scale cliff or gradual.\n"
        "- Llama-3.2: notably deeper (0.44-0.51). GQA + RoPE seem to push the\n"
        "  induction machinery later in the network. Also fewer heads overall\n"
        "  per layer (GQA), so the count is less meaningful as a comparison.\n"
        "  This is the most interesting finding - RoPE might interact with the\n"
        "  prefix mechanism differently than learned positional embeddings.\n"
        "  See the writeup section on positional encoding hypotheses."
    )


if __name__ == "__main__":
    main()