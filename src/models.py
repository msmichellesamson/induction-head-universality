"""
Model loader for induction head universality experiments.

Maps model name strings → HookedTransformer instances with a consistent interface.
The main pain points this solves:
  1. TransformerLens uses different cfg keys for different model families
  2. Llama tokenizers need special handling (no default pad token, BOS weirdness)
  3. GPT-2 vs Pythia vs Llama have different layernorm placements that affect
     how we index activations

I'm keeping this thin on purpose - the goal is just consistent loading,
not abstracting away TransformerLens. We still want to use TL's native
hook API directly in experiments.

Hardware note: tested on a single A100 40GB. Pythia-2.8B is the biggest
model that fits comfortably with activations cached for a batch of 32.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import torch
import transformer_lens
from transformer_lens import HookedTransformer, HookedTransformerConfig

logger = logging.getLogger(__name__)

# Models I've actually tested. Others might work but I haven't verified
# the hook names line up with what induction_score.py expects.
SUPPORTED_MODELS = {
    # Pythia family - nice because EleutherAI trained the whole suite with
    # identical hyperparams except scale, so they're the cleanest comparison.
    "pythia-70m":   "EleutherAI/pythia-70m-deduped",
    "pythia-160m":  "EleutherAI/pythia-160m-deduped",
    "pythia-410m":  "EleutherAI/pythia-410m-deduped",
    "pythia-1.4b":  "EleutherAI/pythia-1.4b-deduped",
    "pythia-2.8b":  "EleutherAI/pythia-2.8b-deduped",

    # GPT-2 family - the OG and heavily studied, good baseline
    "gpt2-small":   "gpt2",
    "gpt2-medium":  "gpt2-medium",
    "gpt2-large":   "gpt2-large",

    # Llama - different architecture family (RoPE, SwiGLU, RMSNorm, GQA in 3.x)
    # Using the smallest available to keep experiments tractable
    "llama-3.2-1b": "meta-llama/Llama-3.2-1B",
    "llama-3.2-3b": "meta-llama/Llama-3.2-3B",

    # Mistral - another RoPE family, sliding window attention in larger versions
    # but 7B is too big for most of what I'm doing, so mostly using this for
    # architecture comparison rather than scale comparison
    "mistral-7b":   "mistralai/Mistral-7B-v0.1",
}

# Model families - used downstream to group results and check if architecture
# differences correlate with induction circuit differences
MODEL_FAMILIES = {
    "pythia":  ["pythia-70m", "pythia-160m", "pythia-410m", "pythia-1.4b", "pythia-2.8b"],
    "gpt2":    ["gpt2-small", "gpt2-medium", "gpt2-large"],
    "llama":   ["llama-3.2-1b", "llama-3.2-3b"],
    "mistral": ["mistral-7b"],
}

# Reverse map: model name → family. Computed once at import.
MODEL_TO_FAMILY = {
    model: family
    for family, models in MODEL_FAMILIES.items()
    for model in models
}


@dataclass
class ModelInfo:
    """
    Metadata that's useful to have alongside the model itself.
    Keeping this separate from the HookedTransformer so callers
    don't have to dig into cfg attributes.
    """
    name: str
    family: str
    hf_name: str
    n_layers: int
    n_heads: int
    d_model: int
    d_head: int
    # Whether the model uses rotary position embeddings (RoPE).
    # Matters because RoPE changes how positional info flows through
    # attention - could affect how induction heads form.
    uses_rope: bool
    # Pre-norm (GPT-2 style) vs post-norm. Llama/Mistral are pre-norm
    # (RMSNorm before attention), which affects residual stream dynamics.
    uses_pre_norm: bool
    # GQA = grouped query attention (Llama 3.x). Reduces KV heads.
    # Unsure yet whether this affects induction head structure - one thing to test.
    uses_gqa: bool
    n_kv_heads: Optional[int] = None  # Only set if uses_gqa
    # Hook name patterns for this model - these can differ across families
    # even when TL tries to normalize them
    attn_hook_pattern: str = "blocks.{layer}.attn.hook_pattern"
    mlp_hook_pattern: str = "blocks.{layer}.hook_mlp_out"


def get_model_family(model_name: str) -> str:
    if model_name not in MODEL_TO_FAMILY:
        raise ValueError(
            f"Unknown model: {model_name!r}. "
            f"Supported: {list(SUPPORTED_MODELS.keys())}"
        )
    return MODEL_TO_FAMILY[model_name]


def load_model(
    model_name: str,
    device: Optional[str] = None,
    dtype: torch.dtype = torch.float32,
    fold_ln: bool = True,
    center_writing_weights: bool = True,
    center_unembed: bool = True,
) -> tuple[HookedTransformer, ModelInfo]:
    """
    Load a HookedTransformer and return it with a ModelInfo struct.

    fold_ln / center_writing_weights / center_unembed are TransformerLens
    preprocessing options. I'm defaulting them to True because:
    - fold_ln: folds layernorm params into adjacent weight matrices, which
      makes the residual stream cleaner to analyze (no LN distortion)
    - center_writing_weights: makes residual stream have zero mean,
      simplifies analysis of what heads write
    - center_unembed: same idea at the output

    NOTE: if you're doing careful weight-based analysis rather than
    activation-based, turn these off - they change the actual weights.
    For induction score experiments we're looking at attention patterns,
    so the defaults should be fine.
    """
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(
            f"Unknown model {model_name!r}. "
            f"Known models: {sorted(SUPPORTED_MODELS.keys())}"
        )

    hf_name = SUPPORTED_MODELS[model_name]
    family = get_model_family(model_name)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Auto-selected device: {device}")

    logger.info(f"Loading {model_name} ({hf_name}) on {device} with dtype={dtype}")

    # Family-specific loading logic lives here rather than being scattered
    # through the experiments
    if family == "llama":
        model = _load_llama(hf_name, device, dtype, fold_ln, center_writing_weights, center_unembed)
    elif family == "mistral":
        model = _load_mistral(hf_name, device, dtype, fold_ln, center_writing_weights, center_unembed)
    else:
        # GPT-2 and Pythia load cleanly with default TL settings
        model = HookedTransformer.from_pretrained(
            hf_name,
            device=device,
            dtype=dtype,
            fold_ln=fold_ln,
            center_writing_weights=center_writing_weights,
            center_unembed=center_unembed,
        )

    model.eval()

    info = _build_model_info(model_name, family, hf_name, model)
    _log_model_summary(info)

    return model, info


def _load_llama(
    hf_name: str,
    device: str,
    dtype: torch.dtype,
    fold_ln: bool,
    center_writing_weights: bool,
    center_unembed: bool,
) -> HookedTransformer:
    """
    Llama needs special tokenizer handling:
    - No pad token by default. TL will complain. Setting pad = eos is the
      standard workaround but it does mean padding tokens look like EOS
      during training - shouldn't matter for our eval-only experiments.
    - BOS token (id=1) is prepended by default by the Llama tokenizer.
      TL's from_pretrained handles this but worth knowing.
    - Llama 3.x uses a different tokenizer (tiktoken-based) vs Llama 2
      (sentencepiece). TL should handle both but I've only tested 3.x.
    """
    from transformers import AutoTokenizer

    # Load tokenizer separately to fix pad token before TL sees it
    tokenizer = AutoTokenizer.from_pretrained(hf_name)
    if tokenizer.pad_token is None:
        logger.warning(
            "Llama tokenizer has no pad token - setting pad_token = eos_token. "
            "This is fine for our experiments but be aware if adapting this."
        )
        tokenizer.pad_token = tokenizer.eos_token

    model = HookedTransformer.from_pretrained(
        hf_name,
        tokenizer=tokenizer,
        device=device,
        dtype=dtype,
        fold_ln=fold_ln,
        center_writing_weights=center_writing_weights,
        center_unembed=center_unembed,
        # fold_value_biases is useful for Llama since it has no value biases
        # (unlike GPT-2) - TL handles this but being explicit
        fold_value_biases=True,
    )

    return model


def _load_mistral(
    hf_name: str,
    device: str,
    dtype: torch.dtype,
    fold_ln: bool,
    center_writing_weights: bool,
    center_unembed: bool,
) -> HookedTransformer:
    """
    Mistral is similar to Llama but with sliding window attention in some layers.
    TransformerLens might not support SWA fully - using Mistral-7B-v0.1 which
    has SWA but TL seems to handle it by just ignoring the window constraint
    (effectively treating it as full attention). This is a known approximation.

    TODO: check whether this matters for induction heads. Induction operates
    on relatively short distances so SWA probably doesn't change much, but
    worth verifying once I have baseline results from smaller models.
    """
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(hf_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = HookedTransformer.from_pretrained(
        hf_name,
        tokenizer=tokenizer,
        device=device,
        dtype=dtype,
        fold_ln=fold_ln,
        center_writing_weights=center_writing_weights,
        center_unembed=center_unembed,
        fold_value_biases=True,
    )

    return model


def _build_model_info(
    model_name: str,
    family: str,
    hf_name: str,
    model: HookedTransformer,
) -> ModelInfo:
    """
    Extract structural info from the loaded model's cfg.
    TransformerLens normalizes a lot of this but some things
    need to be inferred from what we know about the architecture.
    """
    cfg = model.cfg

    uses_rope = getattr(cfg, "positional_embedding_type", "standard") == "rotary"
    # NOTE: this is a bit fragile - TL's cfg attribute naming has shifted
    # across versions. Checking a few possibilities.
    uses_pre_norm = getattr(cfg, "normalization_type", "LN") in ("RMS", "LNPre")

    # GQA: Llama 3.x uses it. TL exposes n_key_value_heads when present.
    n_kv_heads = getattr(cfg, "n_key_value_heads", None)
    uses_gqa = n_kv_heads is not None and n_kv_heads != cfg.n_heads

    return ModelInfo(
        name=model_name,
        family=family,
        hf_name=hf_name,
        n_layers=cfg.n_layers,
        n_heads=cfg.n_heads,
        d_model=cfg.d_model,
        d_head=cfg.d_head,
        uses_rope=uses_rope,
        uses_pre_norm=uses_pre_norm,
        uses_gqa=uses_gqa,
        n_kv_heads=n_kv_heads,
    )


def _log_model_summary(info: ModelInfo) -> None:
    logger.info(
        f"Loaded {info.name} | "
        f"family={info.family} | "
        f"layers={info.n_layers} | "
        f"heads={info.n_heads} | "
        f"d_model={info.d_model} | "
        f"d_head={info.d_head} | "
        f"rope={info.uses_rope} | "
        f"pre_norm={info.uses_pre_norm} | "
        f"gqa={info.uses_gqa}"
        + (f" (kv_heads={info.n_kv_heads})" if info.uses_gqa else "")
    )


def get_hook_name(info: ModelInfo, hook_type: str, layer: int) -> str:
    """
    Resolve a hook name for a given model and hook type.

    TransformerLens mostly normalizes hook names across models, but there
    are edge cases - especially for GQA models where the K/V projections
    have different shapes. This centralizes the lookup so experiments
    don't have to worry about it.

    hook_type options:
        "attn_pattern"   - attention pattern (post-softmax), shape [batch, head, seq, seq]
        "attn_z"         - value-weighted sum, shape [batch, seq, head, d_head]
        "attn_out"       - attention layer output, shape [batch, seq, d_model]
        "mlp_out"        - MLP output, shape [batch, seq, d_model]
        "resid_pre"      - residual stream before attn, shape [batch, seq, d_model]
        "resid_mid"      - residual stream after attn, before MLP
        "resid_post"     - residual stream after MLP
        "q", "k", "v"    - Q/K/V projections
    """
    hook_map = {
        "attn_pattern": f"blocks.{layer}.attn.hook_pattern",
        "attn_z":       f"blocks.{layer}.attn.hook_z",
        "attn_out":     f"blocks.{layer}.hook_attn_out",
        "mlp_out":      f"blocks.{layer}.hook_mlp_out",
        "resid_pre":    f"blocks.{layer}.hook_resid_pre",
        "resid_mid":    f"blocks.{layer}.hook_resid_mid",
        "resid_post":   f"blocks.{layer}.hook_resid_post",
        "q":            f"blocks.{layer}.attn.hook_q",
        "k":            f"blocks.{layer}.attn.hook_k",
        "v":            f"blocks.{layer}.attn.hook_v",
    }

    if hook_type not in hook_map:
        raise ValueError(f"Unknown hook_type {hook_type!r}. Options: {list(hook_map.keys())}")

    return hook_map[hook_type]


def get_all_attn_pattern_hooks(info: ModelInfo) -> list[str]:
    """
    Return hook names for attention patterns across all layers.
    Convenience wrapper used by induction_score.py when it needs to
    cache patterns for every layer in one forward pass.
    """
    return [get_hook_name(info, "attn_pattern", layer) for layer in range(info.n_layers)]


def models_by_family(family: str) -> list[str]:
    """Return model names for a given family, sorted by size (roughly)."""
    if family not in MODEL_FAMILIES:
        raise ValueError(f"Unknown family {family!r}. Options: {list(MODEL_FAMILIES.keys())}")
    return MODEL_FAMILIES[family]


def all_model_names() -> list[str]:
    return list(SUPPORTED_MODELS.keys())


# Quick sanity check - run this if you want to verify loading works
# before running a full experiment. Won't download anything if models
# are already cached.
if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    test_models = ["gpt2-small", "pythia-70m"]
    if len(sys.argv) > 1:
        test_models = sys.argv[1:]

    for name in test_models:
        print(f"\n--- Testing {name} ---")
        try:
            model, info = load_model(name)
            # Quick shape check: run a dummy forward pass
            dummy_tokens = torch.randint(0, info.d_model, (1, 16))
            # Don't use d_model for vocab - TL exposes d_vocab
            dummy_tokens = torch.randint(0, model.cfg.d_vocab, (1, 16))
            with torch.no_grad():
                logits = model(dummy_tokens)
            print(f"  Forward pass OK. Output shape: {logits.shape}")

            # Check a hook exists
            hook = get_hook_name(info, "attn_pattern", layer=0)
            print(f"  Layer 0 attn pattern hook: {hook}")
            print(f"  All layers hook list length: {len(get_all_attn_pattern_hooks(info))}")
        except Exception as e:
            print(f"  FAILED: {e}")
            raise