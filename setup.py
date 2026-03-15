#!/usr/bin/env python3
"""
Compute average post-MLP residual stream norms per layer on LMSYS-CHAT-1M.

These norms are used to scale steering coefficients in call_models.py, so that
a coefficient of 1.0 corresponds to adding one "average residual norm" worth of
displacement along the steering direction at the target layer.

Usage:
    cd steering_experiments
    python setup.py

Output:
    {SETUP_DATA_DIR}/{model_short}_layer_norms.pt
    Keys: model_name, n_layers, n_samples, mean_norms (tensor of shape [n_layers])
"""

import sys
from pathlib import Path

import torch
from tqdm import tqdm

from assistant_axis import get_config
from assistant_axis.internals import ProbingModel
import experiment_config as cfg

# ── Config ────────────────────────────────────────────────────────────────────
MODELS = cfg.MODELS
SETUP_DATA_DIR = Path(getattr(cfg, "SETUP_DATA_DIR", "setup_data"))
N_SAMPLES = getattr(cfg, "NORM_N_SAMPLES", 1000)
MAX_LENGTH = getattr(cfg, "NORM_MAX_LENGTH", 512)
DATASET_PATH = getattr(cfg, "NORM_DATASET", "lmsys/lmsys-chat-1m")
FORCE = getattr(cfg, "FORCE_RECOMPUTE_NORMS", False)


def model_short_name(model_dict: dict) -> str:
    """Extract short name from model path: 'meta-llama/Llama-3-8B' -> 'llama-3-8b'"""
    return model_dict["model"].split("/")[-1].lower()


def norms_path(model_dict: dict) -> Path:
    """Return the path where norms for this model should be saved."""
    return SETUP_DATA_DIR / f"{model_short_name(model_dict)}_layer_norms.pt"


def _load_lmsys_samples(n_samples: int, max_length: int) -> list[str]:
    """Stream text samples from LMSYS-CHAT-1M (first user turn per conversation)."""
    from datasets import load_dataset

    dataset = load_dataset(DATASET_PATH, split="train", streaming=True)
    texts = []
    for item in dataset:
        conv = item.get("conversation", [])
        for turn in conv:
            if turn.get("role") == "user":
                content = turn.get("content", "").strip()
                if len(content) >= 30:
                    texts.append(content[: max_length * 5])
                    break
        if len(texts) >= n_samples:
            break
    return texts


def compute_layer_norms(model_dict: dict) -> Path:
    """
    Measure average post-MLP residual stream norms per layer.

    Runs a forward pass for each sampled text, hooks the output of each
    transformer block (= post-MLP residual stream), computes the mean L2 norm
    across token positions, and averages over all samples.

    Returns:
        Path to the saved .pt file.
    """
    model_name = model_dict["model"]
    save_path = norms_path(model_dict)

    config = get_config(model_name)
    n_layers = config["total_layers"]
    target_layer = config["target_layer"]

    print(f"Model         : {model_name}")
    print(f"Total layers  : {n_layers}")
    print(f"Target layer  : {target_layer}")

    # ── Load model ────────────────────────────────────────────────────────────
    print("\nLoading model...")
    pm = ProbingModel(model_name)
    layers = pm.get_layers()
    if len(layers) != n_layers:
        print(f"  Warning: config says {n_layers} layers but found {len(layers)}; using {len(layers)}")
        n_layers = len(layers)

    # ── Load dataset ──────────────────────────────────────────────────────────
    print(f"\nLoading {N_SAMPLES} samples from '{DATASET_PATH}'...")
    texts = _load_lmsys_samples(N_SAMPLES, MAX_LENGTH)
    print(f"  Loaded {len(texts)} samples")

    # ── Accumulate norms ──────────────────────────────────────────────────────
    norm_sums = torch.zeros(n_layers, dtype=torch.float64)
    norm_counts = torch.zeros(n_layers, dtype=torch.float64)

    print("\nComputing per-layer norms...")
    for text in tqdm(texts):
        tokens = pm.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_LENGTH,
            add_special_tokens=True,
        )
        input_ids = tokens["input_ids"].to(pm.device)

        if input_ids.shape[1] < 2:
            continue

        captured: dict[int, float] = {}
        handles = []

        def _make_hook(idx):
            def _hook(module, inp, out):
                tensor = out[0] if isinstance(out, tuple) else out
                captured[idx] = tensor[0].float().norm(dim=-1).mean().item()
            return _hook

        for i in range(n_layers):
            handles.append(layers[i].register_forward_hook(_make_hook(i)))

        try:
            with torch.inference_mode():
                pm.model(input_ids)
        finally:
            for h in handles:
                h.remove()

        for i, val in captured.items():
            norm_sums[i] += val
            norm_counts[i] += 1

    mean_norms = (norm_sums / norm_counts.clamp(min=1)).float()

    # ── Save ──────────────────────────────────────────────────────────────────
    SETUP_DATA_DIR.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model_name": model_name,
            "n_layers": n_layers,
            "n_samples": int(norm_counts[0].item()),
            "mean_norms": mean_norms,
        },
        save_path,
    )

    print(f"\nSaved to: {save_path}")
    print(f"Norm at target layer {target_layer}: {mean_norms[target_layer]:.2f}")

    # Free memory
    del pm
    torch.cuda.empty_cache()

    return save_path


def main():
    """Compute norms for all models in experiment_config.MODELS."""
    for i, model_dict in enumerate(MODELS, 1):
        model_name = model_dict["model"]
        save_path = norms_path(model_dict)

        print(f"\n{'#' * 60}")
        print(f"  [{i}/{len(MODELS)}]  {model_name}")
        print(f"{'#' * 60}")

        if save_path.exists() and not FORCE:
            print(f"\nNorms already found at {save_path} — skipping.")
            print(f"  (set FORCE_RECOMPUTE_NORMS = True in experiment_config to recompute)")
            continue

        compute_layer_norms(model_dict)

    print(f"\n{'#' * 60}")
    print(f"  All done. Norms in {SETUP_DATA_DIR}/")
    print(f"{'#' * 60}\n")


if __name__ == "__main__":
    main()