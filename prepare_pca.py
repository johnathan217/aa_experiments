#!/usr/bin/env python3
"""
Compute PCA on role vectors for each model and save the results.

Usage:
    cd steering_experiments
    python prepare_pca.py

Output:
    {SETUP_DATA_DIR}/{model_short}_pca.pt
    Keys: pc_dirs, var_exp, chosen_layer, n_layers, n_roles
"""

from pathlib import Path

import torch
import numpy as np

from assistant_axis import compute_pca, MeanScaler
import experiment_config as cfg

# ── Config ────────────────────────────────────────────────────────────────────
MODELS = cfg.MODELS
SETUP_DATA_DIR = Path(getattr(cfg, "SETUP_DATA_DIR", "setup_data"))
LAYER_FRAC = getattr(cfg, "LAYER_FRAC", 0.5)
N_PCS = getattr(cfg, "N_PCS", 10)
FORCE = getattr(cfg, "FORCE_RECOMPUTE_PCA", False)


def model_short_name(model_dict: dict) -> str:
    """Extract short name from model path: 'meta-llama/Llama-3-8B' -> 'llama-3-8b'"""
    return model_dict["model"].split("/")[-1].lower()


def pca_path(model_dict: dict) -> Path:
    """Return the path where PCA results for this model should be saved."""
    return SETUP_DATA_DIR / f"{model_short_name(model_dict)}_pca.pt"


def compute_and_save_pca(model_dict: dict) -> Path:
    """
    Load role vectors, compute PCA, canonicalize, and save results.

    Returns:
        Path to the saved .pt file.
    """
    vectors_dir = Path(model_dict["vectors_dir"])
    save_path = pca_path(model_dict)

    # Load all role vectors
    roles = {}
    default = None

    for p in sorted(vectors_dir.glob('*.pt')):
        data = torch.load(p, map_location='cpu', weights_only=False)
        vec = data['vector'] if isinstance(data, dict) else data

        if p.stem == 'default':
            default = vec
        else:
            roles[p.stem] = vec

    if not roles:
        raise ValueError(f"No role vectors found in {vectors_dir}")

    # Determine layer count and chosen layer
    first_vec = next(iter(roles.values()))
    n_layers = first_vec.shape[0]
    chosen_layer = int(n_layers * LAYER_FRAC)

    print(f"  Vectors dir: {vectors_dir}")
    print(f"  Roles found: {len(roles)}")
    print(f"  Layer: {chosen_layer}/{n_layers} (frac={LAYER_FRAC})")

    # Extract vectors at chosen layer
    role_names = sorted(roles.keys())
    role_matrix = np.stack([
        roles[r][chosen_layer].float().numpy()
        for r in role_names
    ])
    default_vec = default[chosen_layer].float().numpy() if default is not None else None

    # Compute PCA
    scaler = MeanScaler()
    _, var_exp, _, pca, scaler = compute_pca(role_matrix, layer=None, scaler=scaler, verbose=False)

    # Extract and canonicalize PCs
    pcs = pca.components_[:N_PCS].copy()

    # Canonicalize PC1 so 'consultant' (or default) projects positively
    canon_vec = roles.get('consultant')
    if canon_vec is not None:
        canon_vec = canon_vec[chosen_layer].float().numpy()
    else:
        canon_vec = default_vec

    if canon_vec is not None:
        dsc = scaler.transform(canon_vec.reshape(1, -1)).squeeze()
        dnrm = dsc / np.linalg.norm(dsc)
        if np.dot(dnrm, pcs[0] / np.linalg.norm(pcs[0])) < 0:
            pcs[0] = -pcs[0]

    # Normalize
    pcs = pcs / np.linalg.norm(pcs, axis=1, keepdims=True)

    print(f"  PC1 explains {var_exp[0]*100:.1f}% variance")

    # Save
    SETUP_DATA_DIR.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "pc_dirs": torch.tensor(pcs, dtype=torch.float32),
            "var_exp": torch.tensor(var_exp[:N_PCS], dtype=torch.float32),
            "chosen_layer": chosen_layer,
            "n_layers": n_layers,
            "n_roles": len(roles),
            "role_names": role_names,
        },
        save_path,
    )

    print(f"  Saved to: {save_path}")

    return save_path


def main():
    """Compute PCA for all models in experiment_config.MODELS."""
    for i, model_dict in enumerate(MODELS, 1):
        model_name = model_dict["model"]
        save_path = pca_path(model_dict)

        print(f"\n{'#' * 60}")
        print(f"  [{i}/{len(MODELS)}]  {model_name}")
        print(f"{'#' * 60}")

        if save_path.exists() and not FORCE:
            print(f"\nPCA already found at {save_path} — skipping.")
            print(f"  (set FORCE_RECOMPUTE_PCA = True in experiment_config to recompute)")
            continue

        compute_and_save_pca(model_dict)

    print(f"\n{'#' * 60}")
    print(f"  All done. PCA results in {SETUP_DATA_DIR}/")
    print(f"{'#' * 60}\n")


if __name__ == "__main__":
    main()