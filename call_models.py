#!/usr/bin/env python3
"""
Run steering experiments across models, templates, PCs, and coefficients.

Usage:
    cd steering_experiments
    python call_models.py

Prerequisites:
    Run these scripts first to generate required files in SETUP_DATA_DIR:

    1. python setup.py
       Creates: {model_short}_layer_norms.pt
       Contains post-MLP residual stream norms for scaling steering coefficients.

    2. python prepare_pca.py
       Creates: {model_short}_pca.pt
       Contains PCA results (pc_dirs, var_exp, chosen_layer, n_layers, n_roles).

    Where {model_short} is the lowercased final segment of the HuggingFace model
    path (e.g., 'meta-llama/Llama-3-8B' -> 'llama-3-8b').

Reads configuration from experiment_config.py and outputs results to JSONL files
in OUTPUT_DIR, one per model: {model_short}_steering_results.jsonl
"""

import json
import uuid
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from assistant_axis import ActivationSteering
import experiment_config as cfg

# ── Config ────────────────────────────────────────────────────────────────────
MODELS = cfg.MODELS
PROMPTS = cfg.PROMPTS
TEMPLATES = cfg.TEMPLATES
PCS = cfg.PCS
COEFFICIENTS = cfg.COEFFICIENTS

SYSTEM_PROMPT = getattr(cfg, "SYSTEM_PROMPT", None)
MAX_NEW_TOKENS = getattr(cfg, "MAX_NEW_TOKENS", 512)
TEMPERATURE = getattr(cfg, "TEMPERATURE", 0.7)
DO_SAMPLE = not getattr(cfg, "GREEDY", False)
N_SAMPLES = getattr(cfg, "N_SAMPLES", 1)

SETUP_DATA_DIR = Path(getattr(cfg, "SETUP_DATA_DIR", "setup_data"))
OUTPUT_DIR = Path(getattr(cfg, "OUTPUT_DIR", "outputs"))


def model_short_name(model_dict: dict) -> str:
    """Extract short name from model path: 'meta-llama/Llama-3-8B' -> 'llama-3-8b'"""
    return model_dict["model"].split("/")[-1].lower()


def norms_path(model_dict: dict) -> Path:
    """Return the path where norms for this model are saved."""
    return SETUP_DATA_DIR / f"{model_short_name(model_dict)}_layer_norms.pt"


def pca_path(model_dict: dict) -> Path:
    """Return the path where PCA results for this model are saved."""
    return SETUP_DATA_DIR / f"{model_short_name(model_dict)}_pca.pt"


def output_path(model_dict: dict) -> Path:
    """Return the output JSONL path for this model."""
    return OUTPUT_DIR / f"{model_short_name(model_dict)}_steering_results.jsonl"


def load_model(model_name: str, device: str = "cuda"):
    """Load model and tokenizer from HuggingFace."""
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map=device
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def load_pca(model_dict: dict):
    """
    Load precomputed PCA results.

    Returns:
        pc_dirs: (n_pcs, hidden_dim) array of normalized PC directions
        chosen_layer: the layer index used
        var_exp: variance explained by each PC
    """
    path = pca_path(model_dict)
    if not path.exists():
        raise FileNotFoundError(
            f"PCA file not found: {path}\n"
            f"Run prepare_pca.py first to compute PCA on role vectors."
        )

    data = torch.load(path, map_location='cpu', weights_only=False)

    print(f"  Loaded PCA: layer {data['chosen_layer']}/{data['n_layers']}, {data['n_roles']} roles")
    print(f"  PC1 explains {data['var_exp'][0]*100:.1f}% variance")

    return (
        data["pc_dirs"].numpy(),
        data["chosen_layer"],
        data["var_exp"].numpy()
    )


def load_residual_norms(model_dict: dict, chosen_layer: int) -> float:
    """
    Load post-MLP residual stream norm at the chosen layer.
    Raises FileNotFoundError if norms haven't been computed.
    """
    path = norms_path(model_dict)
    if not path.exists():
        raise FileNotFoundError(
            f"Norms file not found: {path}\n"
            f"Run setup.py first to compute residual stream norms."
        )

    data = torch.load(path, map_location='cpu', weights_only=False)
    norm = float(data["mean_norms"][chosen_layer])
    print(f"  Residual norm at layer {chosen_layer}: {norm:.2f}")
    return norm


def get_native_template(model_dict: dict):
    """Load chat template from native_template model if specified."""
    if model_dict.get("native_template"):
        template_tokenizer = AutoTokenizer.from_pretrained(model_dict["native_template"])
        return template_tokenizer.chat_template
    return None


def build_prompt(tokenizer, template: str, prompt_text: str, native_chat_template=None) -> str:
    """Build the full prompt string based on template type."""
    if template == "none":
        return prompt_text

    messages = []
    if SYSTEM_PROMPT:
        messages.append({"role": "system", "content": SYSTEM_PROMPT})
    messages.append({"role": "user", "content": prompt_text})

    if template == "chatml":
        chat_template = cfg.CHATML_TEMPLATE
    elif template == "native":
        chat_template = native_chat_template
    else:
        raise ValueError(f"Unknown template: {template}")

    return tokenizer.apply_chat_template(
        messages,
        chat_template=chat_template,
        tokenize=False,
        add_generation_prompt=True
    )


def generate(
    model,
    tokenizer,
    prompt: str,
    pc_vector=None,
    coeff: float = 0.0,
    layer_idx: int = None,
    norm_scale: float = 1.0
) -> str:
    """Generate a response, optionally with activation steering."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    gen_kwargs = dict(
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE if DO_SAMPLE else None,
        do_sample=DO_SAMPLE,
        pad_token_id=tokenizer.eos_token_id
    )

    if pc_vector is not None:
        # Scale coefficient by residual stream norm
        scaled_coeff = coeff * norm_scale
        steering_vector = torch.tensor(pc_vector, dtype=torch.float16)

        with ActivationSteering(
            model,
            steering_vectors=[steering_vector],
            coefficients=[scaled_coeff],
            layer_indices=[layer_idx],
            intervention_type="addition",
            positions="all"
        ):
            with torch.no_grad():
                outputs = model.generate(inputs.input_ids, **gen_kwargs)
    else:
        with torch.no_grad():
            outputs = model.generate(inputs.input_ids, **gen_kwargs)

    generated_tokens = outputs[0, inputs.input_ids.shape[1]:]
    return tokenizer.decode(generated_tokens, skip_special_tokens=True)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for model_idx, model_dict in enumerate(MODELS, 1):
        model_name = model_dict["model"]
        out_path = output_path(model_dict)

        print(f"\n{'=' * 60}")
        print(f"  [{model_idx}/{len(MODELS)}]  {model_name}")
        print(f"{'=' * 60}")

        # Load precomputed PCA (errors if missing)
        pc_dirs, chosen_layer, var_exp = load_pca(model_dict)

        # Load residual norms for scaling (errors if missing)
        norm_scale = load_residual_norms(model_dict, chosen_layer)

        # Load model
        model, tokenizer = load_model(model_name)
        native_chat_template = get_native_template(model_dict)

        # Write results incrementally per model
        with open(out_path, "w") as fh:
            for template in TEMPLATES:
                prompts = PROMPTS["completion"] if template == "none" else PROMPTS["response"]

                for prompt_text in prompts:
                    full_prompt = build_prompt(tokenizer, template, prompt_text, native_chat_template)

                    for pc in PCS:
                        # baseline (pc=None) or steered
                        coeff_list = [0.0] if pc is None else COEFFICIENTS

                        # Get the PC vector if steering (1-indexed in config)
                        pc_vector = pc_dirs[pc - 1] if pc is not None else None

                        for coeff in coeff_list:
                            for sample_idx in range(N_SAMPLES):
                                response = generate(
                                    model, tokenizer, full_prompt,
                                    pc_vector=pc_vector,
                                    coeff=coeff,
                                    layer_idx=chosen_layer,
                                    norm_scale=norm_scale
                                )

                                record = {
                                    "id": str(uuid.uuid4()),
                                    "model": model_name,
                                    "template": template,
                                    "prompt": prompt_text,
                                    "pc": pc,
                                    "coefficient": coeff if pc is not None else None,
                                    "layer": chosen_layer,
                                    "sample_idx": sample_idx,
                                    "response": response
                                }

                                fh.write(json.dumps(record) + "\n")
                                fh.flush()

                                print(f"  [{template}] pc={pc} coeff={coeff} sample={sample_idx}")

        print(f"  Wrote results to {out_path}")

        # Free memory before next model
        del model
        torch.cuda.empty_cache()

    print(f"\n{'=' * 60}")
    print(f"  All done. Results in {OUTPUT_DIR}/")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()