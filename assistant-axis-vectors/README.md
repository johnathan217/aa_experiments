---
license: mit
task_categories:
- other
tags:
- persona-drift
- activation-steering
- mechanistic-interpretability
---

# The Assistant Axis: Situating and Stabilizing the Default Persona of Language Models

This repository contains pre-computed axes and persona vectors for Gemma 2 27B, Qwen 3 32B, and Llama 3.3 70B, as described in the paper [The Assistant Axis: Situating and Stabilizing the Default Persona of Language Models](https://huggingface.co/papers/2601.10387).

[**Paper**](https://huggingface.co/papers/2601.10387) | [**Code**](https://github.com/safety-research/assistant-axis) | [**Demo**](https://neuronpedia.org/assistant-axis)

The **Assistant Axis** is a direction in activation space that captures how "Assistant-like" a model's current persona is. It can be used to:
- **Monitor** persona drift in real-time by projecting activations onto the axis.
- **Steer** model behavior toward or away from the default Assistant persona.
- **Mitigate** persona-based jailbreaks through activation capping.

## Usage

You can load the pre-computed axes using the `huggingface_hub` library. To use the helper functions like `load_axis`, you should clone the [official repository](https://github.com/safety-research/assistant-axis).

```python
from huggingface_hub import hf_hub_download
from assistant_axis import load_model, load_axis

# Load model
model, tokenizer = load_model("google/gemma-2-27b-it")

# Download pre-computed axis
axis_path = hf_hub_download(
    repo_id="lu-christina/assistant-axis-vectors",
    filename="gemma-2-27b/assistant_axis.pt",
    repo_type="dataset"
)
axis = load_axis(axis_path)
```

## Citation

```bibtex
@misc{lu2026assistant,
      title={The Assistant Axis: Situating and Stabilizing the Default Persona of Language Models}, 
      author={Christina Lu and Jack Gallagher and Jonathan Michala and Kyle Fish and Jack Lindsey},
      year={2026},
      eprint={2601.10387},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2601.10387}, 
}
```