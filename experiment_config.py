# Experiment configuration for sample.py
# Edit this file to define the prompts, templates, PCs, and steering strengths.

# ── Paths (relative to script location) ───────────────────────────────────────
SETUP_DATA_DIR = "setup_data"
OUTPUT_DIR = "outputs"

# ── Prompts ───────────────────────────────────────────────────────────────────
# "completion" prompts are used when template="none" (raw completion mode).
# "response" prompts are used when template="chatml" or "native" (chat mode).

PROMPTS = {
    "completion": [
        "I am ",
        "My role is ",
        "It is I, ",
        "Me? I am ",
        "Who am I? I am ",
        "People would say I am ",
        "My identity is ",
        "My purpose is ",
        "My role is "
    ],
    "response": [
        "Who are you?",
        "who are you?",
        "Who are you",
        "What is your identity?",
        "what is your identity",
        "Who am I communicating with?",
        "What are you?",
        "what are you?",
        "Tell me who you are",
        "Tell me what you are",
        "Tell me about yourself",
        "So who are you?",
        "tell me who you are?",
        "What is your role?",
        "what is your role",
    ]
}

# ── Templates ─────────────────────────────────────────────────────────────────
# "none"   – raw completion, no chat formatting
# "chatml" – explicit <|im_start|> / <|im_end|> wrapping
# "native" – model's own chat template

TEMPLATES = ["native", "chatml", "none"]

CHATML_TEMPLATE = """{% for message in messages %}
<|im_start|>{{ message['role'] }}
{{ message['content'] }}<|im_end|>
{% endfor %}
<|im_start|>assistant
"""

# ── Principal components ──────────────────────────────────────────────────────
# None  → no steering (baseline)
# 1, 2, … → steer with the N-th PC from the PCA file (1-indexed)

PCS = [None, 1]

# ── Steering coefficients ─────────────────────────────────────────────────────
# In units of the average post-MLP residual stream norm at the target layer.
# Ignored when pc=None.

COEFFICIENTS = [-2.0, -1.5, -1.0, -0.75, -0.5, -0.3, -0.2, -0.1, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0]

# ── Generation settings ───────────────────────────────────────────────────────

MAX_NEW_TOKENS = 512
TEMPERATURE = 0.7
GREEDY = False          # set True to override TEMPERATURE with greedy decoding
N_SAMPLES = 30          # independent samples per (prompt, template, pc, coeff) combo

# ── Layer selection ───────────────────────────────────────────────────────────
LAYER_FRAC = 0.5        # relative depth used for all models (0 = input, 1 = final)
N_PCS = 10              # number of PCs to compute

# ── Optional system prompt ────────────────────────────────────────────────────
# Set to None for no system prompt.

SYSTEM_PROMPT = None

# ── Setup / Norm computation settings ─────────────────────────────────────────
NORM_N_SAMPLES = 1000
NORM_MAX_LENGTH = 512
NORM_DATASET = "lmsys/lmsys-chat-1m"
FORCE_RECOMPUTE_NORMS = False

# ── Models ────────────────────────────────────────────────────────────────────
# native_template: HuggingFace model name whose chat template to use for
#                  template="native". Useful for base models that don't have
#                  their own chat template. Omit (or set None) to use the
#                  model's own tokenizer template.

MODELS = [
    {
        "model":           "dfurman/LLaMA-7B",
        "vectors_dir":     "outputs/llama-7b/vectors",
        "native_template": "PKU-Alignment/alpaca-7b-reproduced",
    },
    {
        "model":       "PKU-Alignment/alpaca-7b-reproduced",
        "vectors_dir": "outputs/alpaca-7b/vectors",
    },
    {
        "model":           "meta-llama/Llama-2-7b-hf",
        "vectors_dir":     "outputs/llama2-7b-hf/vectors",
        "native_template": "meta-llama/Llama-2-7b-chat-hf",
    },
    {
        "model":       "meta-llama/Llama-2-7b-chat-hf",
        "vectors_dir": "outputs/llama2-7b-chat-hf/vectors",
    },
    {
        "model":           "meta-llama/Meta-Llama-3-8B",
        "vectors_dir":     "outputs/llama3-8b/vectors",
        "native_template": "meta-llama/Meta-Llama-3-8B-Instruct",
    },
    {
        "model":       "meta-llama/Meta-Llama-3-8B-Instruct",
        "vectors_dir": "outputs/llama3-8b-instruct/vectors",
    },
]


# ── Count ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    n_baseline = sum(1 for pc in PCS if pc is None)
    n_steered = sum(1 for pc in PCS if pc is not None)

    total = 0
    for template in TEMPLATES:
        n_prompts = len(PROMPTS["completion"] if template == "none" else PROMPTS["response"])
        total += n_prompts * N_SAMPLES * (n_baseline + n_steered * len(COEFFICIENTS))

    print(f"Templates    : {TEMPLATES}")
    print(f"PCs          : {PCS}  ->  {n_baseline} baseline, {n_steered} steered")
    print(f"Coefficients : {COEFFICIENTS}")
    print(f"Samples/combo: {N_SAMPLES}")
    print()
    for template in TEMPLATES:
        n_prompts = len(PROMPTS["completion"] if template == "none" else PROMPTS["response"])
        t_total = n_prompts * N_SAMPLES * (n_baseline + n_steered * len(COEFFICIENTS))
        print(f"  {template:8s}  {n_prompts} prompts  ->  {t_total} calls")
    print()
    print(f"Total LLM calls: {total}")