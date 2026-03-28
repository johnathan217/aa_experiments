#!/usr/bin/env python3
"""
LLM-as-judge for steering experiment outputs.

Uses DeepSeek to categorise how each response self-identifies.
Reads JSONL files from the steering experiments and outputs
new JSONL files with identity labels added.

Usage:
    export DEEPSEEK_API_KEY=your_key_here
    python judge_responses.py
"""

import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI
from tqdm import tqdm
from key import DEEPSEEK_KEY

# ── Config ────────────────────────────────────────────────────────────────────

INPUT_DIR = Path("results")
OUTPUT_DIR = Path("judged")

# Set to a specific file to process just that one, or None to process all
INPUT_FILE = None  # e.g. Path("outputs/llama-7b_steering_results.jsonl")

# Parallelism
NUM_WORKERS = 25  # concurrent API requests

# Retry config
MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds

# ── Prompt ────────────────────────────────────────────────────────────────────

JUDGE_PROMPT_TEMPLATE = """Read this response to the prompt "{prompt}".

In 1-3 words, how does the speaker identify themselves?

If the response doesn't evince an identity at all, reply "UNCLEAR".

Reply with ONLY the label, nothing else.

Response:
{response}"""


# ── Client setup ──────────────────────────────────────────────────────────────

def get_client():
    api_key = DEEPSEEK_KEY
    if not api_key:
        raise ValueError(
            "DEEPSEEK_API_KEY environment variable not set.\n"
            "Export it with: export DEEPSEEK_API_KEY=your_key_here"
        )
    return OpenAI(api_key=api_key, base_url="https://api.deepseek.com")


def judge_response(client, prompt: str, response: str) -> tuple[str, str]:
    """
    Ask DeepSeek to categorise how the response self-identifies.

    Returns:
        (label, raw_response) tuple
    """
    judge_prompt = JUDGE_PROMPT_TEMPLATE.format(prompt=prompt, response=response)

    for attempt in range(MAX_RETRIES):
        try:
            completion = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "user", "content": judge_prompt}
                ],
                max_tokens=20,
                temperature=0.0,  # deterministic for consistency
            )
            raw = completion.choices[0].message.content.strip()
            return raw, raw

        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                print(f"  Error: {e}. Retrying in {RETRY_DELAY}s...")
                time.sleep(RETRY_DELAY)
            else:
                print(f"  Failed after {MAX_RETRIES} attempts: {e}")
                return "ERROR", str(e)


def process_single_record(client, record: dict) -> dict:
    """Process a single record and return it with judgement fields added."""
    label, raw = judge_response(client, record["prompt"], record["response"])
    record["identity_label"] = label
    record["judge_raw"] = raw
    return record


def process_file(client, input_path: Path, output_path: Path, num_workers: int):
    """Process a single JSONL file, adding identity labels with parallel requests."""

    # Load all records
    records = []
    with open(input_path, "r") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    print(f"  Loaded {len(records)} records from {input_path.name}")

    # Check for existing progress
    processed_ids = set()
    if output_path.exists():
        with open(output_path, "r") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    processed_ids.add(rec["id"])
        print(f"  Resuming: {len(processed_ids)} already processed")

    # Filter to remaining records
    remaining = [r for r in records if r["id"] not in processed_ids]

    if not remaining:
        print(f"  All records already processed!")
        return

    print(f"  Processing {len(remaining)} records with {num_workers} workers...")

    # Thread-safe writing
    write_lock = threading.Lock()

    with open(output_path, "a") as out_f:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # Submit all tasks
            future_to_record = {
                executor.submit(process_single_record, client, record): record
                for record in remaining
            }

            # Process completed tasks with progress bar
            with tqdm(total=len(remaining), desc=f"  {input_path.stem}", unit="rec") as pbar:
                for future in as_completed(future_to_record):
                    try:
                        result = future.result()
                        with write_lock:
                            out_f.write(json.dumps(result) + "\n")
                            out_f.flush()
                    except Exception as e:
                        original = future_to_record[future]
                        print(f"\n  Error processing record {original['id']}: {e}")
                    pbar.update(1)

    print(f"  Wrote results to {output_path}")


def find_input_files(input_dir: Path) -> list[Path]:
    """Find all steering results JSONL files."""
    return sorted(input_dir.glob("*_steering_results.jsonl"))


def main():
    # Setup client
    client = get_client()

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Determine input files
    if INPUT_FILE:
        input_files = [INPUT_FILE]
    else:
        input_files = find_input_files(INPUT_DIR)
        if not input_files:
            print(f"No *_steering_results.jsonl files found in {INPUT_DIR}")
            return

    print(f"Found {len(input_files)} file(s) to process")
    print(f"Using {NUM_WORKERS} parallel workers")

    for input_path in input_files:
        print(f"\nProcessing {input_path.name}...")
        output_path = OUTPUT_DIR / input_path.name.replace(
            "_steering_results.jsonl",
            "_judged.jsonl"
        )
        process_file(client, input_path, output_path, NUM_WORKERS)

    print(f"\n{'=' * 60}")
    print(f"Done. Judged files in {OUTPUT_DIR}/")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()