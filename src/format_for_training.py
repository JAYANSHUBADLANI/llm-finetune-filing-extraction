"""
Converts data_prep.py's examples into the prompt/completion JSONL format
mlx_lm's LoRA trainer expects. Splits dev examples into train and a held
out valid slice by filing, not by row, so the same filing's nine fields
never appear split across train and valid, which would leak that filing's
statement text into validation.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
SEED = 7
VALID_FRACTION = 0.15


def to_prompt_completion(example: dict) -> dict:
    prompt = (
        f"Financial statement tables for {example['company']}:\n\n"
        f"{example['statement_text']}\n\n"
        f"{example['question']}"
    )
    completion = str(example["true_value"])
    return {"prompt": prompt, "completion": completion}


def main() -> None:
    with open(DATA_DIR / "dev_examples.json") as f:
        dev = json.load(f)
    with open(DATA_DIR / "test_examples.json") as f:
        test = json.load(f)

    filings = sorted(set(e["doc_id"] for e in dev))
    rng = random.Random(SEED)
    rng.shuffle(filings)
    n_valid = max(1, int(len(filings) * VALID_FRACTION))
    valid_filings = set(filings[:n_valid])

    train_examples = [e for e in dev if e["doc_id"] not in valid_filings]
    valid_examples = [e for e in dev if e["doc_id"] in valid_filings]

    print(f"train: {len(train_examples)} examples, {len(filings) - n_valid} filings")
    print(f"valid: {len(valid_examples)} examples, {n_valid} filings")
    print(f"test: {len(test)} examples, {len(set(e['doc_id'] for e in test))} filings")

    mlx_data_dir = DATA_DIR / "mlx_format"
    mlx_data_dir.mkdir(exist_ok=True)

    for name, examples in [("train", train_examples), ("valid", valid_examples), ("test", test)]:
        path = mlx_data_dir / f"{name}.jsonl"
        with open(path, "w") as f:
            for e in examples:
                f.write(json.dumps(to_prompt_completion(e)) + "\n")
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
