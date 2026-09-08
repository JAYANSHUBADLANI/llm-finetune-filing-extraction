"""
Builds a fixed set of few-shot demonstrations from the TRAIN split only,
for the baseline this project's README was missing: does putting a
handful of worked examples straight in the prompt get a 3B model most of
the way to what LoRA fine-tuning achieves, without training anything?
That is the obvious cheaper alternative to check before crediting LoRA
with an improvement.

The same 3 demonstrations are reused for every test example in a run,
not resampled per example, otherwise "the few-shot baseline" would
really be many different few-shot baselines, one per test example, and
a difference between two test examples could just be a difference in
which demonstrations they happened to get.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from format_for_training import SEED, VALID_FRACTION

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

DEMO_MAX_CHARS = 1000
DEMO_FIELDS = ["revenue", "eps_diluted", "total_liabilities"]


def train_filings() -> set[str]:
    """Recomputes format_for_training.py's own train/valid filing split.

    Duplicated rather than imported as a function because
    format_for_training.py does this split inline in main(), not as a
    reusable function; recomputing it here with the same SEED and
    VALID_FRACTION constants that module exports is exact, not
    approximate, and is covered by
    tests/test_format_for_training.py::test_no_filing_appears_in_both_train_and_valid.
    """
    dev = json.load(open(DATA_DIR / "dev_examples.json"))
    filings = sorted(set(e["doc_id"] for e in dev))
    rng = random.Random(SEED)
    rng.shuffle(filings)
    n_valid = max(1, int(len(filings) * VALID_FRACTION))
    return set(filings[n_valid:])


def select_demonstrations(k: int = 3) -> list[dict]:
    """Picks k train-split examples spanning distinct fields, deterministically."""
    dev = json.load(open(DATA_DIR / "dev_examples.json"))
    candidates = [e for e in dev if e["doc_id"] in train_filings()]
    candidates.sort(key=lambda e: (e["doc_id"], e["field"]))  # deterministic order

    chosen = []
    for field in DEMO_FIELDS[:k]:
        match = next((e for e in candidates if e["field"] == field), None)
        if match is not None:
            chosen.append(match)
    for e in candidates:
        if len(chosen) >= k:
            break
        if e not in chosen:
            chosen.append(e)
    return chosen[:k]


def build_messages(demonstrations: list[dict], real_question_prompt: str) -> list[dict]:
    messages = []
    for demo in demonstrations:
        statement_text = demo["statement_text"][:DEMO_MAX_CHARS]
        user_content = (
            f"Financial statement tables for {demo['company']}:\n\n"
            f"{statement_text}\n\n"
            f"{demo['question']}"
        )
        messages.append({"role": "user", "content": user_content})
        messages.append({"role": "assistant", "content": str(demo["true_value"])})
    messages.append({"role": "user", "content": real_question_prompt})
    return messages
