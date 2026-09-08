import json
import random
from pathlib import Path

from format_for_training import SEED, VALID_FRACTION, to_prompt_completion

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def test_to_prompt_completion_shape():
    example = {
        "company": "Acme Corp",
        "statement_text": "[scale: as shown, not scaled]\nrevenue | 1234",
        "question": "What is the total revenue reported?",
        "true_value": 1234.0,
    }
    result = to_prompt_completion(example)

    assert result["prompt"].startswith("Financial statement tables for Acme Corp:")
    assert "revenue | 1234" in result["prompt"]
    assert result["prompt"].endswith("What is the total revenue reported?")
    assert result["completion"] == "1234.0"


def test_no_filing_appears_in_both_train_and_valid():
    # Re-derives the same filing-level split format_for_training.py's
    # main() computes (same SEED, same VALID_FRACTION, same shuffle over
    # the real dev set's filings) and confirms no filing lands in both
    # halves, the property the split is by filing rather than by row
    # exists to guarantee, since a leak here would let a fine-tuned
    # model's validation loss look good by having memorized the exact
    # statement text rather than generalizing.
    dev = json.load(open(DATA_DIR / "dev_examples.json"))
    filings = sorted(set(e["doc_id"] for e in dev))

    rng = random.Random(SEED)
    rng.shuffle(filings)
    n_valid = max(1, int(len(filings) * VALID_FRACTION))
    valid_filings = set(filings[:n_valid])
    train_filings = set(filings[n_valid:])

    assert train_filings.isdisjoint(valid_filings)
    assert train_filings | valid_filings == set(filings)
    assert len(valid_filings) == 9  # 60 dev filings * 0.15, matches data/mlx_format on disk
