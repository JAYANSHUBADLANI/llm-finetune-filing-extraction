"""
Zero-shot diagnostic: does a much bigger model (already downloaded for
the sar-multiagent-drafting project, not fetched for this one) close any
of this task's dominant failure, wrong_row, a different line item read
off the table entirely? A 3B model's mistake there could be a genuine
model-capacity ceiling, or it could be an artifact of this project's own
small-model-oriented setup (short context budget, terse prompt). This
script answers only the first possibility, cheaply: no LoRA fine-tuning
of qwen2.5:14b is done or attempted here, it's a GGUF model served by
Ollama, not the MLX format `run_lora_training.py` fine-tunes, so this is
zero-shot only, a capability check, not a full third method to add
alongside zero-shot/few-shot/fine-tuned in README.md's main comparison.

Uses stdlib `urllib` rather than the `requests` package specifically to
avoid adding a new dependency to requirements.txt for one diagnostic
script.
"""

from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import evaluate_model  # reuses parse_numeric
from format_for_training import to_prompt_completion

SOURCE_PROJECT = Path(__file__).resolve().parents[2] / "filing-extraction-benchmark"
sys.path.insert(0, str(SOURCE_PROJECT / "src"))
from filingbench.scoring import values_agree  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen2.5:14b"


def ollama_generate(prompt: str, timeout: float = 60.0) -> str:
    payload = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {"num_predict": 32},
    }).encode()
    req = urllib.request.Request(
        OLLAMA_URL, data=payload, headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = json.loads(resp.read())
    return body["message"]["content"]


def run_eval(label: str, limit: int | None = None) -> list[dict]:
    with open(DATA_DIR / "test_examples.json") as f:
        meta = json.load(f)
    if limit:
        meta = meta[:limit]

    records = []
    for i, m in enumerate(meta):
        prompt = to_prompt_completion(m)["prompt"]
        t0 = time.time()
        try:
            raw_output = ollama_generate(prompt)
        except (urllib.error.URLError, TimeoutError) as e:
            raw_output = ""
            print(f"  request failed for {m['doc_id']}/{m['field']}: {e}")
        elapsed = time.time() - t0

        predicted = evaluate_model.parse_numeric(raw_output)
        correct = predicted is not None and values_agree(predicted, m["true_value"], m["unit"])
        records.append({
            "label": label,
            "doc_id": m["doc_id"],
            "company": m["company"],
            "field": m["field"],
            "true_value": m["true_value"],
            "unit": m["unit"],
            "raw_output": raw_output,
            "predicted": predicted,
            "correct": correct,
            "seconds": elapsed,
        })
        if (i + 1) % 20 == 0:
            n_correct_so_far = sum(r["correct"] for r in records)
            print(f"  {label}: {i + 1}/{len(meta)} done ({n_correct_so_far}/{i + 1} correct so far)")

    n_correct = sum(r["correct"] for r in records)
    print(f"{label}: {n_correct}/{len(records)} correct ({n_correct / len(records):.1%})")
    return records


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    RESULTS_DIR.mkdir(exist_ok=True)
    records = run_eval(args.label, args.limit)
    out_path = RESULTS_DIR / f"predictions_{args.label}.json"
    with open(out_path, "w") as f:
        json.dump(records, f, indent=2)
    print(f"written to {out_path}")
