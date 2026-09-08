"""
Generates an answer for every test example with a given model (optionally
with a LoRA adapter applied) and scores it with the source
filing-extraction-benchmark project's own values_agree function, the same
0.5 percent relative tolerance rules and the API language model were
scored under, so a number here is comparable to that project's numbers on
the scoring rule itself. What is NOT comparable is stated in README.md:
this project's input is the statements only text, not the full document
that project's 0.9321 test figure was measured against.
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

from mlx_lm import generate, load

import few_shot

SOURCE_PROJECT = Path(__file__).resolve().parents[2] / "filing-extraction-benchmark"
sys.path.insert(0, str(SOURCE_PROJECT / "src"))
from filingbench.scoring import values_agree  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
BASE_MODEL = "mlx-community/Llama-3.2-3B-Instruct-4bit"


def parse_numeric(text: str) -> float | None:
    """Extract the model's intended numeric answer from free text.

    Takes the last number in the text, not the first. Found necessary
    against the real test set, not assumed: several outputs show the
    model's arithmetic before its answer ("8,625 + ... = 75,496", answer
    75,496) or restate a year before the value ("2023: 30,425", answer
    30,425, not the year 2023 that a first-match regex would grab). The
    instruction to the model asks for only the final number, but a 3B
    model does not reliably comply, so parsing has to be robust to it
    rather than trust the instruction was followed. See PROGRESS.md.
    """
    text = text.strip().replace(",", "").replace("$", "")
    matches = re.findall(r"-?\d+\.?\d*", text)
    if not matches:
        return None
    try:
        return float(matches[-1])
    except ValueError:
        return None


def slice_range(seq: list, offset: int, limit: int | None) -> list:
    """A fixed contiguous [offset, offset+limit) slice, for chunked runs.

    Added because a long-lived evaluate_model.py process on this machine
    correlates with a wall-clock-timed stall (elapsed time far exceeding
    the process's own accumulated CPU time) that example count or model
    size don't predict, several short-lived processes, each covering a
    chunk via offset, complete reliably where one long process covering
    the same total examples does not. See PROGRESS.md. offset=0 with the
    previous limit-only behavior is unchanged, no end index needed when
    limit is falsy.
    """
    end = offset + limit if limit else None
    return seq[offset:end]


def run_eval(
    adapter_path: str | None, label: str, limit: int | None = None, model_path: str | None = None,
    few_shot_k: int = 0, offset: int = 0,
) -> list[dict]:
    # Loading the base model with adapter_path set, for repeated generation
    # calls, reproducibly consumed memory rapidly with no generation
    # progress on this machine, twice. Fusing the adapter once with
    # run_fuse.py and loading the fused checkpoint as a plain local model
    # here avoids that path entirely; model_path is how the fine-tuned
    # evaluation actually gets run, adapter_path is kept only because
    # run_fuse.py's own loading needs the same MPI patch this file does
    # not, so it is a separate script. See PROGRESS.md.
    if model_path:
        model, tokenizer = load(model_path)
    elif adapter_path:
        model, tokenizer = load(BASE_MODEL, adapter_path=adapter_path)
    else:
        model, tokenizer = load(BASE_MODEL)

    with open(DATA_DIR / "mlx_format" / "test.jsonl") as f:
        examples = [json.loads(line) for line in f]
    with open(DATA_DIR / "test_examples.json") as f:
        meta = json.load(f)
    assert len(examples) == len(meta)

    examples = slice_range(examples, offset, limit)
    meta = slice_range(meta, offset, limit)

    demonstrations = few_shot.select_demonstrations(few_shot_k) if few_shot_k else []
    if few_shot_k:
        print(f"  {label}: using {len(demonstrations)} few-shot demonstrations "
              f"({[d['field'] for d in demonstrations]})")

    records = []
    for i, (ex, m) in enumerate(zip(examples, meta)):
        if demonstrations:
            messages = few_shot.build_messages(demonstrations, ex["prompt"])
        else:
            messages = [{"role": "user", "content": ex["prompt"]}]
        chat_prompt = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False,
        )
        t0 = time.time()
        raw_output = generate(model, tokenizer, prompt=chat_prompt, max_tokens=32, verbose=False)
        elapsed = time.time() - t0

        predicted = parse_numeric(raw_output)
        correct = (
            predicted is not None
            and values_agree(predicted, m["true_value"], m["unit"])
        )
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
            print(f"  {label}: {i + 1}/{len(examples)} done", flush=True)

    n_correct = sum(r["correct"] for r in records)
    print(f"{label}: {n_correct}/{len(records)} correct ({n_correct / len(records):.1%})", flush=True)
    return records


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter-path", default=None)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--label", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0,
                         help="skip this many test examples before --limit is applied, for chunked runs")
    parser.add_argument("--few-shot-k", type=int, default=0,
                         help="number of train-split demonstrations to prepend (0 = zero-shot)")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(exist_ok=True)
    records = run_eval(args.adapter_path, args.label, args.limit, model_path=args.model_path,
                        few_shot_k=args.few_shot_k, offset=args.offset)
    out_path = RESULTS_DIR / f"predictions_{args.label}.json"
    with open(out_path, "w") as f:
        json.dump(records, f, indent=2)
    print(f"written to {out_path}")
