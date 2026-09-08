"""
Checks whether the ground truth value for each test example actually
appears anywhere in the table text the model was given, at all, a
question no earlier analysis in this project asked. Motivated by a real
finding while investigating why wrong_row failures cluster so heavily
on Microsoft (98.1% wrong_row at the best model, see PROGRESS.md): its
statement_text for at least one filing turned out to be footnote
schedule tables (an allowance-for-doubtful-accounts rollforward, an
unrecognized-tax-benefits reconciliation) with no income statement or
balance sheet content at all, not a hard table for the model to read
correctly but no table with the right answer in it at all.

This is necessarily a heuristic, not an exact check: it looks for the
true value's digits at a handful of common financial-statement scales
(as reported, thousands, millions, billions) with ordinary thousands-
separator and parenthetical-negative formatting, not an exact match to
whatever rounding or precision the filer's own XBRL tag carries versus
what's printed. A true value could in principle be "present" in a form
this check doesn't recognize (unusual formatting), which would make this
UNDERSTATE the coverage problem, not overstate it, a value flagged
"present" is fairly reliable, a value flagged "missing" a bit less
mechanically certain, though the concentration on specific companies and
specific subtotal fields (not spread evenly, see main()) is corroborating
evidence this reflects something real about the underlying data, not
just noise from the heuristic.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"

SCALES = (1.0, 1e3, 1e6, 1e9)


def value_plausibly_present(text: str, true_value: float, unit: str) -> bool:
    candidates = [true_value] if unit == "USD/shares" else [true_value / s for s in SCALES]
    for c in candidates:
        magnitude = abs(c)
        for rounding in (0, 1):
            formatted = f"{magnitude:,.{rounding}f}".rstrip("0").rstrip(".")
            variants = [formatted, formatted.replace(",", "")]
            if c < 0:
                variants += [f"({formatted}", f"({formatted})", f"-{formatted}"]
            if any(v in text for v in variants):
                return True
    return False


def find_unanswerable_examples(examples: list[dict]) -> list[dict]:
    return [e for e in examples if not value_plausibly_present(e["statement_text"], e["true_value"], e["unit"])]


def achievable_accuracy(label: str, unanswerable_keys: set[tuple[str, str]]) -> tuple[int, int, float]:
    """Accuracy on only the examples where the answer was actually present
    in what the model was given, the fairer number to credit or blame a
    model for, separate from a data-coverage problem it can't fix by
    reading better."""
    records = json.load(open(RESULTS_DIR / f"predictions_{label}.json"))
    answerable = [r for r in records if (r["doc_id"], r["field"]) not in unanswerable_keys]
    n_correct = sum(r["correct"] for r in answerable)
    return n_correct, len(answerable), n_correct / len(answerable) if answerable else 0.0


def main() -> None:
    test = json.load(open(DATA_DIR / "test_examples.json"))
    unanswerable = find_unanswerable_examples(test)
    unanswerable_keys = {(e["doc_id"], e["field"]) for e in unanswerable}

    print(f"{len(unanswerable)}/{len(test)} test examples have no plausible representation "
          f"of the true value anywhere in the given table text ({len(unanswerable) / len(test):.1%})")

    by_field = Counter(e["field"] for e in unanswerable)
    print("\nby field:")
    for field, n in by_field.most_common():
        total = sum(1 for e in test if e["field"] == field)
        print(f"  {field:25s} {n:3d}/{total:<4d}  ({n / total:.1%})")

    by_doc = Counter(e["doc_id"] for e in unanswerable)
    fully = sum(1 for doc_id in {e["doc_id"] for e in test}
                if all((doc_id, f) in unanswerable_keys
                       for f in [e["field"] for e in test if e["doc_id"] == doc_id]))
    print(f"\n{len(by_doc)} distinct filings have at least one unanswerable field; "
          f"{fully} have every field unanswerable")

    print("\nachievable accuracy (excluding unanswerable examples) vs raw accuracy:")
    for label in ["zeroshot_base", "fewshot_base_full", "finetuned_v3_full",
                  "zeroshot_7b_full", "fewshot_7b_full", "finetuned_7b_full"]:
        try:
            records = json.load(open(RESULTS_DIR / f"predictions_{label}.json"))
        except FileNotFoundError:
            continue
        raw_correct = sum(r["correct"] for r in records)
        n_correct, n_total, rate = achievable_accuracy(label, unanswerable_keys)
        print(f"  {label:20s} raw {raw_correct}/{len(records)} ({raw_correct / len(records):.1%})  "
              f"-> achievable {n_correct}/{n_total} ({rate:.1%})")


if __name__ == "__main__":
    main()
