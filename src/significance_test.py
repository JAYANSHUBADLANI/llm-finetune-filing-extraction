"""
McNemar's exact test for comparing two methods scored on the identical
set of examples. Needed here specifically because zero-shot, few-shot,
and fine-tuned were all evaluated on the same 653 (doc_id, field) pairs, a paired comparison, not two independent samples, the same reason
whisper-benchmark's own significance_test.py exists: a naive independent-
samples test on paired data over- or under-states significance depending
on how correlated the two methods' mistakes are, and the honest answer
here turned out to depend on getting this right (see README.md: fine-
tuned looks numerically ahead of few-shot, 21.6% vs 18.7%, but that gap
is not the kind a paired test calls significant).

Only the discordant pairs (one method right, the other wrong) carry any
information about which method is better; agreements, right or wrong,
say nothing about a difference between them. Exact (binomial), not the
chi-squared approximation, since the discordant count here (153) is
comfortably large but there is no reason to accept approximation error
for a one-line binomial CDF.
"""

from __future__ import annotations

from math import comb


def mcnemar_test(a: dict[tuple, bool], b: dict[tuple, bool]) -> dict:
    """a and b map the same set of example keys to correct/incorrect."""
    keys = set(a) & set(b)
    if keys != set(a) or keys != set(b):
        raise ValueError("a and b must be scored on the identical set of examples")

    a_only = sum(1 for k in keys if a[k] and not b[k])
    b_only = sum(1 for k in keys if b[k] and not a[k])
    n = a_only + b_only

    if n == 0:
        p_value = 1.0
    else:
        smaller = min(a_only, b_only)
        p_value = min(1.0, sum(comb(n, i) for i in range(smaller + 1)) * 2 / (2 ** n))

    return {"a_only": a_only, "b_only": b_only, "discordant_n": n, "p_value": p_value}


if __name__ == "__main__":
    import json
    import sys
    from pathlib import Path

    RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"

    def load(label):
        records = json.load(open(RESULTS_DIR / f"predictions_{label}.json"))
        return {(r["doc_id"], r["field"]): r["correct"] for r in records}

    label_a, label_b = sys.argv[1], sys.argv[2]
    result = mcnemar_test(load(label_a), load(label_b))
    print(f"{label_a} vs {label_b}: {label_a}-only={result['a_only']}, "
          f"{label_b}-only={result['b_only']}, "
          f"n={result['discordant_n']}, p={result['p_value']:.5f}")
