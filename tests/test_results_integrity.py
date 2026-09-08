"""
Guards the specific numeric claims made in README.md against the checked-in
results/*.json files going stale (regenerated with different examples,
edited by hand, etc.) without the README being updated to match. Not a
test of model quality, a test that the numbers in prose and the data
they're computed from still agree.
"""

import json
from pathlib import Path

from significance_test import mcnemar_test

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def _load(name):
    return json.load(open(RESULTS_DIR / f"predictions_{name}.json"))


def _accuracy(records):
    return sum(r["correct"] for r in records) / len(records)


def _key(record):
    return (record["doc_id"], record["field"])


def test_zeroshot_full_matches_readme():
    records = _load("zeroshot_base")
    assert len(records) == 653
    assert _accuracy(records) == 23 / 653


def test_zeroshot_and_finetuned_20_example_slices_are_the_identical_inputs():
    # README's whole comparison depends on this: the 0%/0%/15% figures
    # are only comparable because all three runs scored the same 20
    # (doc_id, field) pairs, not three different 20-example samples.
    zeroshot_first_20 = _load("zeroshot_base")[:20]
    run1 = _load("finetuned_test")
    run3 = _load("finetuned_v3_test")

    assert len(run1) == len(run3) == 20
    keys = [_key(r) for r in zeroshot_first_20]
    assert [_key(r) for r in run1] == keys
    assert [_key(r) for r in run3] == keys


def test_shared_slice_accuracies_match_readme():
    assert _accuracy(_load("zeroshot_base")[:20]) == 0 / 20
    assert _accuracy(_load("finetuned_test")) == 0 / 20
    assert _accuracy(_load("finetuned_v3_test")) == 3 / 20


def test_run1_and_zeroshot_predictions_mostly_identical_as_progress_md_claims():
    zeroshot_first_20 = _load("zeroshot_base")[:20]
    run1 = _load("finetuned_test")

    identical = sum(
        1 for a, b in zip(zeroshot_first_20, run1)
        if a["predicted"] == b["predicted"]
        or (a["predicted"] is not None and b["predicted"] is not None
            and abs(a["predicted"] - b["predicted"]) < 1e-6)
    )
    assert identical == 13


def test_full_653_headline_comparison_matches_readme():
    zeroshot = _load("zeroshot_base")
    fewshot = _load("fewshot_base_full")
    finetuned = _load("finetuned_v3_full")

    assert len(zeroshot) == len(fewshot) == len(finetuned) == 653
    keys = {_key(r) for r in zeroshot}
    assert {_key(r) for r in fewshot} == keys
    assert {_key(r) for r in finetuned} == keys

    assert _accuracy(zeroshot) == 23 / 653
    assert _accuracy(fewshot) == 122 / 653
    assert _accuracy(finetuned) == 141 / 653


def test_finetuned_vs_fewshot_difference_is_not_significant_as_readme_states():
    # The project's central, easy-to-get-wrong claim: fine-tuning scores
    # numerically higher than free few-shot prompting on the full test
    # set, but a paired test says that gap is not distinguishable from
    # noise at this sample size. If this test ever fails, the README's
    # conclusion needs re-deriving, not the test loosened to match.
    fewshot = {_key(r): r["correct"] for r in _load("fewshot_base_full")}
    finetuned = {_key(r): r["correct"] for r in _load("finetuned_v3_full")}

    result = mcnemar_test(fewshot, finetuned)

    assert result["p_value"] > 0.05
    assert result["p_value"] == 0.14537 or abs(result["p_value"] - 0.14537) < 1e-4


def test_both_methods_beat_zeroshot_significantly():
    zeroshot = {_key(r): r["correct"] for r in _load("zeroshot_base")}
    fewshot = {_key(r): r["correct"] for r in _load("fewshot_base_full")}
    finetuned = {_key(r): r["correct"] for r in _load("finetuned_v3_full")}

    assert mcnemar_test(zeroshot, fewshot)["p_value"] < 0.001
    assert mcnemar_test(zeroshot, finetuned)["p_value"] < 0.001


def test_finetuned_7b_full_matches_readme():
    records = _load("finetuned_7b_full")
    assert len(records) == 653
    assert _accuracy(records) == 283 / 653


def test_finetuned_7b_full_covers_the_identical_653_questions_in_order():
    # Built by concatenating 6 chunk files (see PROGRESS.md: full-653
    # single-process runs stalled repeatedly, chunking worked around it)
    #, this guards that the concatenation didn't drop, duplicate, or
    # reorder anything relative to every other method's result file.
    zeroshot = _load("zeroshot_base")
    finetuned_7b = _load("finetuned_7b_full")
    assert [_key(r) for r in finetuned_7b] == [_key(r) for r in zeroshot]


def test_7b_finetuned_beats_3b_scale_results_significantly():
    finetuned_3b = {_key(r): r["correct"] for r in _load("finetuned_v3_full")}
    fewshot_3b = {_key(r): r["correct"] for r in _load("fewshot_base_full")}
    finetuned_7b = {_key(r): r["correct"] for r in _load("finetuned_7b_full")}

    assert mcnemar_test(finetuned_3b, finetuned_7b)["p_value"] < 1e-20
    assert mcnemar_test(fewshot_3b, finetuned_7b)["p_value"] < 1e-20


def test_zeroshot_7b_and_fewshot_7b_full_match_readme():
    zeroshot_7b = _load("zeroshot_7b_full")
    fewshot_7b = _load("fewshot_7b_full")
    assert len(zeroshot_7b) == len(fewshot_7b) == 653
    assert _accuracy(zeroshot_7b) == 202 / 653
    assert _accuracy(fewshot_7b) == 272 / 653

    zeroshot_3b_keys = [_key(r) for r in _load("zeroshot_base")]
    assert [_key(r) for r in zeroshot_7b] == zeroshot_3b_keys
    assert [_key(r) for r in fewshot_7b] == zeroshot_3b_keys


def test_7b_fewshot_vs_finetuned_gap_is_not_significant_even_more_clearly_than_3b():
    # The 3B version of this comparison was already not significant
    # (p=0.145). At 7B the two methods are even closer in raw accuracy
    # (41.7% vs 43.3%) and the paired test agrees even more strongly
    # that there's no established difference, this is the same finding
    # replicating and strengthening at a larger model scale, not a
    # coincidence specific to 3B.
    fewshot_7b = {_key(r): r["correct"] for r in _load("fewshot_7b_full")}
    finetuned_7b = {_key(r): r["correct"] for r in _load("finetuned_7b_full")}

    result = mcnemar_test(fewshot_7b, finetuned_7b)
    assert result["p_value"] > 0.05


def test_7b_zeroshot_beats_3b_finetuned_significantly():
    # The single most surprising number in this project: a 7B model that
    # was never fine-tuned at all beats a 3B model that WAS fine-tuned,
    # significantly. Model scale matters more than the training this
    # project spent the most effort on.
    zeroshot_7b = {_key(r): r["correct"] for r in _load("zeroshot_7b_full")}
    finetuned_3b = {_key(r): r["correct"] for r in _load("finetuned_v3_full")}

    assert _accuracy(_load("zeroshot_7b_full")) > _accuracy(_load("finetuned_v3_full"))
    result = mcnemar_test(finetuned_3b, zeroshot_7b)
    assert result["p_value"] < 0.001


def test_lr_1e4_is_worse_than_5e5_despite_a_clean_loss_curve():
    # The point of this run: a smooth validation loss curve (1.973 ->
    # 0.955 -> 1.006, no oscillation) did not predict this being worse
    # than 5e-5 at actual generation accuracy. Guards the specific claim
    # in README.md's "Learning rate tuning" section against silent drift
    # if these result files are ever regenerated.
    lr_1e4 = _load("finetuned_v4_1e4_100")
    lr_5e5_same_slice = _load("finetuned_v3_full")[:100]

    assert len(lr_1e4) == len(lr_5e5_same_slice) == 100
    assert _accuracy(lr_1e4) == 12 / 100
    assert _accuracy(lr_5e5_same_slice) == 21 / 100
    assert _accuracy(lr_1e4) < _accuracy(lr_5e5_same_slice)
