import pytest

from significance_test import mcnemar_test


def test_no_discordant_pairs_gives_p_one():
    a = {1: True, 2: False, 3: True}
    b = {1: True, 2: False, 3: True}
    result = mcnemar_test(a, b)
    assert result["discordant_n"] == 0
    assert result["p_value"] == 1.0


def test_perfectly_lopsided_disagreement_is_significant():
    # a is right and b is wrong on every one of 20 examples where they
    # disagree at all: about as clear a signal as a paired test can see.
    a = {i: True for i in range(20)}
    b = {i: False for i in range(20)}
    result = mcnemar_test(a, b)
    assert result["a_only"] == 20
    assert result["b_only"] == 0
    assert result["p_value"] < 0.001


def test_evenly_split_disagreement_is_not_significant():
    a = {i: (i % 2 == 0) for i in range(20)}   # right on evens
    b = {i: (i % 2 == 1) for i in range(20)}   # right on odds
    result = mcnemar_test(a, b)
    assert result["a_only"] == 10
    assert result["b_only"] == 10
    assert result["p_value"] == 1.0


def test_matches_the_real_fewshot_vs_finetuned_result():
    # Regression check against the actual numbers this project's README
    # reports: if this ever disagrees, either the underlying prediction
    # files changed or the test/train data changed, and the README's
    # significance claim needs re-deriving, not silently trusting.
    a = {i: True for i in range(67)} | {i: False for i in range(67, 67 + 86 + 55 + 445)}
    b = {i: False for i in range(67)} | {i: True for i in range(67, 67 + 86)} | \
        {i: False for i in range(67 + 86, 67 + 86 + 55 + 445)}
    result = mcnemar_test(a, b)
    assert result["a_only"] == 67
    assert result["b_only"] == 86
    assert result["discordant_n"] == 153
    assert result["p_value"] == pytest.approx(0.14537, abs=1e-4)


def test_raises_on_mismatched_keys():
    with pytest.raises(ValueError):
        mcnemar_test({1: True, 2: False}, {1: True, 3: False})
