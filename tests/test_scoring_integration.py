"""
Confirms the source project's scoring function is importable and behaves
as this project's own README describes it (0.5% relative tolerance,
absolute tolerance for per-share values), on this project's own field
types. Not a test of filing-extraction-benchmark's own correctness, that project has its own test suite, only that the import path used by
evaluate_model.py resolves and the specific tolerance rule every result
in this project's README depends on has not silently changed.
"""

from filingbench.scoring import values_agree


def test_exact_match_is_correct():
    assert values_agree(76_559_000_000.0, 76_559_000_000.0, "USD")


def test_within_half_percent_relative_tolerance_is_correct():
    assert values_agree(76_940_705.0, 76_559_000.0, "USD")  # +0.499%


def test_just_outside_half_percent_relative_tolerance_is_wrong():
    assert not values_agree(77_000_000.0, 76_559_000.0, "USD")  # +0.576%


def test_scaling_error_is_wrong():
    # The exact failure mode this project's scale-tagging fix targets:
    # answering in the table's displayed scale instead of full units.
    assert not values_agree(76_559.0, 76_559_000_000.0, "USD")


def test_eps_accepts_within_half_cent_absolute_tolerance():
    assert values_agree(3.275, 3.27, "USD/shares")


def test_eps_rejects_a_miss_outside_both_absolute_and_relative_tolerance():
    # 0.08 misses the 0.005 absolute EPS tolerance, and 0.08/3.27 = 2.4%
    # also misses the 0.5% relative tolerance the general case falls
    # back to, so this has to be wrong under either rule.
    assert not values_agree(3.35, 3.27, "USD/shares")
