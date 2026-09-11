"""
Confirms the scoring rule behaves as this project's own README describes it
(0.5% relative tolerance, absolute tolerance for per-share values) on this
project's own field types, and that the local copy of that rule has not
drifted from the filing-extraction-benchmark project it was copied from.

The tolerance tests run everywhere. The drift test needs the source project
checked out beside this one, so it skips when it is absent rather than
failing: a machine without that checkout can still verify the rule this
project's README depends on, it just cannot verify the two are identical.
Not a test of filing-extraction-benchmark's own correctness, that project
has its own test suite.
"""

import itertools
import sys
from pathlib import Path

import pytest

from scoring_rules import classify, is_scaling_error, values_agree

SOURCE_SRC = Path(__file__).resolve().parents[2] / "filing-extraction-benchmark" / "src"


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


def _source_scoring():
    """The source project's module, or None when it is not checked out here."""
    if not (SOURCE_SRC / "filingbench" / "scoring.py").exists():
        return None
    sys.path.insert(0, str(SOURCE_SRC))
    try:
        from filingbench import scoring
    except ImportError:
        return None
    return scoring


# Spans the cases the copied rules actually branch on: exact agreement, either
# side of the 0.5% relative band, powers of ten in both directions, zero truth,
# and per-share values around the half-cent absolute tolerance.
_VALUES = [0.0, 3.27, 3.275, 3.35, 76_559.0, 76_940_705.0, 77_000_000.0,
           76_559_000.0, 76_559_000_000.0, -1234.5]
_UNITS = ["USD", "USD/shares"]
_STATUSES = ["unparseable_response", "row_not_found", "not_attempted", "llm_error"]


def test_local_rules_match_the_source_project_cell_for_cell():
    scoring = _source_scoring()
    if scoring is None:
        pytest.skip("filing-extraction-benchmark is not checked out beside this project")

    for predicted, truth, unit in itertools.product(_VALUES, _VALUES, _UNITS):
        assert values_agree(predicted, truth, unit) == scoring.values_agree(predicted, truth, unit), (
            f"values_agree drifted at {predicted=} {truth=} {unit=}"
        )
        assert is_scaling_error(predicted, truth) == scoring.is_scaling_error(predicted, truth), (
            f"is_scaling_error drifted at {predicted=} {truth=}"
        )

    others = [3.27, 76_559_000.0]
    for predicted, truth, unit, status in itertools.product(
        [*_VALUES, None], _VALUES, _UNITS, _STATUSES
    ):
        assert classify(predicted, status, truth, unit, others) == scoring.classify(
            predicted, status, truth, unit, others
        ), f"classify drifted at {predicted=} {status=} {truth=} {unit=}"


def test_status_sets_match_the_source_project():
    scoring = _source_scoring()
    if scoring is None:
        pytest.skip("filing-extraction-benchmark is not checked out beside this project")

    import scoring_rules

    assert scoring_rules.PARSE_ERROR_STATUSES == scoring.PARSE_ERROR_STATUSES
    assert scoring_rules.NOT_FOUND_STATUSES == scoring.NOT_FOUND_STATUSES
    assert scoring_rules.RELATIVE_TOLERANCE == scoring.RELATIVE_TOLERANCE
    assert scoring_rules.ABSOLUTE_TOLERANCE_PER_SHARE == scoring.ABSOLUTE_TOLERANCE_PER_SHARE
    assert scoring_rules.POWERS_OF_TEN == scoring.POWERS_OF_TEN
