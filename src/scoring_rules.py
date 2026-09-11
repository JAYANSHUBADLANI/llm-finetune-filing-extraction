"""The scoring rules this project's numbers are measured under.

These are a verbatim copy of the primitives in the filing-extraction-benchmark
project's `filingbench/scoring.py`. They are copied rather than imported
because that project is a separate repository: importing it across a sibling
directory works on a machine that happens to have both checked out next to
each other, and fails everywhere else, including CI.

Copying a scorer risks the two drifting apart, which would quietly make this
project's accuracy figures incomparable to the ones its README compares them
against. `tests/test_scoring_integration.py` guards that: when the source
project is checked out alongside this one it asserts the two implementations
agree cell for cell, and skips when it is not there.

Only the three functions this project actually calls are copied. The dataframe
level scoring, the Wilson intervals and the reporting helpers stay in the
source project, which is the only place that needs them.
"""

from __future__ import annotations

import math

RELATIVE_TOLERANCE = 0.005
ABSOLUTE_TOLERANCE_PER_SHARE = 0.005
POWERS_OF_TEN = [10 ** k for k in (-9, -6, -3, -2, -1, 1, 2, 3, 6, 9)]

NOT_FOUND_STATUSES = {"statement_not_found", "row_not_found", "column_unresolved",
                      "not_attempted", "llm_no_answer", "llm_error"}
PARSE_ERROR_STATUSES = {"no_value_in_row", "unparseable_response"}


def values_agree(predicted: float, truth: float, unit: str) -> bool:
    if unit == "USD/shares":
        if abs(predicted - truth) <= ABSOLUTE_TOLERANCE_PER_SHARE:
            return True
    if truth == 0:
        return abs(predicted) <= ABSOLUTE_TOLERANCE_PER_SHARE
    return abs(predicted - truth) / abs(truth) <= RELATIVE_TOLERANCE


def is_scaling_error(predicted: float, truth: float) -> bool:
    if predicted == 0 or truth == 0:
        return False
    ratio = predicted / truth
    return any(math.isclose(ratio, power, rel_tol=RELATIVE_TOLERANCE)
               for power in POWERS_OF_TEN)


def classify(predicted: float | None, status: str, truth: float, unit: str,
             other_period_values: list[float]) -> str:
    if predicted is None:
        if status in PARSE_ERROR_STATUSES:
            return "parse_error"
        return "not_found"
    if values_agree(predicted, truth, unit):
        return "correct"
    if is_scaling_error(predicted, truth):
        return "scaling_error"
    for other in other_period_values:
        if values_agree(predicted, other, unit):
            return "wrong_period"
    return "wrong_row"
