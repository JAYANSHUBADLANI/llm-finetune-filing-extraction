from audit_input_coverage import (
    achievable_accuracy,
    find_unanswerable_examples,
    value_plausibly_present,
)


def test_value_present_plain_millions_scale():
    text = "Total revenues | 58,158 | 76,559 | 101,127"
    assert value_plausibly_present(text, 76_559_000_000.0, "USD")


def test_value_present_negative_with_missing_closing_paren():
    # The real case this guards: the source table's row splitting leaves
    # a lone opening paren with no matching close ("(1,975" not
    # "(1,975)"), a side effect of data_prep.py's column-alignment fix
    # dropping bare ")" filler cells (see PROGRESS.md). A naive check for
    # the standard "(1,975)" accounting format alone would misclassify
    # this real, present value as missing.
    text = "(Loss)/earnings from operations | (1,975 | 11,987 | 10,344"
    assert value_plausibly_present(text, -1_975_000_000.0, "USD")


def test_value_present_per_share_uses_raw_units_not_scaled():
    text = "Diluted (loss)/earnings per share | ($1.12 | $17.85 | $13.85"
    assert value_plausibly_present(text, -1.12, "USD/shares")


def test_value_absent_from_unrelated_table():
    text = ("Beginning unrecognized tax benefits | 15,593 | 14,550 | 13,792\n"
            "Ending unrecognized tax benefits | 17,120 | 15,593 | 14,550")
    assert not value_plausibly_present(text, 411_976_000_000.0, "USD")


def test_find_unanswerable_examples_filters_correctly():
    examples = [
        {"statement_text": "Total revenues | 76,559", "true_value": 76_559_000_000.0, "unit": "USD",
         "doc_id": "a", "field": "revenue"},
        {"statement_text": "unrelated footnote text", "true_value": 411_976_000_000.0, "unit": "USD",
         "doc_id": "a", "field": "total_assets"},
    ]
    unanswerable = find_unanswerable_examples(examples)
    assert [e["field"] for e in unanswerable] == ["total_assets"]


def test_achievable_accuracy_excludes_only_unanswerable_keys(tmp_path, monkeypatch):
    import json

    import audit_input_coverage

    results_dir = tmp_path / "results"
    results_dir.mkdir()
    monkeypatch.setattr(audit_input_coverage, "RESULTS_DIR", results_dir)

    records = [
        {"doc_id": "a", "field": "revenue", "correct": True},
        {"doc_id": "a", "field": "total_assets", "correct": False},  # unanswerable, excluded
        {"doc_id": "b", "field": "revenue", "correct": False},
    ]
    with open(results_dir / "predictions_dummy.json", "w") as f:
        json.dump(records, f)

    n_correct, n_total, rate = achievable_accuracy("dummy", {("a", "total_assets")})

    assert n_total == 2
    assert n_correct == 1
    assert rate == 0.5
