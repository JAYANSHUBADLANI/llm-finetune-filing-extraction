import json
from pathlib import Path

import pandas as pd
import pytest

import data_prep

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "prepared"


@pytest.fixture(autouse=True)
def prepared_fixtures(monkeypatch):
    """Read prepared filings from the frozen copies in tests/fixtures.

    These tests parse two real filings, which live in the separate
    filing-extraction-benchmark project. Pointing at the checkout beside this
    one makes them pass on the machine that happens to have it and fail
    everywhere else, so the two filings they need are committed here instead.
    See tests/fixtures/prepared/README.md.
    """
    monkeypatch.setattr(data_prep, "PREPARED_DIR", FIXTURE_DIR)


@pytest.fixture(scope="module")
def a_real_doc_id():
    test_examples = json.load(open(DATA_DIR / "test_examples.json"))
    return test_examples[0]["doc_id"]


def test_load_statement_text_tags_every_table_with_a_scale(a_real_doc_id):
    text = data_prep.load_statement_text(a_real_doc_id)
    assert text.startswith("[scale:")


def test_bare_currency_symbol_cell_is_dropped():
    # The real bug this guards against: the source project's parser
    # leaves a bare "$" as its own cell on some rows but not others,
    # which inflates that row's cell count relative to the header row
    # and misaligns which pipe-separated position is which year.
    assert not data_prep._is_meaningful_cell("$")
    assert not data_prep._is_meaningful_cell(" $ ")


def test_bare_closing_paren_cell_is_dropped():
    assert not data_prep._is_meaningful_cell(")")


def test_zero_width_space_only_cell_is_dropped():
    # str.strip() does not remove U+200B, so this cell silently passed
    # the old `if str(c).strip()` filter as if it were real content.
    assert not data_prep._is_meaningful_cell("​")
    assert not data_prep._is_meaningful_cell("​​")


def test_a_number_cell_is_kept():
    assert data_prep._is_meaningful_cell("(13,583)")
    assert data_prep._is_meaningful_cell("2020")


def test_a_label_cell_is_kept():
    assert data_prep._is_meaningful_cell("Purchases of investments")


def test_load_statement_text_rows_align_with_their_header():
    # Every data row under a two-year header should carry exactly as
    # many pipe-separated fields as that header, not more, the direct,
    # observable consequence of the filler-cell fix above. This specific
    # real filing has a bare "$" cell on some of its rows but not others
    # before the fix, which is exactly what misaligned columns (see
    # PROGRESS.md); picked for this test because the bug was confirmed
    # against it directly, not because it's representative of all docs.
    text = data_prep.load_statement_text("0000021344_0000021344-21-000008", max_chars=1500)
    lines = text.split("\n")
    header = next(l for l in lines if l.startswith("Year Ended"))
    n_header_fields = len(header.split(" | "))
    data_lines = lines[lines.index(header) + 1:]
    for line in data_lines:
        if not line.strip() or line.startswith("[scale"):
            break
        assert len(line.split(" | ")) == n_header_fields, line


def test_load_statement_text_respects_max_chars(a_real_doc_id):
    text = data_prep.load_statement_text(a_real_doc_id, max_chars=500)
    assert len(text) <= 500


def test_all_nine_fields_have_labels():
    expected = {
        "revenue", "operating_income", "net_income", "eps_diluted",
        "total_assets", "total_liabilities", "stockholders_equity",
        "cash_and_equivalents", "operating_cash_flow",
    }
    assert set(data_prep.FIELD_LABELS) == expected


def test_build_examples_skips_docs_with_no_cached_statement(monkeypatch, capsys):
    def fake_load(doc_id, max_chars=3200):
        if doc_id == "missing_doc":
            raise FileNotFoundError
        return "[scale: as shown, not scaled]\nrevenue | 100"

    monkeypatch.setattr(data_prep, "load_statement_text", fake_load)
    ground_truth = pd.DataFrame([
        {"doc_id": "present_doc", "company": "Acme", "field": "revenue",
         "true_value": 100.0, "unit": "USD"},
        {"doc_id": "missing_doc", "company": "Acme", "field": "revenue",
         "true_value": 100.0, "unit": "USD"},
    ])

    examples = data_prep.build_examples(ground_truth)

    assert [e["doc_id"] for e in examples] == ["present_doc"]
    assert "1 doc_ids had no cached prepared JSON" in capsys.readouterr().out


def test_build_examples_embeds_scale_instruction_in_question(monkeypatch):
    monkeypatch.setattr(
        data_prep, "load_statement_text",
        lambda doc_id, max_chars=3200: "[scale: Dollars in millions]\nrevenue | 1,234",
    )
    ground_truth = pd.DataFrame([
        {"doc_id": "d1", "company": "Acme", "field": "net_income",
         "true_value": 1234000000.0, "unit": "USD"},
    ])

    [example] = data_prep.build_examples(ground_truth)

    assert "net income" in example["question"]
    assert "scale" in example["question"].lower()
    assert example["statement_text"] == "[scale: Dollars in millions]\nrevenue | 1,234"
