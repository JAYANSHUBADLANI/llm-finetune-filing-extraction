"""
Builds fine-tuning and evaluation examples from the filing-extraction-benchmark
project's already labeled data, read only, nothing in that project is
modified. Reuses its ground truth (1,225 labeled cells across 9 financial
statement fields for 24 real S&P sized companies' 10-Ks), its dev/test
split, and its cached, pre-parsed filing tables, rather than re-scraping
SEC EDGAR or inventing a new labeled set.

Input compaction: that project's own language model comparison used the
full stripped filing text, up to about 560,000 characters, which is the
honest apples to apples comparison for a frontier API model but is not
practical to LoRA fine-tune a 1B to 3B parameter model against on a
laptop. Uses only the tables the source project's own parser already
classified as a financial statement (income_statement, balance_sheet,
cash_flow_statement), typically a few thousand characters, not the full
document. This is a different, cheaper context mode than the "document"
mode the 0.9321 test accuracy figure in that project's README was
measured under, and that difference is carried through to every result
here rather than glossed over: see README.md for what this does and does
not make comparable.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

SOURCE_PROJECT = Path(__file__).resolve().parents[2] / "filing-extraction-benchmark"
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

# Named rather than built inline at the call site so the tests can point it at
# their own frozen copies of two filings and run without the source project
# checked out. Regenerating the dataset still reads the real cache.
PREPARED_DIR = SOURCE_PROJECT / "data" / "cache" / "prepared"

FIELD_LABELS = {
    "revenue": "total revenue",
    "operating_income": "operating income",
    "net_income": "net income",
    "eps_diluted": "diluted earnings per share",
    "total_assets": "total assets",
    "total_liabilities": "total liabilities",
    "stockholders_equity": "total stockholders equity",
    "cash_and_equivalents": "cash and cash equivalents",
    "operating_cash_flow": "net cash provided by operating activities",
}


def _is_meaningful_cell(cell: str) -> bool:
    """A cell carries no information for column alignment unless it has
    a letter or digit. The source project's own table parser leaves a
    lot of filler in `row` that isn't visible as such: a bare "$" split
    into its own cell (3,480 occurrences sampled across 30 filings), a
    bare ")" from a negative-number sign split off from its digits
    (1,151), and zero-width space characters (2,873) that `str.strip()`
    does not remove, since U+200B isn't whitespace by Python's
    definition, so it silently passed the old `if str(c).strip()` check
    as if it were real content. Every one of these inflates a row's cell
    count relative to the header row above it without adding a value,
    which misaligns which pipe-separated position corresponds to which
    reported year between the header and the data rows beneath it, a
    real, checked-against-the-actual-cached-data cause of the wrong_row
    and wrong_period failures analyze_errors.py's breakdown found
    dominating every method's mistakes. See PROGRESS.md.
    """
    return bool(re.search(r"[0-9A-Za-z]", cell.replace("​", "")))


def load_statement_text(doc_id: str, max_chars: int = 3200) -> str:
    # Each table carries its own resolved scale (for example 1,000,000 for
    # a "Dollars in millions" caption), already computed by the source
    # project's parser, and different tables in the same filing can use
    # different scales. Tagging each table's block with its own scale
    # phrase, rather than concatenating all rows with no scale context, is
    # necessary: without it a model reads "76,559" off an income statement
    # captioned in millions and answers 76559 instead of 76559000000, a
    # scaling error, not an extraction error, caught during development
    # by inspecting a zero shot run's raw outputs, see PROGRESS.md.
    path = PREPARED_DIR / f"{doc_id}.json"
    with open(path) as f:
        doc = json.load(f)
    parts = []
    for table in doc["tables"]:
        if table["statement"] is None:
            continue
        rows_text = []
        for row in table["rows"]:
            cells = [str(c).strip() for c in row if _is_meaningful_cell(str(c))]
            line = " | ".join(cells)
            if line:
                rows_text.append(line)
        if not rows_text:
            continue
        scale_note = f"[scale: {table['scale_phrase']}]" if table["scale_phrase"] else "[scale: as shown, not scaled]"
        parts.append(f"{scale_note}\n" + "\n".join(rows_text))
    text = "\n\n".join(parts)
    return text[:max_chars]


def build_examples(ground_truth: pd.DataFrame) -> list[dict]:
    examples = []
    missing_docs = set()
    for doc_id, group in ground_truth.groupby("doc_id"):
        try:
            statement_text = load_statement_text(doc_id)
        except FileNotFoundError:
            missing_docs.add(doc_id)
            continue
        if not statement_text.strip():
            continue
        for _, row in group.iterrows():
            field_label = FIELD_LABELS[row["field"]]
            examples.append({
                "doc_id": doc_id,
                "company": row["company"],
                "field": row["field"],
                "statement_text": statement_text,
                "question": f"What is the {field_label} reported in these financial statement tables? "
                            f"Each table states its own scale in brackets before it, for example "
                            f"[scale: Dollars in millions] means a value shown as 1,234 is actually "
                            f"1,234,000,000. Convert the value you find to the actual full number "
                            f"using that table's stated scale before answering. "
                            f"Answer with only the final number in full, unscaled units, no "
                            f"currency symbol, no commentary.",
                "true_value": row["true_value"],
                "unit": row["unit"],
            })
    if missing_docs:
        print(f"warning: {len(missing_docs)} doc_ids had no cached prepared JSON, skipped: "
              f"{sorted(missing_docs)[:5]}{'...' if len(missing_docs) > 5 else ''}")
    return examples


def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    ground_truth = pd.read_csv(SOURCE_PROJECT / "results" / "ground_truth.csv")
    split = pd.read_csv(SOURCE_PROJECT / "results" / "split.csv")
    split = split[split["split"].isin(["dev", "test"])][["cik", "split"]]

    ground_truth = ground_truth.merge(split, on="cik", how="inner")
    print(f"ground truth rows after joining split: {len(ground_truth)}")
    print(ground_truth["split"].value_counts())

    for split_name in ["dev", "test"]:
        subset = ground_truth[ground_truth["split"] == split_name]
        examples = build_examples(subset)
        print(f"{split_name}: {len(examples)} examples built "
              f"from {subset['doc_id'].nunique()} filings")
        with open(DATA_DIR / f"{split_name}_examples.json", "w") as f:
            json.dump(examples, f, indent=2)


if __name__ == "__main__":
    main()
