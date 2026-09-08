import json
from pathlib import Path

import few_shot

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def test_train_filings_disjoint_from_valid_filings():
    dev = json.load(open(DATA_DIR / "dev_examples.json"))
    all_filings = set(e["doc_id"] for e in dev)
    train = few_shot.train_filings()
    assert train < all_filings
    assert len(all_filings) - len(train) == 9  # matches test_format_for_training.py


def test_select_demonstrations_only_uses_train_filings():
    train = few_shot.train_filings()
    demos = few_shot.select_demonstrations(3)
    assert len(demos) == 3
    assert all(d["doc_id"] in train for d in demos)


def test_select_demonstrations_is_deterministic():
    assert few_shot.select_demonstrations(3) == few_shot.select_demonstrations(3)


def test_select_demonstrations_prefers_distinct_fields():
    demos = few_shot.select_demonstrations(3)
    fields = [d["field"] for d in demos]
    assert len(set(fields)) == len(fields)


def test_build_messages_alternates_user_and_assistant():
    demos = few_shot.select_demonstrations(2)
    messages = few_shot.build_messages(demos, "What is the real question?")

    assert len(messages) == 2 * len(demos) + 1
    assert [m["role"] for m in messages] == ["user", "assistant", "user", "assistant", "user"]
    assert messages[-1]["content"] == "What is the real question?"


def test_build_messages_demo_answer_is_bare_true_value():
    demos = few_shot.select_demonstrations(1)
    messages = few_shot.build_messages(demos, "real question")
    assert messages[1]["content"] == str(demos[0]["true_value"])


def test_build_messages_truncates_demo_statement_text():
    demos = few_shot.select_demonstrations(1)
    demo = demos[0]
    assert len(demo["statement_text"]) > few_shot.DEMO_MAX_CHARS  # precondition: truncation matters here

    messages = few_shot.build_messages(demos, "real question")

    assert demo["statement_text"][:few_shot.DEMO_MAX_CHARS] in messages[0]["content"]
    assert demo["statement_text"] not in messages[0]["content"]
