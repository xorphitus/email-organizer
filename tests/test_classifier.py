import json

import pytest

from email_organizer.classifier import build_record, decide, is_prequantized, split_logits
from email_organizer.message import Message

from conftest import CATEGORIES


def test_build_record_shape():
    msg = Message(
        id="1", key="1", account="a", mailbox="INBOX", sender="Shop <s@x>", to="me@x",
        subject="Receipt", date="2026-10-01", list_unsubscribe="<mailto:u@x>",
        text_body="Total: $5\n-- \nsig",
    )
    rec = build_record(msg, CATEGORIES)
    assert rec["state"] == {
        "from": "Shop <s@x>", "to": "me@x", "subject": "Receipt", "date": "2026-10-01",
        "is_mailing_list": True, "body": "Total: $5",
    }
    q = rec["questions"]["category"]
    assert q["type"] == "choice"
    assert q["instructions"] == "Which folder should this email be filed in?"
    assert q["criteria"] == {k: c.criteria for k, c in CATEGORIES.items()}


def test_decide_threshold():
    d = decide({"a": 0.7, "b": 0.3}, 0.6)
    assert d.category == "a" and d.prob == pytest.approx(0.7) and not d.uncertain
    d = decide({"a": 0.55, "b": 0.45}, 0.6)
    assert d.category is None and d.top == "a" and d.uncertain
    assert decide({"a": 0.6, "b": 0.4}, 0.6).category == "a"  # inclusive


def test_split_logits_layouts():
    # single record: out[0] is that record's question list
    assert split_logits([["q1", "q2"]], [2]) == [["q1", "q2"]]
    # flattened: out[0] holds all questions in record order
    assert split_logits([["a1", "b1", "b2"], "extra"], [1, 2]) == [["a1"], ["b1", "b2"]]
    # per record: out[i] is record i
    assert split_logits([["a1"], ["b1"]], [1, 1]) == [["a1"], ["b1"]]
    with pytest.raises(ValueError):
        split_logits([["x", "y", "z"]], [1, 1])


def test_is_prequantized(tmp_path):
    assert not is_prequantized(tmp_path)
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "qwen3_5"}))
    assert not is_prequantized(tmp_path)
    (tmp_path / "config.json").write_text(json.dumps({"quantization_config": {"load_in_4bit": True}}))
    assert is_prequantized(tmp_path)
