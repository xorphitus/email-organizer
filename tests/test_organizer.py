import pytest

from email_organizer import INBOX
from email_organizer.organizer import run_account, undo_run

from conftest import FakeProvider, StubClassifier, make_msg


def _msgs():
    return [
        make_msg(1, "Your receipt"),
        make_msg(2, "Weekly news"),
        make_msg(3, "hi there"),
        make_msg(4, "???"),  # below threshold
        make_msg(5, "Another receipt"),
    ]


def _run(provider, state, categories, clf=None, dry_run=False, **kw):
    clf = clf or StubClassifier()
    run_id = state.start_run(dry_run)
    out = run_account(provider, lambda: clf, state, categories, run_id, limit=kw.pop("limit", 50), dry_run=dry_run, **kw)
    return run_id, out


def test_run_moves_and_logs(state, categories):
    p = FakeProvider(_msgs())
    run_id, out = _run(p, state, categories)
    actions = {o.message.key: (o.action, o.folder) for o in out}
    assert actions == {
        "key1": ("moved", "Receipts"),
        "key2": ("moved", "Newsletters"),
        "key3": ("kept", None),
        "key4": ("uncertain", None),
        "key5": ("moved", "Receipts"),
    }
    assert sorted(p.created) == ["Newsletters", "Receipts"]
    assert [m.key for m in p.folders["Receipts"]] == ["key1", "key5"]
    assert [m.key for m in p.folders[INBOX]] == ["key3", "key4"]

    rows = state.decisions(run_id)
    assert len(rows) == 5
    r4 = next(r for r in rows if r.msg_key == "key4")
    assert r4.action == "uncertain" and r4.category == "receipts" and r4.probs["receipts"] == 0.4
    assert all(state.is_processed("fake", f"key{i}") for i in range(1, 6))


def test_second_run_skips_processed(state, categories):
    p = FakeProvider(_msgs())
    _run(p, state, categories)
    clf = StubClassifier()
    _, out = _run(p, state, categories, clf=clf)
    assert out == [] and clf.calls == 0  # model not even loaded
    _, out = _run(p, state, categories, reprocess=True)
    assert {o.message.key for o in out} == {"key3", "key4"}


def test_dry_run_moves_nothing(state, categories):
    p = FakeProvider(_msgs())
    run_id, out = _run(p, state, categories, dry_run=True)
    assert p.moves == [] and p.created == []
    assert len(p.folders[INBOX]) == 5
    assert not state.is_processed("fake", "key1")
    assert {r.action for r in state.decisions(run_id)} == {"would-moved", "kept", "uncertain"}
    with pytest.raises(ValueError, match="dry run"):
        undo_run(state, run_id, lambda name: p)


def test_limit(state, categories):
    p = FakeProvider(_msgs())
    _, out = _run(p, state, categories, limit=2)
    assert [o.message.key for o in out] == ["key1", "key2"]


def test_move_failure_is_not_marked_processed(state, categories):
    p = FakeProvider(_msgs())
    p.fail_folder = "Newsletters"
    run_id, out = _run(p, state, categories)
    assert next(o for o in out if o.message.key == "key2").action == "error"
    assert not state.is_processed("fake", "key2")
    assert state.is_processed("fake", "key1")  # other folders still went through


def test_undo_restores_inbox(state, categories):
    p = FakeProvider(_msgs())
    run_id, _ = _run(p, state, categories)
    counts = undo_run(state, run_id, lambda name: p)
    assert counts == {"fake": 3}
    assert sorted(m.key for m in p.folders[INBOX]) == [f"key{i}" for i in range(1, 6)]
    assert p.folders["Receipts"] == [] and p.folders["Newsletters"] == []
    # undone messages are reconsidered next run; kept/uncertain ones stay processed
    assert not state.is_processed("fake", "key1")
    assert state.is_processed("fake", "key3")
    # undo is idempotent
    assert undo_run(state, run_id, lambda name: p) == {}
    assert sum(r["undone"] for r in state.recent_runs()) == 3


def test_undo_unknown_run(state):
    with pytest.raises(KeyError):
        undo_run(state, "nope", lambda name: None)
