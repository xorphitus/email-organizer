from __future__ import annotations

from collections.abc import Sequence

import pytest

from email_organizer import INBOX, is_inbox
from email_organizer.classifier import Decision, decide
from email_organizer.config import Category
from email_organizer.message import Message, MessageRef
from email_organizer.state import StateDB

CATEGORIES = {
    "receipts": Category(criteria="Receipts and invoices", folder="Receipts"),
    "newsletters": Category(criteria="Newsletters", folder="Newsletters"),
    "personal": Category(criteria="Personal mail", folder="INBOX"),
}


def make_msg(i: int, subject: str, account: str = "fake") -> Message:
    return Message(
        id=f"id{i}", key=f"key{i}", account=account, mailbox=INBOX,
        sender=f"sender{i}@example.com", to="me@example.com", subject=subject,
        date="2026-10-01T00:00:00Z", text_body=f"body {i}",
    )


class FakeProvider:
    """In-memory mailbox: folder name -> list of messages."""

    def __init__(self, messages: list[Message], name: str = "fake"):
        self.name = name
        self.folders: dict[str, list[Message]] = {INBOX: list(messages)}
        self.created: list[str] = []
        self.moves: list[tuple[list[str], str]] = []
        self.fail_folder: str | None = None

    def fetch_inbox(self, limit, skip=lambda key: False):
        return [m for m in self.folders[INBOX] if not skip(m.key)][:limit]

    def ensure_folders(self, names):
        for n in names:
            if n not in self.folders:
                self.folders[n] = []
                self.created.append(n)

    def move(self, refs: Sequence[MessageRef], folder: str):
        if folder == self.fail_folder:
            raise RuntimeError("boom")
        dest = INBOX if is_inbox(folder) else folder
        self.moves.append(([r.key for r in refs], dest))
        for r in refs:
            src = self.folders[INBOX if is_inbox(r.folder) else r.folder]
            msg = next(m for m in src if m.key == r.key)
            src.remove(msg)
            self.folders.setdefault(dest, []).append(msg)

    def close(self):
        pass


class StubClassifier:
    """Classifies by keyword in the subject; fixed probabilities."""

    def __init__(self, threshold: float = 0.6):
        self.threshold = threshold
        self.calls = 0

    def classify(self, messages):
        self.calls += 1
        out = []
        for m in messages:
            s = m.subject.lower()
            if "receipt" in s:
                probs = {"receipts": 0.9, "newsletters": 0.05, "personal": 0.05}
            elif "news" in s:
                probs = {"receipts": 0.1, "newsletters": 0.8, "personal": 0.1}
            elif "hi" in s:
                probs = {"receipts": 0.05, "newsletters": 0.05, "personal": 0.9}
            else:
                probs = {"receipts": 0.4, "newsletters": 0.35, "personal": 0.25}
            out.append(decide(probs, self.threshold))
        return out


@pytest.fixture
def state():
    db = StateDB(":memory:")
    yield db
    db.close()


@pytest.fixture
def categories():
    return CATEGORIES
