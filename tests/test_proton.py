import pytest

from email_organizer.config import ProtonAccount
from email_organizer.message import MessageRef
from email_organizer.providers.proton import ProtonProvider


def raw(n: int, subject: str, mid: bool = True) -> bytes:
    hdr = f"Message-ID: <m{n}@x>\r\n" if mid else ""
    return (
        f"From: Sender {n} <s{n}@x.com>\r\nTo: me@proton.me\r\nSubject: {subject}\r\n{hdr}"
        f"Date: Thu, 01 Oct 2026 10:00:00 +0000\r\nContent-Type: text/plain\r\n\r\nBody {n}\r\n"
    ).encode()


class FakeIMAP:
    """Just enough of IMAPClient: folders hold {uid: raw bytes}."""

    def __init__(self):
        self.folders = {"INBOX": {1: raw(1, "old"), 2: raw(2, "mid"), 3: raw(3, "new", mid=False)},
                        "Folders": {}, "Folders/Receipts": {}}
        self.validity = {name: 100 + i for i, name in enumerate(self.folders)}
        self.selected = None
        self.readonly = None
        self.log: list[tuple] = []
        self.next_uid = 50

    def list_folders(self):
        return [((), b"/", n) for n in self.folders]

    def create_folder(self, name):
        self.log.append(("CREATE", name))
        self.folders[name] = {}
        self.validity[name] = 999

    def select_folder(self, name, readonly=False):
        self.selected, self.readonly = name, readonly
        return {b"UIDVALIDITY": self.validity[name], b"EXISTS": len(self.folders[name])}

    def search(self, criteria):
        box = self.folders[self.selected]
        if criteria == "ALL":
            return list(box)
        assert criteria[:2] == ["HEADER", "Message-ID"]
        return [u for u, r in box.items() if f"Message-ID: {criteria[2]}".encode() in r]

    def fetch(self, uids, items):
        self.log.append(("FETCH", tuple(uids), tuple(items)))
        assert all(i.startswith("BODY.PEEK[") for i in items), "must not set \\Seen"
        box = self.folders[self.selected]
        out = {}
        for u in uids:
            if items == ["BODY.PEEK[]"]:
                out[u] = {b"BODY[]": box[u]}
            else:
                head = b"".join(l + b"\r\n" for l in box[u].split(b"\r\n") if l.startswith(b"Message-ID")) + b"\r\n"
                out[u] = {b"BODY[HEADER.FIELDS (MESSAGE-ID)]": head}
        return out

    def move(self, uids, dest):
        assert not self.readonly
        self.log.append(("MOVE", self.selected, tuple(uids), dest))
        for u in uids:
            self.next_uid += 1
            self.folders[dest][self.next_uid] = self.folders[self.selected].pop(u)

    def logout(self):
        pass


@pytest.fixture
def proton():
    fake = FakeIMAP()
    acct = ProtonAccount(type="proton", name="pm", username="me@proton.me", verify=False)
    return fake, ProtonProvider(acct, client_factory=lambda a: fake)


def test_folder_mapping(proton):
    _, p = proton
    assert p.imap_folder("Receipts") == "Folders/Receipts"
    assert p.imap_folder("inbox") == "INBOX"
    assert p.imap_folder("Labels/Work") == "Labels/Work"


def test_fetch_newest_first_peek_and_dedupe(proton):
    fake, p = proton
    msgs = p.fetch_inbox(10, skip=lambda key: key == "<m2@x>")
    assert [m.subject for m in msgs] == ["new", "old"]
    assert fake.readonly is True
    new, old = msgs
    assert old.key == "<m1@x>" and old.id == "100:1" and old.text_body.strip() == "Body 1"
    assert new.key.startswith("sha256:")  # no Message-ID
    # the skipped message's body was never downloaded
    body_fetches = [e[1] for e in fake.log if e[0] == "FETCH" and e[2] == ("BODY.PEEK[]",)]
    assert 2 not in sum(body_fetches, ())


def test_fetch_limit(proton):
    _, p = proton
    assert [m.subject for m in p.fetch_inbox(1)] == ["new"]


def test_ensure_folders(proton):
    fake, p = proton
    p.ensure_folders(["Receipts", "Newsletters", "INBOX"])
    assert [e for e in fake.log if e[0] == "CREATE"] == [("CREATE", "Folders/Newsletters")]


def test_move_and_move_back_by_message_id(proton):
    fake, p = proton
    msgs = p.fetch_inbox(10)
    old = next(m for m in msgs if m.key == "<m1@x>")
    p.move([old.ref()], "Receipts")
    assert ("MOVE", "INBOX", (1,), "Folders/Receipts") in fake.log
    assert len(fake.folders["Folders/Receipts"]) == 1

    # undo: the UID from INBOX is stale in the target folder, so it's found by Message-ID
    p.move([MessageRef(id=old.id, key=old.key, folder="Receipts")], "INBOX")
    assert fake.log[-1][0] == "MOVE" and fake.log[-1][1] == "Folders/Receipts" and fake.log[-1][3] == "INBOX"
    assert any(b"<m1@x>" in r for r in fake.folders["INBOX"].values())


def test_move_with_changed_uidvalidity_searches(proton):
    fake, p = proton
    p.fetch_inbox(10)
    fake.validity["INBOX"] = 555  # Bridge resync
    p.move([MessageRef(id="100:1", key="<m1@x>", folder="INBOX")], "Receipts")
    assert ("MOVE", "INBOX", (1,), "Folders/Receipts") in fake.log
