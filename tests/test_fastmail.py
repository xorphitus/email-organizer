import json

import httpx
import pytest

from email_organizer.config import FastmailAccount
from email_organizer.message import MessageRef
from email_organizer.providers.fastmail import USING, FastmailProvider, JMAPError

SESSION = {
    "apiUrl": "https://api.fastmail.com/jmap/api/",
    "primaryAccounts": {"urn:ietf:params:jmap:mail": "u123"},
}
MAILBOXES = [
    {"id": "mb-inbox", "name": "Inbox", "parentId": None, "role": "inbox"},
    {"id": "mb-arch", "name": "Archive", "parentId": None, "role": "archive"},
    {"id": "mb-rcpt", "name": "Receipts", "parentId": None, "role": None},
    {"id": "mb-sub", "name": "Shops", "parentId": "mb-rcpt", "role": None},
]
EMAILS = {
    "e1": {
        "id": "e1", "messageId": ["<a@x>"], "mailboxIds": {"mb-inbox": True},
        "from": [{"name": "Shop", "email": "shop@x.com"}], "to": [{"name": None, "email": "me@x.com"}],
        "subject": "Receipt", "receivedAt": "2026-10-01T10:00:00Z",
        "header:List-Id:asText": None, "header:List-Unsubscribe:asText": "<mailto:u@x.com>",
        "textBody": [{"partId": "1", "type": "text/plain"}], "bodyValues": {"1": {"value": "Total $5"}},
    },
    "e2": {
        "id": "e2", "messageId": ["<b@x>"], "mailboxIds": {"mb-inbox": True},
        "from": [{"email": "news@x.com"}], "to": [], "subject": "News", "receivedAt": "2026-09-30T10:00:00Z",
        "header:List-Id:asText": "<news.x.com>", "header:List-Unsubscribe:asText": None,
        "textBody": [{"partId": "2", "type": "text/html"}], "bodyValues": {"2": {"value": "<p>Hello <b>you</b></p>"}},
    },
}


class FakeJMAP:
    def __init__(self, inbox_ids=("e1", "e2")):
        self.inbox_ids = list(inbox_ids)
        self.requests: list[dict] = []
        self.created = 0
        self.set_fail: set[str] = set()

    def __call__(self, request: httpx.Request) -> httpx.Response:
        assert request.headers["Authorization"] == "Bearer tok"
        if request.method == "GET":
            assert str(request.url) == "https://api.fastmail.com/jmap/session"
            return httpx.Response(200, json=SESSION)
        body = json.loads(request.content)
        self.requests.append(body)
        assert body["using"] == USING
        responses = []
        for name, args, cid in body["methodCalls"]:
            assert args["accountId"] == "u123"
            responses.append([name, self.handle(name, args), cid])
        return httpx.Response(200, json={"methodResponses": responses, "sessionState": "s"})

    def handle(self, name, args):
        if name == "Mailbox/get":
            return {"list": MAILBOXES}
        if name == "Mailbox/set":
            created = {}
            for cid, spec in args["create"].items():
                self.created += 1
                created[cid] = {"id": f"mb-new{self.created}"}
            return {"created": created}
        if name == "Email/query":
            pos, lim = args["position"], args["limit"]
            return {"ids": self.inbox_ids[pos : pos + lim]}
        if name == "Email/get":
            return {"list": [EMAILS[i] for i in args["ids"]]}
        if name == "Email/set":
            nu = {i: {"type": "notFound"} for i in args["update"] if i in self.set_fail}
            return {"updated": {i: None for i in args["update"] if i not in nu}, "notUpdated": nu or None}
        raise AssertionError(name)

    def calls(self, method):
        return [args for req in self.requests for name, args, _ in req["methodCalls"] if name == method]


@pytest.fixture
def jmap(monkeypatch):
    monkeypatch.setenv("FM_TEST_TOKEN", "tok")
    fake = FakeJMAP()
    client = httpx.Client(transport=httpx.MockTransport(fake))
    provider = FastmailProvider(FastmailAccount(type="fastmail", name="fm", token_env="FM_TEST_TOKEN"), client=client)
    return fake, provider


def test_fetch_inbox(jmap):
    fake, p = jmap
    msgs = p.fetch_inbox(10, skip=lambda key: False)
    q = fake.calls("Email/query")[0]
    assert q["filter"] == {"inMailbox": "mb-inbox"}
    assert q["sort"] == [{"property": "receivedAt", "isAscending": False}]
    g = fake.calls("Email/get")[0]
    assert g["ids"] == ["e1", "e2"]
    assert g["fetchTextBodyValues"] is True and g["maxBodyValueBytes"] > 0
    assert "header:List-Id:asText" in g["properties"] and "header:List-Unsubscribe:asText" in g["properties"]

    m1, m2 = msgs
    assert (m1.id, m1.key, m1.sender, m1.to, m1.subject) == ("e1", "e1", "Shop <shop@x.com>", "me@x.com", "Receipt")
    assert m1.text_body == "Total $5" and m1.is_mailing_list
    assert m2.text_body == "Hello you" and m2.list_id == "<news.x.com>"


def test_fetch_skips_processed_and_limits(jmap):
    fake, p = jmap
    msgs = p.fetch_inbox(10, skip=lambda key: key == "e1")
    assert [m.id for m in msgs] == ["e2"]
    assert fake.calls("Email/get")[0]["ids"] == ["e2"]
    assert [m.id for m in p.fetch_inbox(1)] == ["e1"]


def test_fetch_paginates(monkeypatch, jmap):
    fake, p = jmap
    monkeypatch.setattr("email_organizer.providers.fastmail.QUERY_PAGE", 1)
    msgs = p.fetch_inbox(10)
    assert [q["position"] for q in fake.calls("Email/query")] == [0, 1, 2]
    assert len(msgs) == 2


def test_ensure_folders_creates_missing_only(jmap):
    fake, p = jmap
    p.ensure_folders(["Receipts", "Receipts/Shops", "INBOX", "Newsletters", "A/B"])
    creates = [c["create"]["new"] for c in fake.calls("Mailbox/set")]
    assert creates == [
        {"name": "A", "parentId": None},
        {"name": "B", "parentId": "mb-new1"},
        {"name": "Newsletters", "parentId": None},
    ]


def test_move_replaces_mailbox_membership_in_chunks(monkeypatch, jmap):
    fake, p = jmap
    monkeypatch.setattr("email_organizer.providers.fastmail.SET_CHUNK", 1)
    p.move([MessageRef("e1", "e1", "INBOX"), MessageRef("e2", "e2", "INBOX")], "Receipts")
    sets = fake.calls("Email/set")
    assert [s["update"] for s in sets] == [
        {"e1": {"mailboxIds": {"mb-rcpt": True}}},
        {"e2": {"mailboxIds": {"mb-rcpt": True}}},
    ]
    p.move([MessageRef("e1", "e1", "Receipts")], "INBOX")
    assert fake.calls("Email/set")[-1]["update"] == {"e1": {"mailboxIds": {"mb-inbox": True}}}


def test_move_errors(jmap):
    fake, p = jmap
    with pytest.raises(JMAPError, match="does not exist"):
        p.move([MessageRef("e1", "e1", "INBOX")], "Nope")
    fake.set_fail = {"e1"}
    with pytest.raises(JMAPError, match="Email/set failed"):
        p.move([MessageRef("e1", "e1", "INBOX")], "Receipts")
