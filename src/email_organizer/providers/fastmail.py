"""Fastmail via JMAP (RFC 8620 / 8621)."""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Sequence
from typing import Any

import httpx

from .. import INBOX, is_inbox
from ..config import FastmailAccount
from ..message import Message, MessageRef, html_to_text

log = logging.getLogger(__name__)

USING = ["urn:ietf:params:jmap:core", "urn:ietf:params:jmap:mail"]
MAIL_CAP = "urn:ietf:params:jmap:mail"
QUERY_PAGE = 100
SET_CHUNK = 50
MAX_BODY_BYTES = 64 * 1024

EMAIL_PROPERTIES = [
    "id", "messageId", "mailboxIds", "from", "to", "subject", "receivedAt",
    # RFC 8621 §4.1.2: RFC 2369 List-* headers only parse asURLs (List-Id allows asText).
    "header:List-Id:asText", "header:List-Unsubscribe:asURLs",
    "textBody", "bodyValues",
]


class JMAPError(RuntimeError):
    pass


def _addrs(value: list[dict[str, Any]] | None) -> str:
    out = []
    for a in value or []:
        name, email = a.get("name"), a.get("email", "")
        out.append(f"{name} <{email}>" if name else email)
    return ", ".join(out)


class FastmailProvider:
    def __init__(self, account: FastmailAccount, client: httpx.Client | None = None):
        self.name = account.name
        self.account = account
        self.client = client or httpx.Client(timeout=60.0)
        self.client.headers["Authorization"] = f"Bearer {account.token()}"
        self._api_url: str | None = None
        self._account_id: str | None = None
        self._mailboxes: dict[str, str] | None = None  # path -> id
        self._inbox_id: str | None = None

    # plumbing ------------------------------------------------------------
    def _session(self) -> None:
        if self._api_url:
            return
        r = self.client.get(self.account.session_url)
        r.raise_for_status()
        s = r.json()
        self._api_url = s["apiUrl"]
        self._account_id = s["primaryAccounts"][MAIL_CAP]

    def call(self, *method_calls: tuple[str, dict[str, Any]]) -> list[Any]:
        """Send method calls in one request; return the responses' argument dicts."""
        self._session()
        body = {
            "using": USING,
            "methodCalls": [[name, {"accountId": self._account_id, **args}, f"c{i}"]
                            for i, (name, args) in enumerate(method_calls)],
        }
        r = self.client.post(self._api_url, json=body)
        r.raise_for_status()
        results = []
        for name, args, _ in r.json()["methodResponses"]:
            if name == "error":
                detail = {k: v for k, v in args.items() if k != "type"}
                raise JMAPError(f"{args.get('type')}: {detail or 'no details'}")
            results.append(args)
        return results

    # mailboxes -------------------------------------------------------------
    def _load_mailboxes(self) -> dict[str, str]:
        if self._mailboxes is not None:
            return self._mailboxes
        (res,) = self.call(("Mailbox/get", {"ids": None, "properties": ["id", "name", "parentId", "role"]}))
        by_id = {m["id"]: m for m in res["list"]}

        def path(m: dict[str, Any]) -> str:
            parts = [m["name"]]
            while m.get("parentId") and m["parentId"] in by_id:
                m = by_id[m["parentId"]]
                parts.append(m["name"])
            return "/".join(reversed(parts))

        self._mailboxes = {path(m): m["id"] for m in by_id.values()}
        inbox = [m["id"] for m in by_id.values() if m.get("role") == "inbox"]
        if not inbox:
            raise JMAPError("no mailbox with role 'inbox'")
        self._inbox_id = inbox[0]
        return self._mailboxes

    def _mailbox_id(self, folder: str) -> str:
        boxes = self._load_mailboxes()
        if is_inbox(folder):
            assert self._inbox_id
            return self._inbox_id
        if folder not in boxes:
            raise JMAPError(f"mailbox {folder!r} does not exist (run ensure_folders first)")
        return boxes[folder]

    def ensure_folders(self, names: Iterable[str]) -> None:
        boxes = self._load_mailboxes()
        for name in sorted(set(names)):
            if is_inbox(name) or name in boxes:
                continue
            parent_id = None
            parts = name.split("/")
            for depth in range(1, len(parts) + 1):
                sub = "/".join(parts[:depth])
                if sub in boxes:
                    parent_id = boxes[sub]
                    continue
                (res,) = self.call(("Mailbox/set", {"create": {"new": {"name": parts[depth - 1], "parentId": parent_id}}}))
                if "new" not in (res.get("created") or {}):
                    raise JMAPError(f"could not create mailbox {sub!r}: {res.get('notCreated')}")
                parent_id = res["created"]["new"]["id"]
                boxes[sub] = parent_id
                log.info("created mailbox %s", sub)

    # messages ----------------------------------------------------------------
    def fetch_inbox(self, limit: int, skip: Callable[[str], bool] = lambda key: False) -> list[Message]:
        self._load_mailboxes()
        wanted: list[str] = []
        position = 0
        while len(wanted) < limit:
            (res,) = self.call(("Email/query", {
                "filter": {"inMailbox": self._inbox_id},
                "sort": [{"property": "receivedAt", "isAscending": False}],
                "position": position,
                "limit": QUERY_PAGE,
            }))
            ids = res["ids"]
            wanted.extend(i for i in ids if not skip(i))
            position += len(ids)
            if len(ids) < QUERY_PAGE:
                break
        wanted = wanted[:limit]

        messages: list[Message] = []
        for i in range(0, len(wanted), SET_CHUNK):
            (res,) = self.call(("Email/get", {
                "ids": wanted[i : i + SET_CHUNK],
                "properties": EMAIL_PROPERTIES,
                "fetchTextBodyValues": True,
                "maxBodyValueBytes": MAX_BODY_BYTES,
            }))
            messages.extend(self._to_message(e) for e in res["list"])
        return messages

    def _to_message(self, e: dict[str, Any]) -> Message:
        values = e.get("bodyValues") or {}
        chunks = []
        for part in e.get("textBody") or []:
            text = values.get(part.get("partId"), {}).get("value", "")
            chunks.append(html_to_text(text) if part.get("type") == "text/html" else text)
        return Message(
            id=e["id"],
            key=e["id"],
            account=self.name,
            mailbox=INBOX,
            sender=_addrs(e.get("from")),
            to=_addrs(e.get("to")),
            subject=e.get("subject") or "",
            date=e.get("receivedAt") or "",
            list_id=e.get("header:List-Id:asText"),
            list_unsubscribe=", ".join(e.get("header:List-Unsubscribe:asURLs") or []) or None,
            text_body="\n".join(chunks),
        )

    def move(self, refs: Sequence[MessageRef], folder: str) -> None:
        target = self._mailbox_id(folder)
        for i in range(0, len(refs), SET_CHUNK):
            chunk = refs[i : i + SET_CHUNK]
            (res,) = self.call(("Email/set", {
                "update": {r.id: {"mailboxIds": {target: True}} for r in chunk},
            }))
            failed = res.get("notUpdated") or {}
            if failed:
                raise JMAPError(f"Email/set failed for {len(failed)} message(s): {failed}")

    def close(self) -> None:
        self.client.close()
