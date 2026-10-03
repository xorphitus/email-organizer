"""ProtonMail via Proton Mail Bridge (local IMAP, STARTTLS)."""

from __future__ import annotations

import hashlib
import hmac
import logging
import ssl
from collections.abc import Callable, Iterable, Sequence
from typing import Any

from .. import INBOX, is_inbox
from ..config import ProtonAccount
from ..message import Message, MessageRef, extract_text, message_key, parse_rfc822

log = logging.getLogger(__name__)

FETCH_CHUNK = 50
HEADER_ITEM = b"BODY[HEADER.FIELDS (MESSAGE-ID)]"
BODY_ITEM = b"BODY[]"


def _tls_context(account: ProtonAccount) -> ssl.SSLContext:
    if account.cert_fingerprint or not account.verify:
        # Bridge uses a self-signed cert: either pin it (checked after the handshake)
        # or, for localhost only, skip verification entirely.
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        return ctx
    return ssl.create_default_context(cafile=str(account.ca_file) if account.ca_file else None)


def _normalise_fp(fp: str) -> str:
    return fp.replace(":", "").replace(" ", "").lower()


def _default_client_factory(account: ProtonAccount) -> Any:
    from imapclient import IMAPClient

    client = IMAPClient(account.host, port=account.port, ssl=False, timeout=60)
    client.starttls(_tls_context(account))
    if account.cert_fingerprint:
        der = client.socket().getpeercert(binary_form=True)
        actual = hashlib.sha256(der).hexdigest()
        if not hmac.compare_digest(actual, _normalise_fp(account.cert_fingerprint)):
            client.shutdown()
            raise ssl.SSLError(f"Bridge certificate fingerprint mismatch: got sha256 {actual}")
    client.login(account.username, account.password())
    return client


class ProtonProvider:
    def __init__(self, account: ProtonAccount, client_factory: Callable[[ProtonAccount], Any] = _default_client_factory):
        self.name = account.name
        self.account = account
        self.client = client_factory(account)
        self._uidvalidity: int | None = None

    def imap_folder(self, folder: str) -> str:
        if is_inbox(folder):
            return INBOX
        if folder.startswith(("Folders/", "Labels/")):
            return folder
        return self.account.folder_prefix + folder

    # folders ---------------------------------------------------------------
    def ensure_folders(self, names: Iterable[str]) -> None:
        existing = {name for _flags, _delim, name in self.client.list_folders()}
        for name in sorted({self.imap_folder(n) for n in names}):
            if name not in existing:
                self.client.create_folder(name)
                log.info("created folder %s", name)

    # fetch -----------------------------------------------------------------
    def fetch_inbox(self, limit: int, skip: Callable[[str], bool] = lambda key: False) -> list[Message]:
        info = self.client.select_folder(INBOX, readonly=True)
        self._uidvalidity = int(info[b"UIDVALIDITY"])
        uids = sorted(self.client.search("ALL"), reverse=True)  # newest first

        wanted: list[int] = []
        for i in range(0, len(uids), FETCH_CHUNK * 4):
            if len(wanted) >= limit:
                break
            chunk = uids[i : i + FETCH_CHUNK * 4]
            heads = self.client.fetch(chunk, ["BODY.PEEK[HEADER.FIELDS (MESSAGE-ID)]"])
            for uid in chunk:
                head = heads.get(uid, {}).get(HEADER_ITEM)
                key = message_key(parse_rfc822(head)) if head else None
                # A message without Message-ID gets a key from its full headers; that
                # is checked after the body fetch below.
                if key and not key.startswith("sha256:") and skip(key):
                    continue
                wanted.append(uid)
                if len(wanted) >= limit:
                    break

        messages: list[Message] = []
        for i in range(0, len(wanted), FETCH_CHUNK):
            chunk = wanted[i : i + FETCH_CHUNK]
            data = self.client.fetch(chunk, ["BODY.PEEK[]"])
            for uid in chunk:
                raw = data.get(uid, {}).get(BODY_ITEM)
                if raw is None:
                    continue
                msg = self._to_message(uid, raw)
                if not skip(msg.key):
                    messages.append(msg)
        return messages

    def _to_message(self, uid: int, raw: bytes) -> Message:
        em = parse_rfc822(raw)
        return Message(
            id=f"{self._uidvalidity}:{uid}",
            key=message_key(em),
            account=self.name,
            mailbox=INBOX,
            sender=str(em.get("From", "")),
            to=str(em.get("To", "")),
            subject=str(em.get("Subject", "")),
            date=str(em.get("Date", "")),
            list_id=str(em["List-Id"]) if em.get("List-Id") else None,
            list_unsubscribe=str(em["List-Unsubscribe"]) if em.get("List-Unsubscribe") else None,
            text_body=extract_text(em),
        )

    # move ------------------------------------------------------------------
    def _resolve_uids(self, refs: Sequence[MessageRef], folder: str, uidvalidity: int) -> list[int]:
        """UIDs in the selected folder: reuse the fetched UID when still valid, else search by Message-ID."""
        uids = []
        for r in refs:
            validity, _, uid = r.id.partition(":")
            if is_inbox(folder) and uid and validity == str(uidvalidity):
                uids.append(int(uid))
                continue
            if r.key.startswith("sha256:"):
                log.warning("cannot locate message without Message-ID in %s: %s", folder, r.key)
                continue
            found = self.client.search(["HEADER", "Message-ID", r.key])
            if not found:
                log.warning("message %s not found in %s", r.key, folder)
            uids.extend(found)
        return uids

    def move(self, refs: Sequence[MessageRef], folder: str) -> None:
        dest = self.imap_folder(folder)
        by_source: dict[str, list[MessageRef]] = {}
        for r in refs:
            by_source.setdefault(self.imap_folder(r.folder), []).append(r)
        for source, group in by_source.items():
            info = self.client.select_folder(source)
            uids = self._resolve_uids(group, source, int(info[b"UIDVALIDITY"]))
            for i in range(0, len(uids), FETCH_CHUNK):
                self.client.move(uids[i : i + FETCH_CHUNK], dest)

    def close(self) -> None:
        try:
            self.client.logout()
        except Exception:  # noqa: BLE001 - best effort on shutdown
            pass
