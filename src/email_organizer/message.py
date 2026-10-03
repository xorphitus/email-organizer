"""Provider-neutral message model and body text helpers."""

from __future__ import annotations

import email
import email.policy
import hashlib
import re
from dataclasses import dataclass
from email.message import EmailMessage
from html.parser import HTMLParser


@dataclass(frozen=True)
class MessageRef:
    """Enough to locate a message on its server for a move."""

    id: str  # provider id (JMAP email id; IMAP "uidvalidity:uid", may be stale)
    key: str  # stable dedupe key (JMAP id; Message-ID for IMAP)
    folder: str  # folder the message currently lives in


@dataclass
class Message:
    id: str
    key: str
    account: str
    mailbox: str
    sender: str
    to: str
    subject: str
    date: str
    list_id: str | None = None
    list_unsubscribe: str | None = None
    text_body: str = ""

    @property
    def is_mailing_list(self) -> bool:
        return bool(self.list_id or self.list_unsubscribe)

    def ref(self) -> MessageRef:
        return MessageRef(id=self.id, key=self.key, folder=self.mailbox)


# --- HTML → text -----------------------------------------------------------

_BLOCK_TAGS = {
    "p", "div", "br", "tr", "li", "h1", "h2", "h3", "h4", "h5", "h6",
    "table", "section", "article", "header", "footer", "blockquote", "hr",
}
_SKIP_TAGS = {"script", "style", "head", "title", "noscript"}


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in _SKIP_TAGS:
            self._skip += 1
        elif tag in _BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in _SKIP_TAGS:
            self._skip = max(0, self._skip - 1)
        elif tag in _BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data):
        if not self._skip:
            self.parts.append(data)


def html_to_text(html: str) -> str:
    parser = _TextExtractor()
    parser.feed(html)
    parser.close()
    text = "".join(parser.parts)
    text = re.sub(r"[ \t\r\f\v ]+", " ", text)
    return re.sub(r"\n\s*\n+", "\n\n", text).strip()


# --- body cleanup ----------------------------------------------------------

_REPLY_HEADER = re.compile(
    r"^(On .{1,200} wrote:|-{2,}\s*Original Message\s*-{2,}|_{10,}|From: .+)$",
    re.IGNORECASE,
)
_SIG_DELIM = re.compile(r"^(-- ?|—)$")


def clean_body(text: str, max_chars: int | None = None) -> str:
    """Drop quoted replies and signatures, collapse blank lines, truncate."""
    out: list[str] = []
    for line in text.replace("\r\n", "\n").split("\n"):
        stripped = line.rstrip()
        # Only cut once some content was seen, so a forwarded mail isn't emptied.
        has_content = any(o.strip() for o in out)
        if has_content and (_SIG_DELIM.match(stripped) or _REPLY_HEADER.match(stripped.strip())):
            break
        if stripped.lstrip().startswith(">"):
            continue
        out.append(stripped)
    body = re.sub(r"\n{3,}", "\n\n", "\n".join(out)).strip()
    if max_chars is not None and len(body) > max_chars:
        body = body[:max_chars].rstrip() + " …"
    return body


# --- RFC 822 parsing (IMAP providers) ---------------------------------------


def parse_rfc822(raw: bytes) -> EmailMessage:
    return email.message_from_bytes(raw, policy=email.policy.default)  # type: ignore[return-value]


def extract_text(msg: EmailMessage) -> str:
    """Prefer text/plain; fall back to stripped text/html."""
    try:
        part = msg.get_body(preferencelist=("plain",))
        if part is not None:
            return part.get_content()
        part = msg.get_body(preferencelist=("html",))
        if part is not None:
            return html_to_text(part.get_content())
    except (KeyError, LookupError, UnicodeDecodeError):
        pass
    return ""


def message_key(msg: EmailMessage) -> str:
    mid = (msg.get("Message-ID") or "").strip()
    if mid:
        return mid
    basis = "|".join(str(msg.get(h, "")) for h in ("From", "To", "Date", "Subject"))
    return "sha256:" + hashlib.sha256(basis.encode()).hexdigest()
