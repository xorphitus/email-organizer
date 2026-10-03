"""Classify inbox mail with Cloudflare Clef and file it into folders."""

INBOX = "INBOX"


def is_inbox(folder: str) -> bool:
    return folder.strip().upper() == INBOX
