"""Classify inbox mail with Cloudflare Clef and file it into folders."""

import os

# Reduce CUDA memory fragmentation; must be set before torch initialises CUDA.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

INBOX = "INBOX"


def is_inbox(folder: str) -> bool:
    return folder.strip().upper() == INBOX
