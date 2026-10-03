from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from typing import Protocol

from ..message import Message, MessageRef


class MailProvider(Protocol):
    name: str

    def fetch_inbox(self, limit: int, skip: Callable[[str], bool] = lambda key: False) -> list[Message]:
        """Newest-first inbox messages, up to `limit` whose key is not skipped."""
        ...

    def ensure_folders(self, names: Iterable[str]) -> None:
        """Create any missing target folders (config `folder` values)."""
        ...

    def move(self, refs: Sequence[MessageRef], folder: str) -> None:
        """Move messages into `folder` (config name; "INBOX" = inbox), removing them from ref.folder."""
        ...

    def close(self) -> None: ...
