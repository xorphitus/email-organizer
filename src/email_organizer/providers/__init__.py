from __future__ import annotations

from ..config import FastmailAccount, ProtonAccount
from .base import MailProvider


def make_provider(account: FastmailAccount | ProtonAccount) -> MailProvider:
    if isinstance(account, FastmailAccount):
        from .fastmail import FastmailProvider

        return FastmailProvider(account)
    from .proton import ProtonProvider

    return ProtonProvider(account)


__all__ = ["MailProvider", "make_provider"]
