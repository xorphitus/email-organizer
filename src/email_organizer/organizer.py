"""Run / undo orchestration, independent of the CLI and of real providers."""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass

from . import INBOX, is_inbox
from .classifier import Classifier, Decision
from .config import Category
from .message import Message, MessageRef
from .providers.base import MailProvider
from .state import StateDB

log = logging.getLogger(__name__)


@dataclass
class Outcome:
    message: Message
    decision: Decision
    action: str  # moved | kept | uncertain | error  (prefixed "would-" on dry runs)
    folder: str | None


def plan_action(decision: Decision, categories: Mapping[str, Category]) -> tuple[str, str | None]:
    if decision.error:
        return "error", None
    if decision.category is None:
        return "uncertain", None
    folder = categories[decision.category].folder
    if is_inbox(folder):
        return "kept", None
    return "moved", folder


def run_account(
    provider: MailProvider,
    classifier_factory: Callable[[], Classifier],
    state: StateDB,
    categories: Mapping[str, Category],
    run_id: str,
    *,
    limit: int,
    dry_run: bool,
    reprocess: bool = False,
) -> list[Outcome]:
    account = provider.name
    skip = (lambda key: False) if reprocess else (lambda key: state.is_processed(account, key))
    messages = provider.fetch_inbox(limit, skip=skip)
    if not messages:
        return []
    decisions = classifier_factory().classify(messages)

    outcomes = []
    for msg, dec in zip(messages, decisions, strict=True):
        action, folder = plan_action(dec, categories)
        outcomes.append(Outcome(msg, dec, action, folder))

    if not dry_run:
        targets = {o.folder for o in outcomes if o.action == "moved"}
        if targets:
            provider.ensure_folders(targets)
        for folder in sorted(t for t in targets if t):
            group = [o for o in outcomes if o.folder == folder and o.action == "moved"]
            try:
                provider.move([o.message.ref() for o in group], folder)
            except Exception as e:  # noqa: BLE001 - record and continue with other folders
                log.error("%s: moving %d message(s) to %s failed: %s", account, len(group), folder, e)
                for o in group:
                    o.action = "error"
                    o.decision.error = f"move failed: {e}"

    for o in outcomes:
        action = f"would-{o.action}" if dry_run and o.action == "moved" else o.action
        state.record(
            run_id,
            account=account,
            msg_key=o.message.key,
            msg_id=o.message.id,
            sender=o.message.sender,
            subject=o.message.subject,
            from_folder=o.message.mailbox,
            category=o.decision.category or o.decision.top,
            prob=o.decision.prob,
            probs=o.decision.probs,
            action=action,
            to_folder=o.folder,
            # Errors are retried next run; dry runs never mark anything.
            mark_processed=not dry_run and o.action != "error",
        )
    return outcomes


def undo_run(
    state: StateDB,
    run_id: str,
    provider_for: Callable[[str], MailProvider],
) -> dict[str, int]:
    """Move every message the run moved back to its original folder. Returns count per account."""
    run = state.get_run(run_id)
    if run is None:
        raise KeyError(f"unknown run {run_id}")
    if run["dry_run"]:
        raise ValueError(f"run {run_id} was a dry run; nothing to undo")

    pending = [d for d in state.decisions(run_id) if d.action == "moved" and d.undone_at is None]
    by_dest: dict[tuple[str, str], list] = {}
    for d in pending:
        by_dest.setdefault((d.account, d.from_folder or INBOX), []).append(d)

    counts: dict[str, int] = {}
    for (account, dest), rows in by_dest.items():
        provider = provider_for(account)
        refs = [MessageRef(id=d.msg_id, key=d.msg_key, folder=d.to_folder or INBOX) for d in rows]
        provider.move(refs, dest)
        state.mark_undone([d.id for d in rows])
        counts[account] = counts.get(account, 0) + len(rows)
    return counts
