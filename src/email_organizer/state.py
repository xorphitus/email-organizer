"""SQLite state: processed messages and a per-run decision log (for undo)."""

from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id     TEXT PRIMARY KEY,
    started_at TEXT NOT NULL,
    dry_run    INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS processed (
    account      TEXT NOT NULL,
    msg_key      TEXT NOT NULL,
    run_id       TEXT NOT NULL,
    processed_at TEXT NOT NULL,
    PRIMARY KEY (account, msg_key)
);
CREATE TABLE IF NOT EXISTS decisions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id      TEXT NOT NULL REFERENCES runs(run_id),
    account     TEXT NOT NULL,
    msg_key     TEXT NOT NULL,
    msg_id      TEXT NOT NULL,
    sender      TEXT,
    subject     TEXT,
    from_folder TEXT NOT NULL,
    category    TEXT,
    prob        REAL,
    probs       TEXT,
    action      TEXT NOT NULL,   -- moved | kept | uncertain | error | (dry-run: would-move …)
    to_folder   TEXT,
    undone_at   TEXT
);
CREATE INDEX IF NOT EXISTS decisions_run ON decisions(run_id);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class DecisionRow:
    id: int
    run_id: str
    account: str
    msg_key: str
    msg_id: str
    sender: str | None
    subject: str | None
    from_folder: str
    category: str | None
    prob: float | None
    probs: dict[str, float]
    action: str
    to_folder: str | None
    undone_at: str | None


class StateDB:
    def __init__(self, path: Path | str):
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(path))
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)

    def close(self) -> None:
        self.conn.close()

    # runs ------------------------------------------------------------------
    def start_run(self, dry_run: bool) -> str:
        run_id = datetime.now().strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
        with self.conn:
            self.conn.execute(
                "INSERT INTO runs(run_id, started_at, dry_run) VALUES (?, ?, ?)",
                (run_id, _now(), int(dry_run)),
            )
        return run_id

    def get_run(self, run_id: str) -> sqlite3.Row | None:
        return self.conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()

    def recent_runs(self, limit: int = 10) -> list[sqlite3.Row]:
        return self.conn.execute(
            """SELECT r.run_id, r.started_at, r.dry_run, COUNT(d.id) AS n,
                      SUM(d.action = 'moved') AS moved, SUM(d.undone_at IS NOT NULL) AS undone
               FROM runs r LEFT JOIN decisions d ON d.run_id = r.run_id
               GROUP BY r.run_id ORDER BY r.started_at DESC LIMIT ?""",
            (limit,),
        ).fetchall()

    # processed ---------------------------------------------------------------
    def is_processed(self, account: str, msg_key: str) -> bool:
        row = self.conn.execute(
            "SELECT 1 FROM processed WHERE account = ? AND msg_key = ?", (account, msg_key)
        ).fetchone()
        return row is not None

    # decisions -------------------------------------------------------------
    def record(
        self,
        run_id: str,
        *,
        account: str,
        msg_key: str,
        msg_id: str,
        sender: str | None,
        subject: str | None,
        from_folder: str,
        category: str | None,
        prob: float | None,
        probs: dict[str, float],
        action: str,
        to_folder: str | None,
        mark_processed: bool,
    ) -> None:
        with self.conn:
            self.conn.execute(
                """INSERT INTO decisions(run_id, account, msg_key, msg_id, sender, subject,
                       from_folder, category, prob, probs, action, to_folder)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (run_id, account, msg_key, msg_id, sender, subject, from_folder,
                 category, prob, json.dumps(probs), action, to_folder),
            )
            if mark_processed:
                self.conn.execute(
                    """INSERT INTO processed(account, msg_key, run_id, processed_at)
                       VALUES (?, ?, ?, ?)
                       ON CONFLICT(account, msg_key) DO UPDATE
                       SET run_id = excluded.run_id, processed_at = excluded.processed_at""",
                    (account, msg_key, run_id, _now()),
                )

    def decisions(self, run_id: str) -> list[DecisionRow]:
        rows = self.conn.execute(
            "SELECT * FROM decisions WHERE run_id = ? ORDER BY id", (run_id,)
        ).fetchall()
        return [
            DecisionRow(**{**dict(r), "probs": json.loads(r["probs"] or "{}")}) for r in rows
        ]

    def mark_undone(self, decision_ids: list[int]) -> None:
        """Flag decisions as undone and forget the messages so a later run reconsiders them."""
        if not decision_ids:
            return
        now = _now()
        with self.conn:
            for did in decision_ids:
                row = self.conn.execute(
                    "SELECT account, msg_key FROM decisions WHERE id = ?", (did,)
                ).fetchone()
                self.conn.execute("UPDATE decisions SET undone_at = ? WHERE id = ?", (now, did))
                if row:
                    self.conn.execute(
                        "DELETE FROM processed WHERE account = ? AND msg_key = ?",
                        (row["account"], row["msg_key"]),
                    )
