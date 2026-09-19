"""Persistent, timezone-aware reminder scheduling for BracketBot.

The scheduler owns no robot hardware.  It stores typed reminder records in a
small SQLite database and invokes one delivery callback when a reminder is
due.  Speaker and LED ownership remain with the voice assistant.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sqlite3
import threading
import time
from typing import Callable
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


REMINDER_TOOL_IDS = frozenset(
    {"set-reminder", "cancel-reminder", "list-reminders"}
)
DEFAULT_TIMEZONE = "UTC"


def default_reminder_db_path() -> Path:
    configured = os.environ.get("BAYMAX_REMINDER_DB_PATH", "").strip()
    if configured:
        return Path(configured).expanduser()
    return Path.home() / ".local" / "state" / "bracketbot" / "reminders.sqlite3"


def default_timezone_name() -> str:
    configured = os.environ.get("BAYMAX_TIMEZONE", "").strip()
    if configured:
        return configured
    timezone_file = Path("/etc/timezone")
    try:
        name = timezone_file.read_text().strip()
    except OSError:
        name = ""
    return name or DEFAULT_TIMEZONE


def _zone(name: str) -> ZoneInfo:
    try:
        return ZoneInfo(name)
    except ZoneInfoNotFoundError as exc:
        raise ValueError(f"Unknown timezone: {name}") from exc


def _utc_text(timestamp: float) -> str:
    return (
        datetime.fromtimestamp(timestamp, timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z")
    )


@dataclass(frozen=True)
class Reminder:
    id: int
    message: str | None
    due_at: float
    timezone: str
    source: str
    status: str
    created_at: float

    def public(self) -> dict[str, object]:
        local_due = datetime.fromtimestamp(self.due_at, _zone(self.timezone))
        return {
            "id": self.id,
            "message": self.message,
            "due_at_utc": _utc_text(self.due_at),
            "due_at_local": local_due.isoformat(timespec="seconds"),
            "timezone": self.timezone,
            "source": self.source,
            "status": self.status,
            "created_at_utc": _utc_text(self.created_at),
        }


class PersistentReminderScheduler:
    """SQLite-backed scheduler with a single cooperative worker thread."""

    def __init__(
        self,
        path: str | Path,
        deliver: Callable[[Reminder], None],
        *,
        timezone_name: str | None = None,
        clock: Callable[[], float] = time.time,
        poll_interval: float = 1.0,
    ):
        self.path = Path(path).expanduser()
        self.timezone_name = timezone_name or default_timezone_name()
        _zone(self.timezone_name)
        self._deliver = deliver
        self._clock = clock
        self._poll_interval = max(0.05, float(poll_interval))
        self._db_lock = threading.RLock()
        self._wake = threading.Condition()
        self._stop = threading.Event()
        self._closed = False
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(
            self.path,
            timeout=5.0,
            check_same_thread=False,
        )
        self._connection.row_factory = sqlite3.Row
        self._initialize()
        self._thread = threading.Thread(
            target=self._run,
            name="persistent-reminders",
            daemon=True,
        )
        self._thread.start()

    def _initialize(self) -> None:
        with self._db_lock, self._connection:
            self._connection.execute("PRAGMA journal_mode=WAL")
            self._connection.execute("PRAGMA foreign_keys=ON")
            self._connection.execute(
                """
                CREATE TABLE IF NOT EXISTS reminders (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    message TEXT,
                    due_at REAL NOT NULL,
                    timezone TEXT NOT NULL,
                    source TEXT NOT NULL,
                    status TEXT NOT NULL CHECK (
                        status IN ('scheduled', 'delivering', 'fired', 'cancelled')
                    ),
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    fired_at REAL,
                    cancelled_at REAL
                )
                """
            )
            self._connection.execute(
                """
                CREATE INDEX IF NOT EXISTS reminder_due_idx
                ON reminders(status, due_at)
                """
            )
            self._connection.execute(
                """
                CREATE TABLE IF NOT EXISTS reminder_audit (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    reminder_id INTEGER NOT NULL,
                    event TEXT NOT NULL,
                    occurred_at REAL NOT NULL,
                    details TEXT NOT NULL,
                    FOREIGN KEY(reminder_id) REFERENCES reminders(id)
                )
                """
            )
            # A process may have stopped after claiming a reminder but before
            # recording delivery. Requeue it for at-least-once delivery.
            interrupted = self._connection.execute(
                "SELECT id FROM reminders WHERE status = 'delivering'"
            ).fetchall()
            now = self._clock()
            self._connection.execute(
                """
                UPDATE reminders
                SET status = 'scheduled', updated_at = ?
                WHERE status = 'delivering'
                """,
                (now,),
            )
            for row in interrupted:
                self._audit_locked(
                    int(row["id"]),
                    "recovered-after-restart",
                    now,
                    {},
                )

    @staticmethod
    def _record(row: sqlite3.Row) -> Reminder:
        return Reminder(
            id=int(row["id"]),
            message=str(row["message"]) if row["message"] is not None else None,
            due_at=float(row["due_at"]),
            timezone=str(row["timezone"]),
            source=str(row["source"]),
            status=str(row["status"]),
            created_at=float(row["created_at"]),
        )

    def _audit_locked(
        self,
        reminder_id: int,
        event: str,
        occurred_at: float,
        details: dict[str, object],
    ) -> None:
        self._connection.execute(
            """
            INSERT INTO reminder_audit(reminder_id, event, occurred_at, details)
            VALUES (?, ?, ?, ?)
            """,
            (
                reminder_id,
                event,
                occurred_at,
                json.dumps(details, ensure_ascii=False, separators=(",", ":")),
            ),
        )

    def schedule_after(
        self,
        delay_seconds: float,
        message: str | None,
        *,
        source: str = "internal",
        timezone_name: str | None = None,
    ) -> Reminder:
        delay = float(delay_seconds)
        if delay <= 0:
            raise ValueError("Reminder delay must be greater than zero")
        return self.schedule_at(
            self._clock() + delay,
            message,
            source=source,
            timezone_name=timezone_name,
        )

    def schedule_at(
        self,
        due_at: str | float | datetime,
        message: str | None,
        *,
        source: str = "internal",
        timezone_name: str | None = None,
    ) -> Reminder:
        zone_name = timezone_name or self.timezone_name
        zone = _zone(zone_name)
        if isinstance(due_at, datetime):
            due = due_at
        elif isinstance(due_at, str):
            try:
                due = datetime.fromisoformat(due_at.strip().replace("Z", "+00:00"))
            except ValueError as exc:
                raise ValueError("due_at must be an ISO 8601 timestamp") from exc
        else:
            timestamp = float(due_at)
            if not math.isfinite(timestamp):
                raise ValueError("due_at must be a finite timestamp")
            due = datetime.fromtimestamp(timestamp, timezone.utc)
        if due.tzinfo is None:
            due = due.replace(tzinfo=zone)
        due_timestamp = due.astimezone(timezone.utc).timestamp()
        normalized_message = str(message).strip() if message is not None else None
        if normalized_message == "":
            normalized_message = None
        if normalized_message is not None and len(normalized_message) > 500:
            raise ValueError("Reminder messages are limited to 500 characters")
        normalized_source = str(source).strip() or "internal"
        if len(normalized_source) > 80:
            raise ValueError("Reminder sources are limited to 80 characters")
        now = self._clock()
        with self._db_lock, self._connection:
            cursor = self._connection.execute(
                """
                INSERT INTO reminders(
                    message, due_at, timezone, source, status, created_at, updated_at
                ) VALUES (?, ?, ?, ?, 'scheduled', ?, ?)
                """,
                (
                    normalized_message,
                    due_timestamp,
                    zone_name,
                    normalized_source,
                    now,
                    now,
                ),
            )
            reminder_id = int(cursor.lastrowid)
            self._audit_locked(
                reminder_id,
                "scheduled",
                now,
                {"due_at_utc": _utc_text(due_timestamp), "source": normalized_source},
            )
            row = self._connection.execute(
                "SELECT * FROM reminders WHERE id = ?",
                (reminder_id,),
            ).fetchone()
        with self._wake:
            self._wake.notify_all()
        return self._record(row)

    def list_pending(self) -> list[Reminder]:
        with self._db_lock:
            rows = self._connection.execute(
                """
                SELECT * FROM reminders
                WHERE status IN ('scheduled', 'delivering')
                ORDER BY due_at, id
                """
            ).fetchall()
        return [self._record(row) for row in rows]

    def cancel(self, reminder_id: int | None = None) -> list[int]:
        now = self._clock()
        with self._db_lock, self._connection:
            if reminder_id is None:
                rows = self._connection.execute(
                    "SELECT id FROM reminders WHERE status = 'scheduled' ORDER BY id"
                ).fetchall()
            else:
                rows = self._connection.execute(
                    """
                    SELECT id FROM reminders
                    WHERE id = ? AND status = 'scheduled'
                    """,
                    (int(reminder_id),),
                ).fetchall()
            ids = [int(row["id"]) for row in rows]
            for item_id in ids:
                self._connection.execute(
                    """
                    UPDATE reminders
                    SET status = 'cancelled', cancelled_at = ?, updated_at = ?
                    WHERE id = ? AND status = 'scheduled'
                    """,
                    (now, now, item_id),
                )
                self._audit_locked(item_id, "cancelled", now, {})
        if ids:
            with self._wake:
                self._wake.notify_all()
        return ids

    def audit(self, reminder_id: int | None = None, limit: int = 100) -> list[dict]:
        bounded_limit = min(1000, max(1, int(limit)))
        with self._db_lock:
            if reminder_id is None:
                rows = self._connection.execute(
                    """
                    SELECT reminder_id, event, occurred_at, details
                    FROM reminder_audit ORDER BY id DESC LIMIT ?
                    """,
                    (bounded_limit,),
                ).fetchall()
            else:
                rows = self._connection.execute(
                    """
                    SELECT reminder_id, event, occurred_at, details
                    FROM reminder_audit
                    WHERE reminder_id = ? ORDER BY id DESC LIMIT ?
                    """,
                    (int(reminder_id), bounded_limit),
                ).fetchall()
        return [
            {
                "reminder_id": int(row["reminder_id"]),
                "event": str(row["event"]),
                "occurred_at_utc": _utc_text(float(row["occurred_at"])),
                "details": json.loads(str(row["details"])),
            }
            for row in rows
        ]

    def _next_scheduled(self) -> Reminder | None:
        with self._db_lock:
            row = self._connection.execute(
                """
                SELECT * FROM reminders
                WHERE status = 'scheduled'
                ORDER BY due_at, id LIMIT 1
                """
            ).fetchone()
        return self._record(row) if row is not None else None

    def _claim(self, reminder_id: int) -> Reminder | None:
        now = self._clock()
        with self._db_lock, self._connection:
            cursor = self._connection.execute(
                """
                UPDATE reminders SET status = 'delivering', updated_at = ?
                WHERE id = ? AND status = 'scheduled'
                """,
                (now, reminder_id),
            )
            if cursor.rowcount != 1:
                return None
            self._audit_locked(reminder_id, "delivery-claimed", now, {})
            row = self._connection.execute(
                "SELECT * FROM reminders WHERE id = ?",
                (reminder_id,),
            ).fetchone()
        return self._record(row)

    def _finish_delivery(self, reminder_id: int, error: Exception | None) -> None:
        now = self._clock()
        event = "delivery-failed" if error is not None else "delivered"
        details = {"error": f"{type(error).__name__}: {error}"} if error else {}
        with self._db_lock, self._connection:
            if error is None:
                self._connection.execute(
                    """
                    UPDATE reminders
                    SET status = 'fired', fired_at = ?, updated_at = ?
                    WHERE id = ? AND status = 'delivering'
                    """,
                    (now, now, reminder_id),
                )
            else:
                # Keep a failed delivery pending. A short backoff prevents a
                # broken speaker callback from creating a tight retry loop.
                self._connection.execute(
                    """
                    UPDATE reminders
                    SET status = 'scheduled', due_at = ?, updated_at = ?
                    WHERE id = ? AND status = 'delivering'
                    """,
                    (now + 5.0, now, reminder_id),
                )
            self._audit_locked(reminder_id, event, now, details)

    def _run(self) -> None:
        while not self._stop.is_set():
            reminder = self._next_scheduled()
            if reminder is None:
                timeout = self._poll_interval
            else:
                timeout = min(
                    self._poll_interval,
                    max(0.0, reminder.due_at - self._clock()),
                )
            if timeout > 0:
                with self._wake:
                    self._wake.wait(timeout=timeout)
                continue
            claimed = self._claim(reminder.id)
            if claimed is None:
                continue
            error = None
            try:
                self._deliver(claimed)
            except Exception as exc:  # delivery errors are recorded, not fatal
                error = exc
            self._finish_delivery(claimed.id, error)

    def close(self, timeout: float = 3.0) -> None:
        if self._closed:
            return
        self._stop.set()
        with self._wake:
            self._wake.notify_all()
        self._thread.join(timeout=timeout)
        if not self._thread.is_alive():
            with self._db_lock:
                self._connection.close()
                self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *_exc) -> None:
        self.close()
