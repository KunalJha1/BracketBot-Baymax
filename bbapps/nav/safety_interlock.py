"""Pure, dependency-free reader for the perception/navigation safety handoff."""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any, Callable


class GroundSafetyInterlock:
    """Read an atomically published possible-person-on-ground observation.

    A malformed update never clears a previously active stop. The writer uses
    atomic replacement, so malformed reads normally mean an operator should
    inspect the perception service rather than allowing motion to resume.
    """

    def __init__(
        self,
        path: Path,
        *,
        poll_seconds: float = 0.1,
        logger: Callable[[str], Any] | None = None,
    ) -> None:
        self.path = path
        self.poll_seconds = poll_seconds
        self.logger = logger
        self.checked_at = 0.0
        self.active = False
        self.alerts: list[dict[str, Any]] = []

    def read(self, now: float | None = None) -> tuple[bool, list[dict[str, Any]]]:
        now = time.time() if now is None else float(now)
        if now - self.checked_at < self.poll_seconds:
            return self.active, self.alerts
        self.checked_at = now
        try:
            payload = json.loads(self.path.read_text())
            active = bool(
                payload.get("possible_person_on_ground")
                and payload.get("status") == "alert"
            )
            alerts = payload.get("alerts", []) if active else []
            if not isinstance(alerts, list):
                alerts = []
            self.active = active
            self.alerts = alerts
        except FileNotFoundError:
            self.active = False
            self.alerts = []
        except (OSError, ValueError, TypeError) as exc:
            if self.active and self.logger is not None:
                self.logger(
                    f"ground alert unreadable; keeping stop latched: {exc}"
                )
        return self.active, self.alerts
