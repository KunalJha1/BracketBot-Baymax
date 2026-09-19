from datetime import datetime, timezone
import time

from bbapps.greeter.reminders import PersistentReminderScheduler


def wait_for(predicate, timeout=1.5):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.005)
    assert predicate()


def test_timezone_aware_schedule_list_cancel_and_audit(tmp_path):
    scheduler = PersistentReminderScheduler(
        tmp_path / "reminders.sqlite3",
        lambda _reminder: None,
        timezone_name="America/Toronto",
        poll_interval=0.05,
    )
    try:
        reminder = scheduler.schedule_at(
            "2099-01-15T09:30:00",
            "take a stretch break",
            source="test",
        )

        public = reminder.public()
        assert public["due_at_local"] == "2099-01-15T09:30:00-05:00"
        assert public["due_at_utc"] == "2099-01-15T14:30:00Z"
        assert scheduler.list_pending() == [reminder]
        assert scheduler.cancel(reminder.id) == [reminder.id]
        assert scheduler.list_pending() == []
        assert [item["event"] for item in scheduler.audit(reminder.id)] == [
            "cancelled",
            "scheduled",
        ]
    finally:
        scheduler.close()


def test_pending_reminder_survives_scheduler_restart(tmp_path):
    path = tmp_path / "reminders.sqlite3"
    first_deliveries = []
    first = PersistentReminderScheduler(
        path,
        first_deliveries.append,
        timezone_name="UTC",
        poll_interval=0.02,
    )
    reminder = first.schedule_after(0.2, "persistent message", source="test")
    first.close()

    delivered = []
    second = PersistentReminderScheduler(
        path,
        delivered.append,
        timezone_name="UTC",
        poll_interval=0.02,
    )
    try:
        wait_for(lambda: len(delivered) == 1)
        assert first_deliveries == []
        assert delivered[0].id == reminder.id
        assert delivered[0].message == "persistent message"
        assert scheduler_events(second, reminder.id)[0] == "delivered"
        assert second.list_pending() == []
    finally:
        second.close()


def scheduler_events(scheduler, reminder_id):
    return [item["event"] for item in scheduler.audit(reminder_id)]


def test_offset_timestamp_is_normalized_to_utc(tmp_path):
    scheduler = PersistentReminderScheduler(
        tmp_path / "reminders.sqlite3",
        lambda _reminder: None,
        timezone_name="America/Toronto",
    )
    try:
        reminder = scheduler.schedule_at(
            datetime(2099, 7, 1, 13, 0, tzinfo=timezone.utc),
            "UTC event",
        )
        assert reminder.public()["due_at_utc"] == "2099-07-01T13:00:00Z"
        assert reminder.public()["due_at_local"] == "2099-07-01T09:00:00-04:00"
    finally:
        scheduler.close()
