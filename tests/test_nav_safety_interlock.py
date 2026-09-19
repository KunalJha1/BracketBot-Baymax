import json
from pathlib import Path
import sys


NAV_DIR = Path(__file__).parents[1] / "bbapps" / "nav"
sys.path.insert(0, str(NAV_DIR))

from safety_interlock import GroundSafetyInterlock  # noqa: E402


def write_state(path, status, alert, alerts=None):
    path.write_text(
        json.dumps(
            {
                "status": status,
                "possible_person_on_ground": alert,
                "alerts": [] if alerts is None else alerts,
            }
        )
    )


def test_confirmed_observation_activates_interlock(tmp_path):
    path = tmp_path / "alert.json"
    write_state(path, "alert", True, [{"track_id": 9}])
    interlock = GroundSafetyInterlock(path)

    assert interlock.read(100) == (True, [{"track_id": 9}])


def test_clear_or_missing_file_does_not_activate_interlock(tmp_path):
    path = tmp_path / "alert.json"
    interlock = GroundSafetyInterlock(path)

    assert interlock.read(100) == (False, [])
    write_state(path, "clear", False)
    assert interlock.read(101) == (False, [])


def test_malformed_update_cannot_clear_active_stop(tmp_path):
    path = tmp_path / "alert.json"
    logs = []
    write_state(path, "alert", True, [{"track_id": 3}])
    interlock = GroundSafetyInterlock(path, logger=logs.append)
    assert interlock.read(100)[0]

    path.write_text("not json")

    assert interlock.read(101) == (True, [{"track_id": 3}])
    assert logs and "keeping stop latched" in logs[-1]


def test_poll_interval_uses_cached_state(tmp_path):
    path = tmp_path / "alert.json"
    write_state(path, "alert", True)
    interlock = GroundSafetyInterlock(path, poll_seconds=0.1)
    assert interlock.read(100)[0]

    write_state(path, "clear", False)

    assert interlock.read(100.05)[0]
    assert not interlock.read(100.11)[0]
