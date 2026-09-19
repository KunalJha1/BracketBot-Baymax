import sys
import threading

from bbapps.greeter.person_finder import PersonTrackerClient


FAKE_TRACKER = """
import json, sys, time
print(json.dumps({"ready": True}), flush=True)
for line in sys.stdin:
    request = json.loads(line)
    if request.get("cmd") == "acquire":
        if request["purpose"] == "slow":
            time.sleep(0.3)
            continue            # only answers after a cancel
        print(json.dumps({"id": request["id"], "found": True, "purpose": request["purpose"]}), flush=True)
    elif request.get("cmd") == "cancel":
        print(json.dumps({"id": 2, "found": False, "reason": "cancelled"}), flush=True)
"""


def make_client(tmp_path):
    script = tmp_path / "person_tracker.py"
    script.write_text(FAKE_TRACKER)
    uv = tmp_path / "uv"
    uv.write_text(f'#!/bin/sh\nexec {sys.executable} "$3"\n')
    uv.chmod(0o755)
    return PersonTrackerClient(script, uv_bin=str(uv), timeout_s=5.0)


def test_client_round_trips_acquire(tmp_path):
    client = make_client(tmp_path)
    try:
        result = client.acquire("scan", threading.Event())
        assert result == {"id": 1, "found": True, "purpose": "scan"}
    finally:
        client.close()
    assert not client.running()


def test_client_cancel_sends_cancel_and_reports_it(tmp_path):
    client = make_client(tmp_path)
    cancel = threading.Event()
    try:
        client.acquire("scan", threading.Event())
        threading.Timer(0.2, cancel.set).start()
        assert client.acquire("slow", cancel) == {"found": False, "reason": "cancelled"}
    finally:
        client.close()


def test_missing_script_is_unavailable_not_an_error(tmp_path):
    client = PersonTrackerClient(tmp_path / "missing.py")
    result = client.acquire("scan", threading.Event())
    assert result["unavailable"] is True
