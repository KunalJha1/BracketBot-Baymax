import json
from urllib import error

import pytest

from scripts.greeter_action import api_call, remove_pid_file, write_pid_file


class FakeResponse:
    def __init__(self, body):
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def read(self):
        return json.dumps(self.body).encode("utf-8")


def test_bridge_posts_allowlisted_action_to_robot_loopback(monkeypatch):
    calls = []

    def open_request(api_request, timeout):
        calls.append((api_request, timeout))
        return FakeResponse({"ok": True, "message": "Started point"})

    monkeypatch.setattr("scripts.greeter_action.request.urlopen", open_request)

    result = api_call(
        "http://127.0.0.1:8018/",
        "/api/action",
        {"action": "point"},
    )

    api_request, timeout = calls[0]
    assert api_request.full_url == "http://127.0.0.1:8018/api/action"
    assert api_request.method == "POST"
    assert json.loads(api_request.data) == {"action": "point"}
    assert timeout == 3.0
    assert result["ok"] is True


def test_bridge_reports_greeter_unavailable(monkeypatch):
    def unavailable(*args, **kwargs):
        raise error.URLError("connection refused")

    monkeypatch.setattr("scripts.greeter_action.request.urlopen", unavailable)

    with pytest.raises(RuntimeError, match="ensure bbapps/emotion_greeter is running"):
        api_call("http://127.0.0.1:8018", "/api/status")


def test_bridge_pid_file_lifecycle(tmp_path):
    pid_file = tmp_path / "point.pid"

    write_pid_file(pid_file)
    assert int(pid_file.read_text()) > 0
    remove_pid_file(pid_file)
    assert not pid_file.exists()
