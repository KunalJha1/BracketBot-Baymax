import json
import threading
import time
from types import SimpleNamespace
import wave

from scripts.robot_dashboard import (
    ACTION_LIST,
    ACTIONS,
    DEFAULT_SSH_HOSTS,
    EFFECT_RUNNER,
    FOLLOW_GAP_DEFAULT,
    FOLLOW_GAP_MAX,
    FOLLOW_GAP_MIN,
    FOLLOW_MODULES,
    FOLLOW_RUNNER,
    FOLLOW_STATUS_PREFIX,
    ROUTINE_LIST,
    ROUTINES,
    ROOT,
    RUNNER,
    SSH_PROBE_OPTIONS,
    SSH_PROBE_TIMEOUT,
    RobotController,
    action_bundle_paths,
    action_resource_path,
    parse_hosts,
    remote_python_command,
)


def wait_until_idle(controller, timeout=5.0):
    deadline = time.monotonic() + timeout
    while controller.state.snapshot()["running"] and time.monotonic() < deadline:
        time.sleep(0.02)
    assert not controller.state.snapshot()["running"]


def test_catalog_ids_keys_and_routine_steps_are_valid():
    ids = [item.id for item in ACTION_LIST]
    keys = [item.key for item in (*ACTION_LIST, *ROUTINE_LIST) if item.key]

    assert len(ids) == len(set(ids))
    assert len(keys) == len(set(keys))
    assert len(ACTIONS) >= 15
    assert {item.category for item in ACTION_LIST} == {
        "Gestures", "Lights", "Sounds", "Music"
    }
    assert all(set(routine.steps) <= ACTIONS.keys() for routine in ROUTINE_LIST)


def test_all_catalog_resources_exist():
    assert RUNNER.is_file()
    assert EFFECT_RUNNER.is_file()
    for action in ACTION_LIST:
        if action.executor in {"gesture", "sound"}:
            assert action_resource_path(action).is_file(), action.id


def test_public_catalog_hides_execution_details():
    public = ACTION_LIST[0].public()

    assert public["id"] == "wave"
    assert public["risk"] == "motion"
    assert "executor" not in public
    assert "resource" not in public
    assert "source" not in public
    assert "preview_url" not in public

    sound_public = ACTIONS["music-calm"].public()
    assert sound_public["preview_url"] == "/api/audio/music-calm"


def test_generated_music_is_robot_speaker_compatible_pcm():
    for name, minimum_duration in (("baymax_calm.wav", 10), ("baymax_celebration.wav", 8)):
        path = ROOT / "bbapps" / "play_sound" / "wavs" / name
        with wave.open(str(path), "rb") as sound:
            assert sound.getnchannels() == 1
            assert sound.getsampwidth() == 2
            assert sound.getframerate() == 16_000
            assert sound.getcomptype() == "NONE"
            assert sound.getnframes() / sound.getframerate() >= minimum_duration


def test_simulation_runs_single_action_without_ssh():
    controller = RobotController(("not-used",), simulate=True)

    ok, message = controller.run_action("sound-birthday")
    assert ok is True
    assert message == "Started Birthday sound"
    wait_until_idle(controller)

    state = controller.state.snapshot()
    assert state["mode"] == "simulation"
    assert state["host"] == "local-simulator"
    assert state["error"] is None
    assert state["phase"] == "Birthday sound complete"
    assert any("[simulation] sound: Birthday sound" in line for line in state["log"])


def test_simulation_runs_routine_steps_in_declared_order():
    controller = RobotController(("not-used",), simulate=True)

    ok, _ = controller.run_routine("thinking")
    assert ok is True
    wait_until_idle(controller)

    log = controller.state.snapshot()["log"]
    first = log.index("Step 1/2 — Thinking light")
    second = log.index("Step 2/2 — Processing sound")
    assert first < second
    assert ROUTINES["thinking"].steps == ("light-thinking", "sound-processing")


def test_stop_interrupts_simulated_operation():
    controller = RobotController(("not-used",), simulate=True)
    assert controller.run_action("light-calm")[0]
    time.sleep(0.08)

    ok, message = controller.stop()
    assert ok is True
    assert message == "Stop requested"
    wait_until_idle(controller)

    assert controller.state.snapshot()["phase"] == "Stopped safely"


def test_lean_toggle_and_global_stop_restore_balance_in_simulation():
    controller = RobotController(("not-used",), simulate=True)

    assert controller.set_lean(True) == (True, "Lean enabled")
    state = controller.state.snapshot()
    assert state["lean_enabled"] is True
    assert "4.0°" in state["lean_phase"]

    assert controller.stop() == (True, "Stop requested")
    state = controller.state.snapshot()
    assert state["lean_enabled"] is False
    assert state["lean_transition"] is False
    assert state["lean_phase"] == "Balance mode (simulation)"


def test_lean_rejects_non_boolean_and_repeated_state():
    controller = RobotController(("not-used",), simulate=True)

    assert controller.set_lean("yes") == (False, "enabled must be true or false")
    assert controller.set_lean(False) == (False, "Balance mode is already active")


def test_unknown_operations_are_rejected():
    controller = RobotController(("not-used",), simulate=True)

    assert controller.run_action("not-an-action") == (False, "Unknown command")
    assert controller.run_routine("not-a-routine") == (False, "Unknown routine")


def test_parse_hosts_deduplicates_and_preserves_order():
    assert parse_hosts("botwifi, bot, botwifi") == ("botwifi", "bot")


def test_default_routes_try_mdns_wifi_before_usb():
    assert DEFAULT_SSH_HOSTS == (
        "botwifi",
        "bracketbot@bracketbot-184.local",
        "bot",
    )
    assert SSH_PROBE_TIMEOUT > 2 * 3
    assert SSH_PROBE_OPTIONS[-4:] == (
        "-o", "ControlMaster=no", "-o", "ControlPath=none"
    )


def test_action_bundle_contains_each_runner_and_asset_once():
    bundle = action_bundle_paths()

    assert RUNNER in bundle
    assert EFFECT_RUNNER in bundle
    assert len(bundle) == len(set(bundle))
    assert all(path.is_file() for path in bundle)


def test_remote_python_prefers_existing_venv_and_keeps_uv_fallback():
    command = remote_python_command("/tmp/example.py", "--name", "Fist bump")

    assert '"$HOME/bbos/.venv/bin/python"' in command
    assert 'uv" run --no-sync --project "$HOME/bbos"' in command
    assert "'Fist bump'" in command


def test_deploy_skips_unchanged_cached_files(tmp_path, monkeypatch):
    asset = tmp_path / "asset.txt"
    asset.write_text("v1")
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr("scripts.robot_dashboard.subprocess.run", fake_run)
    controller = RobotController(("bot",))

    assert controller._deploy("bot", asset) is True
    assert controller._deploy("bot", asset) is False
    assert len(calls) == 1

    asset.write_text("version two")
    assert controller._deploy("bot", asset) is True
    assert len(calls) == 2


# --- follow mode ------------------------------------------------------------


def wait_for(predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.02)
    raise AssertionError("condition not reached")


def follow_off(controller):
    state = controller.state.snapshot()
    return not state["follow_enabled"] and not state["follow_transition"]


def test_follow_protocol_matches_the_runner():
    import follow_core

    cfg = follow_core.FollowConfig()
    assert FOLLOW_STATUS_PREFIX == follow_core.STATUS_PREFIX
    assert (FOLLOW_GAP_MIN, FOLLOW_GAP_MAX, FOLLOW_GAP_DEFAULT) == (cfg.gap_min, cfg.gap_max, cfg.gap_default)

    controller = RobotController(("not-used",), simulate=True)
    controller.set_follow(True)
    wait_for(lambda: controller.state.snapshot()["follow_status"] is not None)
    simulated = controller.state.snapshot()["follow_status"]
    controller.stop()
    wait_for(lambda: follow_off(controller))

    out = follow_core.FollowLoop(cfg).tick(follow_core.TickInputs(0.0, 0.1, False, 0.0, 0.0, 0.0, 0.0))
    real = json.loads(follow_core.status_line(out)[len(FOLLOW_STATUS_PREFIX):])
    assert set(simulated) == set(real)


def test_follow_simulation_reports_status_and_stops():
    controller = RobotController(("not-used",), simulate=True)

    assert controller.set_follow(True) == (True, "Starting follow mode")
    wait_for(lambda: (controller.state.snapshot()["follow_status"] or {}).get("state") == "FOLLOWING")
    assert controller.state.snapshot()["follow_enabled"] is True
    assert controller.set_follow(True) == (False, "Follow is already on")

    assert controller.stop() == (True, "Stop requested")
    wait_for(lambda: follow_off(controller))
    state = controller.state.snapshot()
    assert state["follow_phase"] == "Follow off"
    assert state["follow_status"] is None
    assert state["error"] is None


def test_follow_excludes_actions_and_lean():
    controller = RobotController(("not-used",), simulate=True)
    controller.set_follow(True)
    wait_for(lambda: controller.state.snapshot()["follow_enabled"])

    assert controller.run_action("wave") == (False, "Stop following before running actions")
    assert controller.run_routine("welcome") == (False, "Stop following before running actions")
    assert controller.set_lean(True) == (False, "Stop following before enabling lean")

    assert controller.set_follow(False) == (True, "Stopping follow mode")
    wait_for(lambda: follow_off(controller))
    assert controller.set_lean(True) == (True, "Lean enabled")
    assert controller.set_follow(True) == (False, "Return to balance mode before following")


def test_follow_gap_validation():
    controller = RobotController(("not-used",), simulate=True)

    for bad in ("1.2", True, None, float("nan")):
        assert controller.set_follow_gap(bad) == (False, "gap must be a number of metres")
    assert controller.set_follow_gap(2.0) == (False, "gap must be between 0.6 and 1.5 m")
    assert controller.set_follow_gap(1.25) == (True, "Gap set to 1.25 m")
    assert controller.state.snapshot()["follow_gap"] == 1.25


def test_action_bundle_ships_the_follow_runner_and_its_modules():
    bundle = action_bundle_paths()
    assert FOLLOW_RUNNER in bundle
    assert all(module in bundle for module in FOLLOW_MODULES)


class FakeFollowProcess:
    """Stands in for the ssh process: records stdin lines, emits runner output."""

    def __init__(self, command, **kwargs):
        self.command = command
        self.lines = []
        self.done = threading.Event()
        self.stdin = self
        self.stdout = self._output()

    def write(self, text):
        self.lines.append(json.loads(text))
        if self.lines[-1]["type"] == "stop":
            self.done.set()

    def flush(self):
        pass

    def _output(self):
        yield "[follow] follow active (v_max 0.15 m/s, gap 1.00 m) - stand in front of the robot\n"
        yield 'FOLLOW_STATUS {"state":"SEARCHING","range":null}\n'
        self.done.wait(5)
        yield "[follow] exit: stop\n"

    def poll(self):
        return 0 if self.done.is_set() else None

    def wait(self):
        self.done.wait(5)
        return 0


def test_follow_robot_mode_streams_heartbeats_gap_and_stop(monkeypatch):
    processes, remote_stops = [], []

    def fake_popen(command, **kwargs):
        processes.append(FakeFollowProcess(command, **kwargs))
        return processes[-1]

    monkeypatch.setattr("scripts.robot_dashboard.subprocess.Popen", fake_popen)
    controller = RobotController(("bot",), follow_args=("--v-max", "0.15"))
    controller.state.host = "bot"
    monkeypatch.setattr(controller, "_deploy", lambda host, *paths: False)
    monkeypatch.setattr(controller, "_request_remote_stop",
                        lambda host, pid_file, name: remote_stops.append(pid_file))

    assert controller.set_follow(True) == (True, "Starting follow mode")
    wait_for(lambda: controller.state.snapshot()["follow_enabled"])
    process = processes[0]
    assert "/tmp/robot_follow.py --gap 1.00 --pid-file /tmp/bracketbot-follow-1.pid --v-max 0.15" in process.command[-1]
    assert '"$HOME/bbos/.venv/bin/python"' in process.command[-1]
    wait_for(lambda: sum(line["type"] == "heartbeat" for line in process.lines) >= 2)
    assert controller.state.snapshot()["follow_status"] == {"state": "SEARCHING", "range": None}

    controller.set_follow_gap(1.3)
    assert {"type": "gap", "m": 1.3} in process.lines

    assert controller.stop() == (True, "Stop requested")
    wait_for(lambda: follow_off(controller))
    assert process.lines[-1] == {"type": "stop"}
    assert remote_stops == ["/tmp/bracketbot-follow-1.pid"]
    state = controller.state.snapshot()
    assert state["follow_phase"] == "Follow off"
    assert state["error"] is None


class CrashingFollowProcess:
    """Stands in for an ssh process whose stdout decoding blows up mid-stream
    (e.g. a UnicodeDecodeError from non-ASCII remote output with text=True)."""

    def __init__(self, command, **kwargs):
        self.command = command
        self.lines = []
        self.terminated = False
        self.stdin = self
        self.stdout = self._output()

    def write(self, text):
        self.lines.append(json.loads(text))

    def flush(self):
        pass

    def _output(self):
        yield "[follow] follow active (v_max 0.15 m/s, gap 1.00 m) - stand in front of the robot\n"
        raise ValueError("simulated decode failure")

    def poll(self):
        return 0 if self.terminated else None

    def terminate(self):
        self.terminated = True

    def wait(self):
        return 0


def test_crashed_follow_reader_terminates_process_and_stops_heartbeats(monkeypatch):
    processes = []

    def fake_popen(command, **kwargs):
        processes.append(CrashingFollowProcess(command, **kwargs))
        return processes[-1]

    monkeypatch.setattr("scripts.robot_dashboard.subprocess.Popen", fake_popen)
    controller = RobotController(("bot",), follow_args=("--v-max", "0.15"))
    controller.state.host = "bot"
    monkeypatch.setattr(controller, "_deploy", lambda host, *paths: False)
    monkeypatch.setattr(controller, "_request_remote_stop", lambda *a, **k: None)

    assert controller.set_follow(True) == (True, "Starting follow mode")
    wait_for(lambda: follow_off(controller))

    process = processes[0]
    assert process.terminated is True
    state = controller.state.snapshot()
    assert state["error"] is not None

    heartbeat_count = sum(line["type"] == "heartbeat" for line in process.lines)
    time.sleep(0.4)
    assert sum(line["type"] == "heartbeat" for line in process.lines) == heartbeat_count


def test_stop_during_startup_sends_stop_and_never_enables_follow(monkeypatch):
    processes = []
    controller = RobotController(("bot",), follow_args=("--v-max", "0.15"))
    controller.state.host = "bot"

    def fake_popen(command, **kwargs):
        process = FakeFollowProcess(command, **kwargs)
        # A stop lands in the window between Popen returning and the dashboard
        # storing follow_process under the lock (the I2 race).
        with controller.state.lock:
            controller.state.follow_requested = False
        processes.append(process)
        return process

    monkeypatch.setattr("scripts.robot_dashboard.subprocess.Popen", fake_popen)
    monkeypatch.setattr(controller, "_deploy", lambda host, *paths: False)
    monkeypatch.setattr(controller, "_request_remote_stop", lambda *a, **k: None)

    assert controller.set_follow(True) == (True, "Starting follow mode")
    wait_for(lambda: follow_off(controller))

    process = processes[0]
    assert {"type": "stop"} in process.lines
    assert controller.state.snapshot()["follow_enabled"] is False


def test_heartbeat_stops_once_the_browser_tab_stops_polling(monkeypatch):
    processes = []

    def fake_popen(command, **kwargs):
        processes.append(FakeFollowProcess(command, **kwargs))
        return processes[-1]

    monkeypatch.setattr("scripts.robot_dashboard.subprocess.Popen", fake_popen)
    controller = RobotController(("bot",), follow_args=("--v-max", "0.15"))
    controller.state.host = "bot"
    monkeypatch.setattr(controller, "_deploy", lambda host, *paths: False)
    monkeypatch.setattr(controller, "_request_remote_stop", lambda *a, **k: None)

    assert controller.set_follow(True) == (True, "Starting follow mode")
    wait_for(lambda: controller.state.snapshot()["follow_enabled"])
    process = processes[0]
    wait_for(lambda: sum(line["type"] == "heartbeat" for line in process.lines) >= 1)

    with controller.state.lock:
        controller.state.last_status_poll = time.monotonic() - 2.0

    heartbeat_count = sum(line["type"] == "heartbeat" for line in process.lines)
    time.sleep(0.5)
    assert sum(line["type"] == "heartbeat" for line in process.lines) == heartbeat_count

    controller.stop()
    wait_for(lambda: follow_off(controller))
