import json
import threading
import time
from types import SimpleNamespace
import wave

import pytest

from scripts.robot_dashboard import (
    ACTION_LIST,
    ACTIONS,
    CAMERA_POINT_RUNNER,
    DEFAULT_SSH_HOSTS,
    EFFECT_RUNNER,
    GREETER_ACTION_RUNNER,
    REMOTE_TABLE_REST_RUNNER,
    REMOTE_GREETER_ACTION_RUNNER,
    ROUTINE_LIST,
    ROUTINES,
    ROOT,
    RUNNER,
    SSH_PROBE_OPTIONS,
    SSH_PROBE_TIMEOUT,
    TABLE_REST_RUNNER,
    RobotConnectionError,
    RobotController,
    PAGE,
    action_bundle_paths,
    action_resource_path,
    file_signature,
    parse_hosts,
    remote_python_command,
    start_source_reloader,
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
    assert len(ACTIONS) >= 17
    assert {item.category for item in ACTION_LIST} == {
        "Gestures", "Lights", "Sounds", "Music", "Positioning"
    }
    assert all(set(routine.steps) <= ACTIONS.keys() for routine in ROUTINE_LIST)
    point = ACTIONS["point-person"]
    assert point.executor == "camera-gesture"
    assert point.resource == "point"
    assert point.channels == ("camera", "left-arm", "right-arm")
    table = ACTIONS["table-rest"]
    assert table.executor == "table-rest"
    assert table.channels == ("depth-camera", "left-arm", "right-arm")
    assert table.risk == "contact-motion"


def test_all_catalog_resources_exist():
    assert RUNNER.is_file()
    assert EFFECT_RUNNER.is_file()
    for action in ACTION_LIST:
        if action.executor in {"gesture", "sound"}:
            assert action_resource_path(action).is_file(), action.id


def test_salute_lifts_holds_then_waves_with_left_arm():
    frames = json.loads(action_resource_path(ACTIONS["salute"]).read_text())
    times = [frame["t"] for frame in frames]
    gaps = [later - earlier for earlier, later in zip(times, times[1:])]
    hold_index = gaps.index(max(gaps))
    source = json.loads(
        (ROOT / "bbapps" / "greeter" / "movements" / "wave.json").read_text()
    )
    extension_source = json.loads(
        (ROOT / "bbapps" / "greeter" / "movements" / "hug.json").read_text()
    )
    recorded_poses = source + extension_source
    wave_poses = frames[hold_index + 1 :]

    assert ACTIONS["salute"].channels == ("left-arm",)
    assert ACTIONS["salute"].risk == "motion"
    assert len(frames) > len(source)
    assert all(later > earlier for earlier, later in zip(times, times[1:]))
    assert max(gaps) >= 0.4
    assert frames[hold_index]["left"] == frames[hold_index + 1]["left"]
    assert max(
        max(frame["left"][joint] for frame in wave_poses)
        - min(frame["left"][joint] for frame in wave_poses)
        for joint in range(1, 7)
    ) > 0.2
    for joint in range(8):
        recorded = [frame["left"][joint] for frame in recorded_poses]
        generated = [frame["left"][joint] for frame in frames]
        assert min(generated) >= min(recorded)
        assert max(generated) <= max(recorded)
    assert all(frame["right"] == frames[0]["right"] for frame in frames)
    assert 5.0 < max(times) < 7.0


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


def test_simulation_exposes_camera_point_action():
    controller = RobotController(("not-used",), simulate=True)

    ok, message = controller.run_action("point-person")
    assert ok is True
    assert message == "Started Point at person"
    wait_until_idle(controller)

    log = controller.state.snapshot()["log"]
    assert any("[simulation] camera-gesture: Point at person" in line for line in log)


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


def test_status_has_per_process_identity_for_browser_hot_reload():
    first = RobotController(("not-used",), simulate=True).state.snapshot()["server_id"]
    second = RobotController(("not-used",), simulate=True).state.snapshot()["server_id"]

    assert first
    assert first != second
    assert "window.location.reload()" in PAGE


def test_source_reloader_waits_for_safe_balance_state(tmp_path):
    source = tmp_path / "dashboard.py"
    source.write_text("first")
    assert file_signature(source) is not None
    controller = RobotController(("not-used",), simulate=True)
    assert controller.set_lean(True)[0]
    stopped = threading.Event()
    reload_requested = threading.Event()
    watcher_stop = threading.Event()
    server = SimpleNamespace(shutdown=stopped.set)
    start_source_reloader(
        server,
        controller,
        source,
        reload_requested,
        watcher_stop,
    )
    try:
        source.write_text("second version")
        assert not stopped.wait(0.9)
        assert controller.set_lean(False)[0]
        assert stopped.wait(1.2)
        assert reload_requested.is_set()
    finally:
        watcher_stop.set()


def test_table_stop_returns_arms_before_restoring_lean_in_simulation():
    controller = RobotController(("not-used",), simulate=True)

    assert controller.set_lean(True)[0]
    assert controller.run_action("table-rest")[0]
    time.sleep(0.05)
    assert controller.set_lean(False) == (
        False,
        "Wait for the current arm action to finish",
    )
    assert controller.stop() == (True, "Stop requested")
    wait_until_idle(controller)

    state = controller.state.snapshot()
    assert state["lean_enabled"] is False
    assert state["lean_phase"] == "Balance mode (simulation)"
    assert state["phase"] == "Stopped safely"


def test_completed_table_placement_releases_dashboard_for_next_action():
    controller = RobotController(("not-used",), simulate=True)

    assert controller.run_action("table-rest")[0]
    wait_until_idle(controller)
    assert controller.state.snapshot()["phase"] == "Place arms on table complete"

    assert controller.run_action("wave")[0]
    assert controller.state.snapshot()["action"] == "Wave"
    assert controller.stop() == (True, "Stop requested")
    wait_until_idle(controller)


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
    assert GREETER_ACTION_RUNNER in bundle
    assert CAMERA_POINT_RUNNER in bundle
    assert TABLE_REST_RUNNER in bundle
    assert len(bundle) == len(set(bundle))
    assert all(path.is_file() for path in bundle)


def test_remote_python_prefers_existing_venv_and_keeps_uv_fallback():
    command = remote_python_command("/tmp/example.py", "--name", "Fist bump")

    assert '"$HOME/bbos/.venv/bin/python"' in command
    assert 'uv" run --no-sync --project "$HOME/bbos"' in command
    assert "'Fist bump'" in command


def test_camera_point_dispatches_to_robot_local_greeter(monkeypatch):
    controller = RobotController(("bot",))
    commands = []
    deployed = []
    controller.state.pid_file = "/tmp/test-point.pid"

    monkeypatch.setattr(
        controller,
        "_deploy",
        lambda host, *paths: deployed.extend(paths),
    )
    monkeypatch.setattr(
        controller,
        "_run_remote_process",
        lambda host, command: commands.append((host, command)),
    )

    controller._execute_camera_gesture("bot", ACTIONS["point-person"])

    assert deployed == [GREETER_ACTION_RUNNER, CAMERA_POINT_RUNNER]
    assert commands[0][0] == "bot"
    assert REMOTE_GREETER_ACTION_RUNNER in commands[0][1]
    assert "point --pid-file /tmp/test-point.pid" in commands[0][1]


def test_table_rest_dispatches_adaptive_robot_runner(monkeypatch):
    controller = RobotController(("bot",))
    commands = []
    deployed = []
    controller.state.pid_file = "/tmp/test-table.pid"

    monkeypatch.setattr(
        controller,
        "_deploy",
        lambda host, *paths: deployed.extend(paths),
    )
    monkeypatch.setattr(
        controller,
        "_run_remote_process",
        lambda host, command: commands.append((host, command)),
    )

    controller._execute_table_rest("bot")

    assert deployed == [TABLE_REST_RUNNER]
    assert commands[0][0] == "bot"
    assert REMOTE_TABLE_REST_RUNNER in commands[0][1]
    assert "--pid-file /tmp/test-table.pid" in commands[0][1]


def test_action_rejection_does_not_mark_robot_offline(monkeypatch):
    controller = RobotController(("bot",))
    controller.state.host = "bot"
    monkeypatch.setattr(
        controller,
        "_execute_action",
        lambda host, info: (_ for _ in ()).throw(RuntimeError("No person visible")),
    )

    assert controller.run_action("point-person")[0]
    wait_until_idle(controller)

    state = controller.state.snapshot()
    assert state["connected"] is True
    assert state["host"] == "bot"
    assert state["error"] == "No person visible"


def test_transport_failure_marks_robot_offline(monkeypatch):
    controller = RobotController(("bot",))
    controller.state.host = "bot"
    monkeypatch.setattr(
        controller,
        "_execute_action",
        lambda host, info: (_ for _ in ()).throw(RobotConnectionError("SSH failed")),
    )

    assert controller.run_action("point-person")[0]
    wait_until_idle(controller)

    assert controller.state.snapshot()["connected"] is False


def test_reconnect_restores_orphaned_lean_before_actions(monkeypatch):
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=0, stdout="[base] restored", stderr="")

    monkeypatch.setattr("scripts.robot_dashboard.subprocess.run", fake_run)
    controller = RobotController(("bot",))

    controller._restore_orphaned_base("bot")

    command = calls[0]
    assert command[0] == "ssh"
    assert command[-2] == "bot"
    assert "pkill -INT" in command[-1]
    assert "robot_base_mode" in command[-1]
    assert controller.state.snapshot()["lean_phase"] == "[base] restored"


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


def test_dashboard_retains_long_table_diagnostic_history():
    controller = RobotController(("not-used",), simulate=True)
    for index in range(300):
        controller.state.add_log(f"[table][candidate] diagnostic={index}")

    visible = controller.state.snapshot()["log"]

    assert len(visible) == 240
    assert visible[0].endswith("diagnostic=60")
    assert visible[-1].endswith("diagnostic=299")


def test_zero_exit_traceback_is_still_reported_as_runner_failure(monkeypatch):
    process = SimpleNamespace(
        stdout=[
            "Traceback (most recent call last):\n",
            "[table][fatal] RuntimeError: no valid table\n",
        ],
        wait=lambda: 0,
    )
    monkeypatch.setattr(
        "scripts.robot_dashboard.subprocess.Popen",
        lambda *args, **kwargs: process,
    )
    controller = RobotController(("bot",))

    with pytest.raises(RuntimeError, match="status 0.*no valid table"):
        controller._run_remote_process("bot", "ignored")

    assert controller.state.snapshot()["log"][-1] == "[remote] runner exited status=0"
