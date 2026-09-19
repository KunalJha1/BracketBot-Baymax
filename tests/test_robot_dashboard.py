import time
import wave

from scripts.robot_dashboard import (
    ACTION_LIST,
    ACTIONS,
    EFFECT_RUNNER,
    ROUTINE_LIST,
    ROUTINES,
    ROOT,
    RUNNER,
    RobotController,
    action_resource_path,
    parse_hosts,
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
