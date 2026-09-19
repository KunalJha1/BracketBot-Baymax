"""Local, accessible web dashboard for allowlisted BracketBot actions.

The server binds to localhost by default. It discovers the first reachable SSH
alias (``botwifi`` then ``bot``), copies small safety-focused runners to the
robot, and executes only actions and routines from the fixed allowlists below.

Use ``--simulate`` to develop the complete UI and sequencing path without a
connected robot. Simulation never opens SSH or writes to BBOS.

    python3 scripts/robot_dashboard.py
    open http://127.0.0.1:8020
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import shlex
import subprocess
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "gesture_test.py"
EFFECT_RUNNER = ROOT / "scripts" / "robot_effect.py"
BASE_RUNNER = ROOT / "scripts" / "robot_base_mode.py"
REMOTE_RUNNER = "/tmp/gesture_test.py"
REMOTE_EFFECT_RUNNER = "/tmp/robot_effect.py"
REMOTE_BASE_RUNNER = "/tmp/robot_base_mode.py"
SSH_OPTIONS = (
    "-o", "BatchMode=yes",
    "-o", "ConnectTimeout=3",
    "-o", "ConnectionAttempts=1",
)

@dataclass(frozen=True)
class ActionSpec:
    """One allowlisted primitive that can also be used as a routine step."""

    id: str
    label: str
    description: str
    category: str
    executor: str
    channels: tuple[str, ...]
    key: str | None = None
    resource: str | None = None
    source: str | None = None
    rgb: tuple[int, int, int] | None = None
    pattern: str | None = None
    duration: float | None = None
    risk: str = "low"

    def public(self):
        data = asdict(self)
        if self.executor == "sound":
            data["preview_url"] = f"/api/audio/{self.id}"
        # Execution details stay server-side. The browser only needs metadata.
        for private in ("executor", "resource", "source", "rgb", "pattern", "duration"):
            data.pop(private)
        return data


@dataclass(frozen=True)
class RoutineSpec:
    """A deterministic series of allowlisted action IDs."""

    id: str
    label: str
    description: str
    steps: tuple[str, ...]
    key: str | None = None

    def public(self):
        return asdict(self)


def _action(action_id, label, description, category, executor, **kwargs):
    return ActionSpec(action_id, label, description, category, executor, **kwargs)


# This catalog is deliberately data, not arbitrary commands. It is the shared
# vocabulary for buttons today and assistant-generated routines later.
ACTION_LIST = (
    _action("wave", "Wave", "Friendly left-arm wave", "Gestures", "gesture",
            channels=("left-arm",), key="1", resource="wave.json", risk="motion"),
    _action("handshake", "Handshake", "Offer and shake the right hand", "Gestures", "gesture",
            channels=("right-arm",), key="2", resource="handshake.json", risk="contact-motion"),
    _action("fist-bump", "Fist bump", "Offer a right-handed fist bump", "Gestures", "gesture",
            channels=("right-arm",), key="3", resource="fist bump.json", risk="contact-motion"),
    _action("hug", "Hug", "Open both arms for a hug", "Gestures", "gesture",
            channels=("left-arm", "right-arm"), key="4", resource="hug.json", risk="contact-motion"),
    _action("dance", "Dance", "Recorded two-arm dance", "Gestures", "gesture",
            channels=("left-arm", "right-arm"), key="d", resource="dance.json",
            source="bbapps/mimic/recordings/dance.json", risk="motion"),
    _action("light-calm", "Calm light", "Slow cyan breathing light", "Lights", "led",
            channels=("led",), key="5", rgb=(72, 205, 220), pattern="pulse", duration=4.0),
    _action("light-ready", "Ready light", "Steady green ready signal", "Lights", "led",
            channels=("led",), key="6", rgb=(70, 220, 120), pattern="solid", duration=3.0),
    _action("light-thinking", "Thinking light", "Blue thinking pulse", "Lights", "led",
            channels=("led",), key="7", rgb=(70, 125, 255), pattern="pulse", duration=4.0),
    _action("light-celebrate", "Celebration light", "Magenta celebration blink", "Lights", "led",
            channels=("led",), key="8", rgb=(255, 70, 210), pattern="blink", duration=4.0),
    _action("lights-off", "Lights off", "Clear the dashboard light signal", "Lights", "led",
            channels=("led",), key="9", rgb=(0, 0, 0), pattern="solid", duration=0.25),
    _action("sound-processing", "Processing sound", "Play the thinking/processing cue", "Sounds", "sound",
            channels=("speaker",), key="p", resource="robot_processing.wav"),
    _action("sound-birthday", "Birthday sound", "Play the short birthday cue", "Sounds", "sound",
            channels=("speaker",), key="b", resource="happy_birthday.wav"),
    _action("sound-low-battery", "Battery reminder", "Play the low-battery reminder", "Sounds", "sound",
            channels=("speaker",), key="l", resource="low_battery_1.wav"),
    _action("music-calm", "Calm melody", "Play an original gentle instrumental", "Music", "sound",
            channels=("speaker",), key="m", resource="baymax_calm.wav"),
    _action("music-celebration", "Upbeat melody", "Play an original upbeat instrumental", "Music", "sound",
            channels=("speaker",), key="u", resource="baymax_celebration.wav"),
)
ACTIONS = {action.id: action for action in ACTION_LIST}

ROUTINE_LIST = (
    RoutineSpec("welcome", "Welcome", "Ready light, then wave", ("light-ready", "wave"), "w"),
    RoutineSpec("thinking", "Thinking", "Thinking light, then processing cue",
                ("light-thinking", "sound-processing"), "t"),
    RoutineSpec("celebrate", "Celebrate", "Celebration light, then birthday cue",
                ("light-celebrate", "sound-birthday"), "c"),
    RoutineSpec("goodbye", "Goodbye", "Calm light, then a wave", ("light-calm", "wave"), "g"),
    RoutineSpec("double-wave", "Double wave", "Wave twice", ("wave", "wave"), "v"),
    RoutineSpec("calm-moment", "Calm moment", "Calm light, then a gentle melody",
                ("light-calm", "music-calm"), "k"),
    RoutineSpec("dance-party", "Dance party", "Celebration light, upbeat melody, then dance",
                ("light-celebrate", "music-celebration", "dance"), "x"),
)
ROUTINES = {routine.id: routine for routine in ROUTINE_LIST}


def validate_catalog():
    ids = [action.id for action in ACTION_LIST]
    keys = [item.key for item in (*ACTION_LIST, *ROUTINE_LIST) if item.key]
    if len(ids) != len(set(ids)) or len(keys) != len(set(keys)):
        raise RuntimeError("action IDs and keyboard shortcuts must be unique")
    for routine in ROUTINE_LIST:
        missing = set(routine.steps) - ACTIONS.keys()
        if missing:
            raise RuntimeError(f"routine {routine.id} has unknown steps: {sorted(missing)}")
    for action in ACTION_LIST:
        if not action.channels:
            raise RuntimeError(f"{action.id} must declare at least one robot channel")
        if action.executor in {"gesture", "sound"} and not action.resource:
            raise RuntimeError(f"{action.id} requires an allowlisted resource")
        if action.executor == "led" and (
            action.rgb is None or action.pattern not in {"solid", "pulse", "blink"}
            or action.duration is None
        ):
            raise RuntimeError(f"{action.id} requires RGB, pattern, and duration")
        if action.executor not in {"gesture", "sound", "led"}:
            raise RuntimeError(f"{action.id} has unsupported executor {action.executor}")


validate_catalog()


def action_resource_path(info):
    if info.source:
        return ROOT / info.source
    if info.executor == "gesture":
        return ROOT / "bbapps" / "greeter" / "movements" / info.resource
    if info.executor == "sound":
        return ROOT / "bbapps" / "play_sound" / "wavs" / info.resource
    raise ValueError(f"{info.id} does not have a file resource")


class DashboardState:
    def __init__(self, ssh_hosts, simulate=False):
        self.ssh_hosts = tuple(ssh_hosts)
        self.simulate = simulate
        self.lock = threading.Lock()
        self.host = "local-simulator" if simulate else None
        self.checking = False
        self.running = False
        self.action = None
        self.operation_id = None
        self.step = None
        self.phase = "Simulation ready — no robot commands will run" if simulate else "Starting connection check"
        self.error = None
        self.log = []
        self.cancel_requested = False
        self.process = None
        self.pid_file = None
        self.lean_enabled = False
        self.lean_requested = False
        self.lean_transition = False
        self.lean_phase = "Balance mode"
        self.lean_process = None
        self.lean_pid_file = None

    def snapshot(self):
        with self.lock:
            return {
                "host": self.host,
                "checking": self.checking,
                "connected": self.host is not None,
                "running": self.running,
                "action": self.action,
                "operation_id": self.operation_id,
                "step": self.step,
                "phase": self.phase,
                "error": self.error,
                "log": list(self.log[-24:]),
                "candidates": list(self.ssh_hosts),
                "mode": "simulation" if self.simulate else "robot",
                "actions": [action.public() for action in ACTION_LIST],
                "routines": [routine.public() for routine in ROUTINE_LIST],
                "lean_enabled": self.lean_enabled,
                "lean_transition": self.lean_transition,
                "lean_phase": self.lean_phase,
            }

    def add_log(self, line):
        line = line.strip()
        if not line:
            return
        with self.lock:
            self.log.append(line)
            del self.log[:-80]
            self.phase = line

    def add_lean_log(self, line):
        line = line.strip()
        if not line:
            return
        with self.lock:
            self.log.append(f"[lean] {line}")
            del self.log[:-80]
            self.lean_phase = line


class RobotController:
    def __init__(self, ssh_hosts, simulate=False):
        self.state = DashboardState(ssh_hosts, simulate=simulate)
        self._discover_lock = threading.Lock()
        self._stop_monitor = threading.Event()
        self._operation_counter = 0
        self._lean_counter = 0

    @staticmethod
    def _probe(host):
        command = [
            "ssh", *SSH_OPTIONS, host,
            'test -x "$HOME/.local/bin/uv" && '
            'test -d "$HOME/bbos"',
        ]
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=5)
            return result.returncode == 0
        except (OSError, subprocess.TimeoutExpired):
            return False

    def discover(self):
        if self.state.simulate:
            with self.state.lock:
                self.state.host = "local-simulator"
                self.state.phase = "Simulation ready — no robot commands will run"
                self.state.error = None
            return
        if not self._discover_lock.acquire(blocking=False):
            return
        with self.state.lock:
            if self.state.running:
                self._discover_lock.release()
                return
            self.state.checking = True
            self.state.phase = "Checking robot connections…"
            self.state.error = None

        try:
            reachable = set()
            with ThreadPoolExecutor(max_workers=len(self.state.ssh_hosts)) as pool:
                futures = {
                    pool.submit(self._probe, host): host for host in self.state.ssh_hosts
                }
                for future in as_completed(futures):
                    if future.result():
                        reachable.add(futures[future])

            # Preserve the configured order when more than one interface works.
            selected = next(
                (host for host in self.state.ssh_hosts if host in reachable), None
            )
            with self.state.lock:
                self.state.host = selected
                if selected:
                    self.state.phase = f"Ready — connected through {selected}"
                else:
                    self.state.phase = "Robot not found"
                    self.state.error = (
                        "No configured SSH connection is reachable. Check Wi-Fi/USB, "
                        "then choose Reconnect."
                    )
        finally:
            with self.state.lock:
                self.state.checking = False
            self._discover_lock.release()

    def discover_async(self):
        threading.Thread(target=self.discover, name="robot-discovery", daemon=True).start()

    def start_monitor(self):
        if self.state.simulate:
            return
        self.discover_async()

        def monitor():
            while not self._stop_monitor.wait(15):
                with self.state.lock:
                    host = self.state.host
                    busy = self.state.running or self.state.checking
                if busy:
                    continue
                if host is None or not self._probe(host):
                    self.discover()

        threading.Thread(target=monitor, name="robot-monitor", daemon=True).start()

    def run_action(self, action):
        if action not in ACTIONS:
            return False, "Unknown command"

        return self._start_operation(action, ACTIONS[action].label, (action,))

    def run_routine(self, routine):
        if routine not in ROUTINES:
            return False, "Unknown routine"
        info = ROUTINES[routine]
        return self._start_operation(routine, info.label, info.steps)

    def _start_operation(self, operation_id, label, steps):

        with self.state.lock:
            if self.state.running:
                return False, f"{self.state.action} is already running"
            host = self.state.host
            if host is None:
                return False, "Robot is not connected; choose Reconnect"
            self.state.running = True
            self.state.action = label
            self.state.operation_id = operation_id
            self.state.step = None
            self.state.phase = f"Preparing {label}…"
            self.state.error = None
            self.state.log = []
            self.state.cancel_requested = False
            self._operation_counter += 1
            self.state.pid_file = f"/tmp/bracketbot-action-{self._operation_counter}.pid"

        threading.Thread(
            target=self._run_operation,
            args=(host, label, tuple(steps)),
            name=f"operation-{operation_id}",
            daemon=True,
        ).start()
        return True, f"Started {label}"

    def _run_operation(self, host, operation_label, steps):
        try:
            for index, action_id in enumerate(steps, start=1):
                with self.state.lock:
                    if self.state.cancel_requested:
                        self.state.phase = "Stopped safely"
                        return
                    self.state.step = {
                        "index": index,
                        "total": len(steps),
                        "id": action_id,
                        "label": ACTIONS[action_id].label,
                    }
                self.state.add_log(
                    f"Step {index}/{len(steps)} — {ACTIONS[action_id].label}"
                )
                self._execute_action(host, ACTIONS[action_id])
                with self.state.lock:
                    if self.state.cancel_requested:
                        self.state.phase = "Stopped safely"
                        return

            with self.state.lock:
                self.state.phase = f"{operation_label} complete"
        except (OSError, subprocess.TimeoutExpired, RuntimeError) as exc:
            with self.state.lock:
                self.state.error = str(exc)
                self.state.phase = "Command failed"
                if not self.state.simulate:
                    self.state.host = None
        finally:
            with self.state.lock:
                self.state.running = False
                self.state.action = None
                self.state.operation_id = None
                self.state.step = None
                self.state.process = None
                self.state.cancel_requested = False
                self.state.pid_file = None

    def _execute_action(self, host, info):
        if self.state.simulate:
            self._simulate_action(info)
            return

        if info.executor == "gesture":
            self._execute_gesture(host, info)
        elif info.executor == "sound":
            self._execute_sound(host, info)
        elif info.executor == "led":
            self._execute_led(host, info)
        else:
            raise RuntimeError(f"Unsupported executor: {info.executor}")

    def _simulate_action(self, info):
        duration = min(info.duration or 1.0, 1.0)
        started = time.monotonic()
        self.state.add_log(f"[simulation] {info.executor}: {info.label}")
        while time.monotonic() - started < duration:
            with self.state.lock:
                if self.state.cancel_requested:
                    self.state.phase = "Simulation stopped"
                    return
            time.sleep(0.05)
        self.state.add_log(f"[simulation] {info.label} complete")

    def _run_remote_process(self, host, remote_command):
        with self.state.lock:
            if self.state.cancel_requested:
                return
        process = subprocess.Popen(
            ["ssh", *SSH_OPTIONS, host, remote_command],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        with self.state.lock:
            self.state.process = process

        assert process.stdout is not None
        for line in process.stdout:
            self.state.add_log(line)
        return_code = process.wait()
        with self.state.lock:
            stopped = self.state.cancel_requested
            self.state.process = None
        if return_code != 0 and not stopped:
            raise RuntimeError(f"Robot command exited with status {return_code}")

    def _deploy(self, host, *paths):
        deploy = subprocess.run(
            ["scp", "-q", *SSH_OPTIONS, *(str(path) for path in paths), f"{host}:/tmp/"],
            capture_output=True,
            text=True,
            timeout=12,
        )
        if deploy.returncode != 0:
            detail = deploy.stderr.strip() or "copy failed"
            raise RuntimeError(f"Could not send action files: {detail}")

    def _execute_gesture(self, host, info):
        self.state.add_log(f"Connecting through {host}")
        movement_path = action_resource_path(info)
        self._deploy(host, RUNNER, movement_path)
        with self.state.lock:
            if self.state.cancel_requested:
                return
            pid_file = self.state.pid_file
        remote_command = (
            'export PATH="$HOME/.local/bin:$PATH"; '
            f'exec "$HOME/.local/bin/uv" run --no-sync --project "$HOME/bbos" '
            f'python {REMOTE_RUNNER} {shlex.quote("/tmp/" + info.resource)} '
            f'--name {shlex.quote(info.label)} --speed 0.6 --execute '
            f'--pid-file {pid_file}'
        )
        self._run_remote_process(host, remote_command)

    def _execute_sound(self, host, info):
        sound_path = action_resource_path(info)
        self._deploy(host, EFFECT_RUNNER, sound_path)
        with self.state.lock:
            if self.state.cancel_requested:
                return
            pid_file = self.state.pid_file
        remote_command = (
            'export PATH="$HOME/.local/bin:$PATH"; '
            f'exec "$HOME/.local/bin/uv" run --no-sync --project "$HOME/bbos" '
            f'python {REMOTE_EFFECT_RUNNER} --pid-file {pid_file} sound '
            f'{shlex.quote("/tmp/" + info.resource)}'
        )
        self._run_remote_process(host, remote_command)

    def _execute_led(self, host, info):
        self._deploy(host, EFFECT_RUNNER)
        with self.state.lock:
            if self.state.cancel_requested:
                return
            pid_file = self.state.pid_file
        rgb = ",".join(str(value) for value in info.rgb)
        remote_command = (
            'export PATH="$HOME/.local/bin:$PATH"; '
            f'exec "$HOME/.local/bin/uv" run --no-sync --project "$HOME/bbos" '
            f'python {REMOTE_EFFECT_RUNNER} --pid-file {pid_file} led '
            f'--rgb {rgb} --pattern {info.pattern} --duration {info.duration}'
        )
        self._run_remote_process(host, remote_command)

    def set_lean(self, enabled):
        if not isinstance(enabled, bool):
            return False, "enabled must be true or false"
        with self.state.lock:
            if self.state.host is None:
                return False, "Robot is not connected; choose Reconnect"
            if self.state.lean_transition:
                return False, "Lean mode is already changing"
            if enabled == self.state.lean_enabled:
                mode = "Lean" if enabled else "Balance"
                return False, f"{mode} mode is already active"

            if self.state.simulate:
                self.state.lean_enabled = enabled
                self.state.lean_requested = enabled
                self.state.lean_phase = (
                    "Lean mode active at 4.0° (simulation)" if enabled
                    else "Balance mode (simulation)"
                )
                self.state.log.append(
                    "[lean] simulated lean enabled" if enabled
                    else "[lean] simulated balance restored"
                )
                return True, "Lean enabled" if enabled else "Balance restored"

            host = self.state.host
            if enabled:
                self._lean_counter += 1
                pid_file = f"/tmp/bracketbot-lean-{self._lean_counter}.pid"
                self.state.lean_pid_file = pid_file
                self.state.lean_requested = True
                self.state.lean_transition = True
                self.state.lean_phase = "Enabling 4.0° lean…"
            else:
                pid_file = self.state.lean_pid_file
                self.state.lean_requested = False
                self.state.lean_transition = True
                self.state.lean_phase = "Returning to balance mode…"

        if enabled:
            threading.Thread(
                target=self._run_lean,
                args=(host, pid_file),
                name="lean-control",
                daemon=True,
            ).start()
            return True, "Enabling lean mode"

        self._request_remote_stop(host, pid_file, "lean-stop")
        return True, "Returning to balance mode"

    def _run_lean(self, host, pid_file):
        return_code = None
        try:
            self._deploy(host, BASE_RUNNER)
            with self.state.lock:
                if not self.state.lean_requested:
                    self.state.lean_transition = False
                    self.state.lean_phase = "Balance mode"
                    return
            remote_command = (
                'export PATH="$HOME/.local/bin:$PATH"; '
                f'exec "$HOME/.local/bin/uv" run --no-sync --project "$HOME/bbos" '
                f'python {REMOTE_BASE_RUNNER} --angle 4.0 --pid-file {pid_file}'
            )
            process = subprocess.Popen(
                ["ssh", *SSH_OPTIONS, host, remote_command],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            with self.state.lock:
                self.state.lean_process = process
            assert process.stdout is not None
            for line in process.stdout:
                self.state.add_lean_log(line)
                if "lean active" in line:
                    with self.state.lock:
                        self.state.lean_enabled = True
                        self.state.lean_transition = False
            return_code = process.wait()
        except (OSError, subprocess.TimeoutExpired, RuntimeError) as exc:
            with self.state.lock:
                self.state.error = f"Lean control failed: {exc}"
        finally:
            with self.state.lock:
                unexpected = self.state.lean_requested
                if unexpected and self.state.error is None:
                    self.state.error = f"Lean control exited with status {return_code}"
                self.state.lean_enabled = False
                self.state.lean_requested = False
                self.state.lean_transition = False
                self.state.lean_process = None
                self.state.lean_pid_file = None
                self.state.lean_phase = (
                    "Lean control stopped unexpectedly" if unexpected else "Balance mode"
                )

    def _request_remote_stop(self, host, pid_file, thread_name):
        if not host or not pid_file:
            return

        def request_stop():
            subprocess.run(
                [
                    "ssh", *SSH_OPTIONS, host,
                    "for attempt in 1 2 3 4 5 6 7 8 9 10; do "
                    f"if test -s {pid_file}; then "
                    f"xargs kill -INT < {pid_file}; exit 0; fi; "
                    "sleep 0.25; done; exit 1",
                ],
                capture_output=True,
                timeout=6,
            )

        threading.Thread(target=request_stop, name=thread_name, daemon=True).start()

    def stop(self):
        with self.state.lock:
            action_running = self.state.running
            lean_running = (
                self.state.lean_enabled
                or self.state.lean_requested
                or self.state.lean_transition
            )
            if not action_running and not lean_running:
                return False, "No action is running"
            host = self.state.host
            pid_file = self.state.pid_file
            lean_pid_file = self.state.lean_pid_file
            if action_running:
                self.state.cancel_requested = True
                self.state.phase = "Stop requested — finishing safely…"
            if lean_running:
                self.state.lean_requested = False
                self.state.lean_transition = not self.state.simulate
                self.state.lean_phase = (
                    "Returning to balance mode…" if not self.state.simulate
                    else "Balance mode (simulation)"
                )
                if self.state.simulate:
                    self.state.lean_enabled = False

        if not self.state.simulate:
            if action_running:
                self._request_remote_stop(host, pid_file, "action-stop")
            if lean_running:
                self._request_remote_stop(host, lean_pid_file, "lean-stop")
        return True, "Stop requested"


PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="color-scheme" content="dark light">
<title>BracketBot controls</title>
<style>
:root { --bg:#101319; --card:#1a2029; --text:#f7f8fa; --muted:#bac4d1;
  --line:#3c4858; --accent:#63d8ff; --good:#72e6a1; --warn:#ffd166; --danger:#ff6577; }
* { box-sizing:border-box; }
body { margin:0; min-height:100vh; background:var(--bg); color:var(--text);
  font:18px/1.5 system-ui,-apple-system,sans-serif; }
main { width:min(760px,calc(100% - 28px)); margin:0 auto; padding:28px 0 48px; }
h1 { margin:0 0 4px; font-size:clamp(1.8rem,5vw,2.6rem); }
.intro { color:var(--muted); margin:0 0 22px; }
.status { border:2px solid var(--line); background:var(--card); border-radius:16px;
  padding:16px 18px; margin-bottom:20px; }
.status-line { display:flex; align-items:center; gap:12px; font-weight:750; }
.dot { width:14px; height:14px; flex:none; border-radius:50%; background:var(--warn); }
.dot.good { background:var(--good); } .dot.bad { background:var(--danger); }
#detail, #base-detail { color:var(--muted); margin:5px 0 0 26px; }
.group { margin:22px 0 0; }
.group h2 { margin:0 0 10px; font-size:1.05rem; color:var(--muted); }
.grid { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:14px; }
button { min-height:94px; border:2px solid var(--line); border-radius:16px; padding:14px;
  text-align:left; background:var(--card); color:var(--text); font:inherit; cursor:pointer; }
button:hover:not(:disabled) { border-color:var(--accent); transform:translateY(-1px); }
button:focus-visible { outline:4px solid var(--accent); outline-offset:3px; }
button:disabled { opacity:.48; cursor:not-allowed; }
.label { display:block; font-size:1.15rem; font-weight:800; }
.desc { display:block; color:var(--muted); font-size:.9rem; margin-top:3px; }
.key { float:right; border:1px solid var(--line); border-radius:6px; padding:1px 7px;
  color:var(--muted); font-size:.8rem; }
.controls { display:grid; grid-template-columns:repeat(3,1fr); gap:14px; margin-top:14px; }
.secondary { min-height:58px; text-align:center; }
.lean-active { background:#173c35; border-color:var(--good); }
.stop { min-height:58px; text-align:center; background:#491923; border-color:var(--danger); font-weight:850; }
.log-wrap { margin-top:22px; }
.log-wrap summary { cursor:pointer; color:var(--muted); }
pre { white-space:pre-wrap; overflow-wrap:anywhere; max-height:220px; overflow:auto;
  background:#090b0f; padding:12px; border-radius:10px; color:#d7e0ea; font-size:.78rem; }
.error { color:#ff9ca8; font-weight:700; margin-top:10px; }
.sr-only { position:absolute; width:1px; height:1px; padding:0; margin:-1px;
  overflow:hidden; clip:rect(0,0,0,0); white-space:nowrap; border:0; }
@media (max-width:560px) { .grid, .controls { grid-template-columns:1fr; } main { padding-top:18px; } }
@media (prefers-reduced-motion:reduce) { * { transition:none!important; scroll-behavior:auto!important; }
  button:hover:not(:disabled) { transform:none; } }
@media (prefers-contrast:more) { :root { --line:#eef3f8; --muted:#eef3f8; } }
</style>
</head>
<body>
<main>
  <h1>BracketBot controls</h1>
  <p class="intro">Run an allowlisted action or a deterministic multi-step routine.</p>
  <section class="status" aria-labelledby="connection-title">
    <h2 id="connection-title" class="sr-only">Robot connection</h2>
    <div class="status-line"><span id="dot" class="dot" aria-hidden="true"></span><span id="status">Connecting…</span></div>
    <p id="detail" aria-live="polite">Checking botwifi and bot</p>
    <p id="base-detail">Base: balance mode</p>
    <p id="error" class="error" role="alert" hidden></p>
  </section>
  <div id="catalog" aria-live="polite"></div>
  <div class="controls">
    <button id="reconnect" class="secondary">Reconnect</button>
    <button id="lean" class="secondary"><span class="key">Z</span>Enable lean</button>
    <button id="stop" class="stop" disabled><span class="key">Esc</span>Stop action</button>
  </div>
  <details class="log-wrap"><summary>Technical details</summary><pre id="log">No activity yet.</pre></details>
</main>
<script>
const statusEl=document.getElementById('status'), detail=document.getElementById('detail');
const dot=document.getElementById('dot'), error=document.getElementById('error');
const stop=document.getElementById('stop'), reconnect=document.getElementById('reconnect');
const lean=document.getElementById('lean'), baseDetail=document.getElementById('base-detail');
const log=document.getElementById('log'), catalog=document.getElementById('catalog');
let current={}, buttons=[], rendered=false, previewAudio=null;
async function post(path, body={}) {
  const response=await fetch(path,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
  return response.json();
}
function makeButton(item, type) {
  const button=document.createElement('button');
  button.dataset[type]=item.id;
  if(item.key){ const key=document.createElement('span'); key.className='key'; key.textContent=item.key; button.append(key); }
  const label=document.createElement('span'); label.className='label'; label.textContent=item.label; button.append(label);
  const desc=document.createElement('span'); desc.className='desc'; desc.textContent=item.description; button.append(desc);
  button.addEventListener('click',()=>{
    if(type==='action'&&current.mode==='simulation'&&item.preview_url) {
      if(previewAudio) previewAudio.pause();
      previewAudio=new Audio(item.preview_url); previewAudio.play().catch(()=>{});
    }
    post(type==='action'?'/api/run':'/api/routine',{[type]:item.id}).then(refresh);
  });
  return button;
}
function renderCatalog() {
  if(rendered) return;
  const groups=new Map();
  for(const action of current.actions) {
    if(!groups.has(action.category)) groups.set(action.category,[]);
    groups.get(action.category).push(action);
  }
  for(const [name,items] of groups) {
    const section=document.createElement('section'); section.className='group'; section.setAttribute('aria-label',name);
    const title=document.createElement('h2'); title.textContent=name; section.append(title);
    const grid=document.createElement('div'); grid.className='grid';
    items.forEach(item=>grid.append(makeButton(item,'action'))); section.append(grid); catalog.append(section);
  }
  const section=document.createElement('section'); section.className='group'; section.setAttribute('aria-label','Routines');
  const title=document.createElement('h2'); title.textContent='Routines'; section.append(title);
  const grid=document.createElement('div'); grid.className='grid';
  current.routines.forEach(item=>grid.append(makeButton(item,'routine'))); section.append(grid); catalog.append(section);
  buttons=[...catalog.querySelectorAll('button')]; rendered=true;
}
async function refresh() {
  try {
    current=await (await fetch('/api/status',{cache:'no-store'})).json();
    renderCatalog();
    const ready=current.connected&&!current.running&&!current.checking;
    buttons.forEach(b=>b.disabled=!ready); stop.disabled=!(current.running||current.lean_enabled||current.lean_transition);
    reconnect.disabled=current.running||current.checking||current.lean_enabled||current.lean_transition;
    lean.disabled=!current.connected||current.lean_transition;
    lean.className='secondary '+(current.lean_enabled?'lean-active':'');
    lean.setAttribute('aria-pressed',String(current.lean_enabled));
    lean.innerHTML=`<span class="key">Z</span>${current.lean_transition?'Changing base mode…':current.lean_enabled?'Return to balance':'Enable lean'}`;
    baseDetail.textContent=`Base: ${current.lean_phase}`;
    dot.className='dot '+(current.connected?'good':current.checking?'':'bad');
    statusEl.textContent=current.running?`${current.action} in progress`:current.mode==='simulation'?'Local simulation':current.connected?`Connected: ${current.host}`:current.checking?'Connecting…':'Robot offline';
    detail.textContent=current.phase;
    error.hidden=!current.error; error.textContent=current.error||'';
    log.textContent=current.log.length?current.log.join('\n'):'No activity yet.';
  } catch (_) {
    dot.className='dot bad'; statusEl.textContent='Dashboard connection lost';
    detail.textContent='Reload this page to reconnect.';
  }
}
reconnect.addEventListener('click',()=>post('/api/discover').then(refresh));
lean.addEventListener('click',()=>post('/api/lean',{enabled:!current.lean_enabled}).then(refresh));
stop.addEventListener('click',()=>{
  if(previewAudio){ previewAudio.pause(); previewAudio.currentTime=0; }
  post('/api/stop').then(refresh);
});
document.addEventListener('keydown',event=>{
  if(event.repeat||event.target.matches('input,textarea,select')) return;
  if(event.key==='Escape'&&(current.running||current.lean_enabled||current.lean_transition)){ event.preventDefault(); stop.click(); return; }
  if(event.key.toLowerCase()==='z'&&!lean.disabled){ event.preventDefault(); lean.click(); return; }
  const action=current.actions?.find(item=>item.key===event.key);
  const routine=current.routines?.find(item=>item.key===event.key);
  const selector=action?`[data-action="${action.id}"]`:routine?`[data-routine="${routine.id}"]`:null;
  if(selector) { const button=document.querySelector(selector); if(!button.disabled) button.click(); }
});
setInterval(refresh,500); refresh();
</script>
</body>
</html>"""


class Handler(BaseHTTPRequestHandler):
    controller = None

    def log_message(self, format, *args):
        return

    def _send(self, body, status=HTTPStatus.OK, content_type="application/json; charset=utf-8"):
        if isinstance(body, (dict, list)):
            body = json.dumps(body).encode()
        elif isinstance(body, str):
            body = body.encode()
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy", "default-src 'self'; style-src 'unsafe-inline'; script-src 'unsafe-inline'")
        self.end_headers()
        self.wfile.write(body)

    def _json_body(self):
        length = min(int(self.headers.get("Content-Length", "0")), 4096)
        try:
            return json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            return {}

    def do_GET(self):
        if self.path == "/":
            self._send(PAGE, content_type="text/html; charset=utf-8")
        elif self.path == "/api/status":
            self._send(self.controller.state.snapshot())
        elif self.path.startswith("/api/audio/"):
            action_id = self.path.removeprefix("/api/audio/")
            action = ACTIONS.get(action_id)
            if action is None or action.executor != "sound":
                self._send({"error": "Not found"}, HTTPStatus.NOT_FOUND)
                return
            try:
                audio = action_resource_path(action).read_bytes()
            except OSError:
                self._send({"error": "Audio unavailable"}, HTTPStatus.NOT_FOUND)
                return
            self._send(audio, content_type="audio/wav")
        else:
            self._send({"error": "Not found"}, HTTPStatus.NOT_FOUND)

    def do_POST(self):
        if self.path == "/api/discover":
            self.controller.discover_async()
            self._send({"ok": True, "message": "Connection check started"})
        elif self.path == "/api/run":
            ok, message = self.controller.run_action(self._json_body().get("action"))
            self._send({"ok": ok, "message": message}, HTTPStatus.ACCEPTED if ok else HTTPStatus.CONFLICT)
        elif self.path == "/api/routine":
            ok, message = self.controller.run_routine(self._json_body().get("routine"))
            self._send({"ok": ok, "message": message}, HTTPStatus.ACCEPTED if ok else HTTPStatus.CONFLICT)
        elif self.path == "/api/lean":
            ok, message = self.controller.set_lean(self._json_body().get("enabled"))
            self._send({"ok": ok, "message": message}, HTTPStatus.ACCEPTED if ok else HTTPStatus.CONFLICT)
        elif self.path == "/api/stop":
            ok, message = self.controller.stop()
            self._send({"ok": ok, "message": message}, HTTPStatus.ACCEPTED if ok else HTTPStatus.CONFLICT)
        else:
            self._send({"error": "Not found"}, HTTPStatus.NOT_FOUND)


def parse_hosts(value):
    hosts = tuple(dict.fromkeys(part.strip() for part in value.split(",") if part.strip()))
    if not hosts:
        raise argparse.ArgumentTypeError("provide at least one SSH host")
    return hosts


def main():
    parser = argparse.ArgumentParser(description="Accessible local BracketBot command dashboard")
    parser.add_argument("--bind", default="127.0.0.1", help="listen address (default: localhost only)")
    parser.add_argument("--port", type=int, default=8020)
    parser.add_argument("--ssh-hosts", type=parse_hosts, default=("botwifi", "bot"),
                        help="comma-separated SSH aliases in priority order")
    parser.add_argument(
        "--simulate",
        action="store_true",
        help="exercise actions and routines locally without SSH or robot hardware",
    )
    args = parser.parse_args()

    controller = RobotController(args.ssh_hosts, simulate=args.simulate)
    Handler.controller = controller
    server = ThreadingHTTPServer((args.bind, args.port), Handler)
    controller.start_monitor()
    print(f"BracketBot dashboard: http://{args.bind}:{args.port}", flush=True)
    if args.simulate:
        print("Mode: local simulation (SSH and BBOS disabled)", flush=True)
    else:
        print(f"SSH candidates: {', '.join(args.ssh_hosts)}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        controller._stop_monitor.set()
        controller.stop()
        deadline = time.monotonic() + 8.0
        while time.monotonic() < deadline:
            state = controller.state.snapshot()
            if not state["running"] and not state["lean_enabled"] and not state["lean_transition"]:
                break
            time.sleep(0.1)
        server.server_close()


if __name__ == "__main__":
    main()
