"""Local, accessible web dashboard for allowlisted BracketBot actions.

The server binds to localhost by default. It discovers the first reachable SSH
route (the configured Wi-Fi alias, the robot's mDNS name, then USB), copies
small safety-focused runners to the robot, and executes only actions and
routines from the fixed allowlists below.

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
import os
from pathlib import Path
import shlex
import subprocess
import sys
import threading
import time
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "gesture_test.py"
EFFECT_RUNNER = ROOT / "scripts" / "robot_effect.py"
BASE_RUNNER = ROOT / "scripts" / "robot_base_mode.py"
GREETER_ACTION_RUNNER = ROOT / "scripts" / "greeter_action.py"
CAMERA_POINT_RUNNER = ROOT / "scripts" / "camera_point_motion.py"
TABLE_REST_RUNNER = ROOT / "scripts" / "table_rest.py"
REMOTE_RUNNER = "/tmp/gesture_test.py"
REMOTE_EFFECT_RUNNER = "/tmp/robot_effect.py"
REMOTE_BASE_RUNNER = "/tmp/robot_base_mode.py"
REMOTE_GREETER_ACTION_RUNNER = "/tmp/greeter_action.py"
REMOTE_TABLE_REST_RUNNER = "/tmp/table_rest.py"
SSH_OPTIONS = (
    "-o", "BatchMode=yes",
    "-o", "ConnectTimeout=3",
    "-o", "ConnectionAttempts=1",
    "-o", "ControlMaster=auto",
    "-o", "ControlPersist=90",
    "-o", "ControlPath=/tmp/bracketbot-dashboard-%C",
    "-o", "ServerAliveInterval=2",
    "-o", "ServerAliveCountMax=2",
)
# Discovery commands must not create a persistent multiplexing master. A new
# master inherits subprocess.run's capture pipes and keeps them open for the
# ControlPersist window, making a successful probe look like a timeout.
SSH_PROBE_OPTIONS = (
    *SSH_OPTIONS,
    "-o", "ControlMaster=no",
    "-o", "ControlPath=none",
)
# A .local name may resolve to several IPv6 addresses. SSH applies its
# ConnectTimeout to each address, so the outer probe must allow more than one
# attempt before deciding that hotspot discovery failed.
SSH_PROBE_TIMEOUT = 12
DEFAULT_SSH_HOSTS = (
    "botwifi",
    "bracketbot@bracketbot-184.local",
    "bot",
)


class RobotConnectionError(RuntimeError):
    """The SSH transport failed, as opposed to an action being safely rejected."""

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
    _action("salute", "Salute", "Extend the left hand, salute, then wave", "Gestures", "gesture",
            channels=("left-arm",), key="s", resource="salute.json", risk="motion"),
    _action("point-person", "Point at person", "Use the camera to point at the primary visible person",
            "Gestures", "camera-gesture", channels=("camera", "left-arm", "right-arm"),
            key="o", resource="point", risk="motion"),
    _action("dance", "Dance", "Recorded two-arm dance", "Gestures", "gesture",
            channels=("left-arm", "right-arm"), key="d", resource="dance.json",
            source="bbapps/mimic/recordings/dance.json", risk="motion"),
    _action("table-rest", "Place arms on table",
            "Detect the tabletop and leave both arms resting there for the next action",
            "Positioning", "table-rest", channels=("depth-camera", "left-arm", "right-arm"),
            key="r", risk="contact-motion"),
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
        if action.executor in {"gesture", "sound", "camera-gesture"} and not action.resource:
            raise RuntimeError(f"{action.id} requires an allowlisted resource")
        if action.executor == "led" and (
            action.rgb is None or action.pattern not in {"solid", "pulse", "blink"}
            or action.duration is None
        ):
            raise RuntimeError(f"{action.id} requires RGB, pattern, and duration")
        if action.executor not in {
            "gesture", "sound", "led", "camera-gesture", "table-rest"
        }:
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


def action_bundle_paths():
    """Files preloaded once per connection to keep button dispatch fast."""
    paths = [
        RUNNER,
        EFFECT_RUNNER,
        BASE_RUNNER,
        GREETER_ACTION_RUNNER,
        CAMERA_POINT_RUNNER,
        TABLE_REST_RUNNER,
    ]
    paths.extend(
        action_resource_path(action)
        for action in ACTION_LIST
        if action.executor in {"gesture", "sound"}
    )
    return tuple(dict.fromkeys(paths))


def remote_python_command(script, *args):
    """Prefer the already-created BBOS venv, with uv as a portable fallback."""
    payload = " ".join(shlex.quote(str(value)) for value in (script, *args))
    return (
        'export PATH="$HOME/.local/bin:$PATH"; '
        'if test -x "$HOME/bbos/.venv/bin/python"; then '
        f'exec "$HOME/bbos/.venv/bin/python" {payload}; '
        "else "
        f'exec "$HOME/.local/bin/uv" run --no-sync --project "$HOME/bbos" '
        f'python {payload}; fi'
    )


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
        self.step_started_at = None
        self.last_dispatch_ms = None
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
        self.stop_lean_after_action = False
        self.server_id = uuid.uuid4().hex

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
                "last_dispatch_ms": self.last_dispatch_ms,
                "phase": self.phase,
                "error": self.error,
                "log": list(self.log[-240:]),
                "candidates": list(self.ssh_hosts),
                "mode": "simulation" if self.simulate else "robot",
                "server_id": self.server_id,
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
            del self.log[:-500]
            self.phase = line

    def add_lean_log(self, line):
        line = line.strip()
        if not line:
            return
        with self.lock:
            self.log.append(f"[lean] {line}")
            del self.log[:-500]
            self.lean_phase = line


class RobotController:
    def __init__(self, ssh_hosts, simulate=False):
        self.state = DashboardState(ssh_hosts, simulate=simulate)
        self._discover_lock = threading.Lock()
        self._stop_monitor = threading.Event()
        self._operation_counter = 0
        self._lean_counter = 0
        self._deploy_lock = threading.Lock()
        self._deployed_files = set()

    @staticmethod
    def _probe(host):
        command = [
            "ssh", *SSH_PROBE_OPTIONS, host,
            'test -x "$HOME/.local/bin/uv" && '
            'test -d "$HOME/bbos"',
        ]
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=SSH_PROBE_TIMEOUT,
            )
            return result.returncode == 0
        except (OSError, subprocess.TimeoutExpired):
            return False

    def _restore_orphaned_base(self, host):
        """Return an unowned lean runner to balance after a dashboard restart."""
        pattern = r"^[^ ]*python[^ ]* /tmp/robot_base_mode\.py( |$)"
        remote = (
            f"if pgrep -f '{pattern}' >/dev/null; then "
            f"pkill -INT -f '{pattern}'; "
            "for attempt in 1 2 3 4 5 6 7 8 9 10 11 12; do "
            f"pgrep -f '{pattern}' >/dev/null || "
            "{ echo '[base] restored orphaned lean to balance'; exit 0; }; "
            "sleep 0.25; done; exit 1; fi"
        )
        result = subprocess.run(
            ["ssh", *SSH_OPTIONS, host, remote],
            capture_output=True,
            text=True,
            timeout=6,
        )
        if result.returncode != 0:
            error_type = RobotConnectionError if result.returncode == 255 else RuntimeError
            raise error_type("Could not safely restore an orphaned lean process")
        if result.stdout.strip():
            self.state.add_lean_log(result.stdout)

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
            if selected:
                with self.state.lock:
                    self.state.host = selected
                    self.state.phase = f"Connected through {selected} — checking base state…"
                self._invalidate_deploy_cache(selected)
                started = time.monotonic()
                try:
                    self._restore_orphaned_base(selected)
                    self._deploy(selected, *action_bundle_paths())
                except (OSError, subprocess.TimeoutExpired, RuntimeError) as exc:
                    with self.state.lock:
                        self.state.host = None
                        self.state.phase = "Robot action preload failed"
                        self.state.error = str(exc)
                else:
                    elapsed = time.monotonic() - started
                    with self.state.lock:
                        self.state.phase = (
                            f"Ready — {len(action_bundle_paths())} action files cached "
                            f"in {elapsed:.1f}s"
                        )
            else:
                with self.state.lock:
                    self.state.host = None
                    self.state.phase = "Robot not found"
                    self.state.error = (
                        "No robot SSH route is reachable. Check the hotspot/Wi-Fi or "
                        "USB connection, then choose Reconnect."
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
                    if host is not None:
                        self._invalidate_deploy_cache(host)
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
                    self.state.step_started_at = time.monotonic()
                    self.state.last_dispatch_ms = None
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
            disconnected_host = None
            with self.state.lock:
                self.state.error = str(exc)
                self.state.phase = "Command failed"
                if not self.state.simulate and isinstance(
                    exc,
                    (OSError, subprocess.TimeoutExpired, RobotConnectionError),
                ):
                    disconnected_host = self.state.host
                    self.state.host = None
            if disconnected_host is not None:
                self._invalidate_deploy_cache(disconnected_host)
        finally:
            restore_balance = False
            with self.state.lock:
                self.state.running = False
                self.state.action = None
                self.state.operation_id = None
                self.state.step = None
                self.state.step_started_at = None
                self.state.process = None
                self.state.cancel_requested = False
                self.state.pid_file = None
                restore_balance = self.state.stop_lean_after_action
                self.state.stop_lean_after_action = False
            if restore_balance:
                self.set_lean(False)

    def _execute_action(self, host, info):
        if self.state.simulate:
            self._simulate_action(info)
            return

        if info.executor == "gesture":
            self._execute_gesture(host, info)
        elif info.executor == "camera-gesture":
            self._execute_camera_gesture(host, info)
        elif info.executor == "table-rest":
            self._execute_table_rest(host)
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
        first_line = True
        output_tail = []
        reported_failure = False
        for line in process.stdout:
            if first_line:
                with self.state.lock:
                    started = self.state.step_started_at
                    if started is not None:
                        self.state.last_dispatch_ms = round(
                            (time.monotonic() - started) * 1000
                        )
                        dispatch_ms = self.state.last_dispatch_ms
                    else:
                        dispatch_ms = None
                if dispatch_ms is not None:
                    self.state.add_log(f"[latency] runner ready in {dispatch_ms} ms")
                first_line = False
            stripped = line.strip()
            if stripped:
                output_tail.append(stripped)
                del output_tail[:-12]
                if (
                    "Traceback (most recent call last)" in stripped
                    or "NOT SAFE TO RUN" in stripped
                    or "[table][fatal]" in stripped
                ):
                    reported_failure = True
            self.state.add_log(line)
        return_code = process.wait()
        with self.state.lock:
            stopped = self.state.cancel_requested
            self.state.process = None
        self.state.add_log(f"[remote] runner exited status={return_code}")
        if (return_code != 0 or reported_failure) and not stopped:
            error_type = RobotConnectionError if return_code == 255 else RuntimeError
            detail = next(
                (
                    line
                    for line in reversed(output_tail)
                    if "[fatal]" in line or "NOT SAFE TO RUN" in line
                ),
                output_tail[-1] if output_tail else "no runner output",
            )
            raise error_type(
                f"Robot command failed (status {return_code}): {detail}"
            )

    def _deploy(self, host, *paths):
        with self._deploy_lock:
            signatures = []
            for path in paths:
                path = Path(path)
                stat = path.stat()
                signature = (host, str(path.resolve()), stat.st_mtime_ns, stat.st_size)
                signatures.append((path, signature))
            missing = [path for path, signature in signatures if signature not in self._deployed_files]
            if not missing:
                return False

            deploy = subprocess.run(
                ["scp", "-q", *SSH_OPTIONS, *(str(path) for path in missing), f"{host}:/tmp/"],
                capture_output=True,
                text=True,
                timeout=20,
            )
            if deploy.returncode != 0:
                detail = deploy.stderr.strip() or "copy failed"
                error_type = RobotConnectionError if deploy.returncode == 255 else RuntimeError
                raise error_type(f"Could not preload action files: {detail}")
            self._deployed_files.update(signature for _, signature in signatures)
            return True

    def _invalidate_deploy_cache(self, host):
        with self._deploy_lock:
            self._deployed_files = {
                signature for signature in self._deployed_files if signature[0] != host
            }

    def _execute_gesture(self, host, info):
        self.state.add_log(f"Connecting through {host}")
        movement_path = action_resource_path(info)
        self._deploy(host, RUNNER, movement_path)
        with self.state.lock:
            if self.state.cancel_requested:
                return
            pid_file = self.state.pid_file
        remote_command = remote_python_command(
            REMOTE_RUNNER,
            "/tmp/" + info.resource,
            "--name", info.label,
            "--speed", "0.6",
            "--execute",
            "--pid-file", pid_file,
        )
        self._run_remote_process(host, remote_command)

    def _execute_sound(self, host, info):
        sound_path = action_resource_path(info)
        self._deploy(host, EFFECT_RUNNER, sound_path)
        with self.state.lock:
            if self.state.cancel_requested:
                return
            pid_file = self.state.pid_file
        remote_command = remote_python_command(
            REMOTE_EFFECT_RUNNER,
            "--pid-file", pid_file,
            "sound", "/tmp/" + info.resource,
        )
        self._run_remote_process(host, remote_command)

    def _execute_camera_gesture(self, host, info):
        self.state.add_log(f"Connecting to the robot camera through {host}")
        self._deploy(host, GREETER_ACTION_RUNNER, CAMERA_POINT_RUNNER)
        with self.state.lock:
            if self.state.cancel_requested:
                return
            pid_file = self.state.pid_file
        remote_command = remote_python_command(
            REMOTE_GREETER_ACTION_RUNNER,
            info.resource,
            "--pid-file", pid_file,
        )
        self._run_remote_process(host, remote_command)

    def _execute_table_rest(self, host):
        self.state.add_log("Scanning for a reachable tabletop with the depth camera")
        self._deploy(host, TABLE_REST_RUNNER)
        with self.state.lock:
            if self.state.cancel_requested:
                return
            pid_file = self.state.pid_file
        remote_command = remote_python_command(
            REMOTE_TABLE_REST_RUNNER,
            "--pid-file", pid_file,
        )
        self._run_remote_process(host, remote_command)

    def _execute_led(self, host, info):
        self._deploy(host, EFFECT_RUNNER)
        with self.state.lock:
            if self.state.cancel_requested:
                return
            pid_file = self.state.pid_file
        rgb = ",".join(str(value) for value in info.rgb)
        remote_command = remote_python_command(
            REMOTE_EFFECT_RUNNER,
            "--pid-file", pid_file,
            "led", "--rgb", rgb,
            "--pattern", info.pattern,
            "--duration", info.duration,
        )
        self._run_remote_process(host, remote_command)

    def set_lean(self, enabled):
        if not isinstance(enabled, bool):
            return False, "enabled must be true or false"
        with self.state.lock:
            if self.state.host is None:
                return False, "Robot is not connected; choose Reconnect"
            if self.state.running:
                return False, "Wait for the current arm action to finish"
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
            remote_command = remote_python_command(
                REMOTE_BASE_RUNNER,
                "--angle", "4.0",
                "--pid-file", pid_file,
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
            table_rest_running = self.state.operation_id == "table-rest"
            defer_lean_stop = action_running and table_rest_running and lean_running
            if action_running:
                self.state.cancel_requested = True
                self.state.phase = "Stop requested — finishing safely…"
            if defer_lean_stop:
                # Keep the base geometry stable while table-rest retraces its
                # checked arm path. _run_operation restores balance afterward.
                self.state.stop_lean_after_action = True
                self.state.lean_phase = "Lean held until arms return safely…"
            elif lean_running:
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
            if lean_running and not defer_lean_stop:
                self._request_remote_stop(host, lean_pid_file, "lean-stop")
        return True, "Stop requested"


PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="color-scheme" content="light">
<title>Baymax Control Center</title>
<style>
:root { --bg:#f4f1ef; --surface:#fff; --surface-soft:#faf8f7; --text:#252427;
  --muted:#777278; --line:#e6dfdc; --red:#e6383f; --red-dark:#b91f2d;
  --red-soft:#fff0f1; --good:#23a66f; --warn:#e19b23; --danger:#d92f3d;
  --shadow:0 18px 50px rgba(86,47,49,.10); }
* { box-sizing:border-box; }
body { margin:0; min-height:100vh; background:
  radial-gradient(circle at 88% 4%,rgba(230,56,63,.13),transparent 24rem),
  linear-gradient(180deg,#fbf9f8 0,var(--bg) 36rem); color:var(--text);
  font:17px/1.5 Inter,Avenir Next,ui-rounded,system-ui,-apple-system,sans-serif; }
body::before { content:""; position:fixed; inset:0; pointer-events:none; opacity:.35;
  background-image:radial-gradient(#c7bebb 1px,transparent 1px); background-size:24px 24px;
  mask-image:linear-gradient(to bottom,black,transparent 35rem); }
main { position:relative; width:min(1060px,calc(100% - 36px)); margin:0 auto; padding:34px 0 64px; }
.hero { display:flex; align-items:center; justify-content:space-between; gap:32px;
  min-height:210px; border-radius:36px; padding:30px 38px; margin-bottom:20px; overflow:hidden;
  background:linear-gradient(125deg,#d72e37 0,#f0494f 56%,#be1e2d 100%);
  box-shadow:0 24px 60px rgba(184,31,43,.23); color:#fff; position:relative; }
.hero::before,.hero::after { content:""; position:absolute; border:1px solid rgba(255,255,255,.18);
  border-radius:50%; width:330px; height:330px; right:-105px; top:-210px; }
.hero::after { width:230px; height:230px; right:120px; top:115px; }
.hero-copy { position:relative; z-index:1; max-width:620px; }
.eyebrow { margin:0 0 8px; font-size:.72rem; font-weight:850; letter-spacing:.18em;
  text-transform:uppercase; opacity:.78; }
h1 { margin:0; font-size:clamp(2.1rem,6vw,4rem); line-height:1; letter-spacing:-.055em; }
.intro { max-width:540px; margin:14px 0 0; color:rgba(255,255,255,.82); font-weight:600; }
.baymax-mark { width:154px; height:154px; flex:none; display:grid; place-items:center;
  border-radius:50%; background:linear-gradient(145deg,#fff,#e9e6e5); position:relative; z-index:1;
  box-shadow:inset -9px -12px 22px rgba(90,68,68,.14),0 20px 28px rgba(92,12,19,.22); }
.baymax-face { width:91px; height:28px; position:relative; }
.baymax-face::before,.baymax-face::after { content:""; position:absolute; top:7px; width:19px;
  height:19px; background:#1f2022; border-radius:50%; z-index:1; }
.baymax-face::before { left:0; } .baymax-face::after { right:0; }
.face-line { position:absolute; left:16px; right:16px; top:15px; height:3px; background:#1f2022; }
.status { display:grid; grid-template-columns:minmax(0,1fr) auto; gap:18px; align-items:center;
  border:1px solid var(--line); background:rgba(255,255,255,.92); border-radius:24px;
  padding:19px 22px; margin-bottom:26px; box-shadow:var(--shadow); backdrop-filter:blur(14px); }
.status-line { display:flex; align-items:center; gap:12px; font-weight:850; }
.status-copy { min-width:0; }
.status-tag { padding:8px 12px; border-radius:999px; background:var(--red-soft); color:var(--red-dark);
  font-size:.7rem; line-height:1; font-weight:900; letter-spacing:.12em; text-transform:uppercase; }
.dot { width:12px; height:12px; flex:none; border-radius:50%; background:var(--warn);
  box-shadow:0 0 0 5px rgba(225,155,35,.13); }
.dot.good { background:var(--good); } .dot.bad { background:var(--danger); }
#detail, #base-detail, #latency-detail { color:var(--muted); margin:4px 0 0 24px; font-size:.9rem; }
#catalog { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:18px; align-items:start; }
.group { margin:0; padding:20px; border:1px solid var(--line); border-radius:28px;
  background:rgba(255,255,255,.72); }
.group h2 { display:flex; align-items:center; gap:10px; margin:0 0 14px; color:#4d484c;
  font-size:.78rem; letter-spacing:.12em; text-transform:uppercase; }
.group-icon { width:28px; height:28px; display:grid; place-items:center; border-radius:9px;
  background:var(--red-soft); color:var(--red); font-size:.9rem; font-weight:900; }
.grid { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:10px; }
button { min-height:104px; border:1px solid var(--line); border-radius:20px; padding:15px;
  text-align:left; background:var(--surface); color:var(--text); font:inherit; cursor:pointer;
  box-shadow:0 5px 14px rgba(61,37,38,.05); transition:transform .18s ease,border-color .18s ease,box-shadow .18s ease; }
button:hover:not(:disabled) { border-color:#ef8d92; transform:translateY(-3px);
  box-shadow:0 12px 24px rgba(160,48,55,.12); }
button:focus-visible { outline:4px solid rgba(230,56,63,.28); outline-offset:3px; }
button:disabled { opacity:.48; cursor:not-allowed; filter:saturate(.4); }
.action-card { position:relative; overflow:hidden; }
.action-card::after { content:""; position:absolute; width:50px; height:50px; right:-25px; bottom:-25px;
  border-radius:50%; background:var(--red-soft); transition:transform .2s ease; }
.action-card:hover::after { transform:scale(1.35); }
.routine-card { background:linear-gradient(145deg,#fff,var(--red-soft)); }
.label { display:block; padding-right:22px; font-size:1rem; font-weight:850; letter-spacing:-.015em; }
.desc { display:block; color:var(--muted); font-size:.79rem; line-height:1.4; margin-top:5px; }
.key { float:right; border:1px solid #e8d9d9; border-radius:7px; padding:1px 7px;
  background:#faf5f4; color:#8b7b7c; font:800 .68rem/1.45 ui-monospace,SFMono-Regular,monospace; }
.controls { display:grid; grid-template-columns:repeat(2,1fr); gap:12px; margin-top:18px; }
.positioning { margin-top:24px; padding:20px; border:1px solid var(--line); border-radius:28px;
  background:linear-gradient(145deg,rgba(255,255,255,.88),rgba(255,240,241,.72)); }
.positioning h2 { display:flex; align-items:center; gap:10px; margin:0 0 5px; color:#4d484c;
  font-size:.78rem; letter-spacing:.12em; text-transform:uppercase; }
.positioning-copy { margin:0 0 14px 38px; color:var(--muted); font-size:.82rem; }
.positioning-grid { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:10px; }
.positioning-grid button { min-height:112px; }
.secondary,.stop { min-height:60px; border-radius:18px; text-align:center; font-size:.9rem; font-weight:800; }
.lean-active { background:#eaf8f2; border-color:#7dccac; color:#14744d; }
.stop { background:var(--red); border-color:var(--red); color:#fff; }
.stop .key { background:rgba(255,255,255,.14); border-color:rgba(255,255,255,.32); color:#fff; }
.log-wrap { margin-top:24px; border-top:1px solid var(--line); padding-top:18px; }
.log-wrap summary { width:max-content; cursor:pointer; color:var(--muted); font-size:.82rem; font-weight:700; }
pre { white-space:pre-wrap; overflow-wrap:anywhere; max-height:560px; overflow:auto;
  background:#292629; padding:14px; border-radius:14px; color:#f7eeee; font-size:.75rem; }
.error { color:var(--danger); font-weight:750; margin:8px 0 0 24px; }
.sr-only { position:absolute; width:1px; height:1px; padding:0; margin:-1px;
  overflow:hidden; clip:rect(0,0,0,0); white-space:nowrap; border:0; }
@media (max-width:820px) { #catalog { grid-template-columns:1fr; } }
@media (max-width:600px) {
  main { width:min(100% - 22px,1060px); padding-top:12px; }
  .hero { min-height:190px; padding:25px 23px; border-radius:28px; }
  .baymax-mark { width:94px; height:94px; position:absolute; right:20px; top:20px; opacity:.24; }
  .baymax-face { transform:scale(.68); }
  .hero-copy { padding-top:55px; } .intro { font-size:.88rem; }
  .status { grid-template-columns:1fr; border-radius:20px; padding:17px; }
  .status-tag { width:max-content; margin-left:24px; }
  .group { padding:15px; border-radius:22px; }
  .grid { grid-template-columns:1fr; }
  .controls,.positioning-grid { grid-template-columns:1fr; }
  button { min-height:92px; }
}
@media (prefers-reduced-motion:reduce) { * { transition:none!important; scroll-behavior:auto!important; }
  button:hover:not(:disabled) { transform:none; } }
@media (prefers-contrast:more) { :root { --line:#524b4c; --muted:#4f4849; } }
</style>
</head>
<body>
<main>
  <header class="hero">
    <div class="hero-copy">
      <p class="eyebrow">Personal healthcare companion</p>
      <h1>Baymax Control Center</h1>
      <p class="intro">Your friendly command station for safe gestures, expressions, and care routines.</p>
    </div>
    <div class="baymax-mark" aria-hidden="true"><div class="baymax-face"><span class="face-line"></span></div></div>
  </header>
  <section class="status" aria-labelledby="connection-title">
    <h2 id="connection-title" class="sr-only">Robot connection</h2>
    <div class="status-copy">
      <div class="status-line"><span id="dot" class="dot" aria-hidden="true"></span><span id="status">Connecting…</span></div>
      <p id="detail" aria-live="polite">Checking Wi-Fi, hotspot, and USB routes</p>
      <p id="base-detail">Base: balance mode</p>
      <p id="latency-detail">Dispatch: waiting for an action</p>
      <p id="error" class="error" role="alert" hidden></p>
    </div>
    <span class="status-tag">System status</span>
  </section>
  <div id="catalog" aria-live="polite"></div>
  <section class="positioning" aria-labelledby="positioning-title">
    <h2 id="positioning-title"><span class="group-icon" aria-hidden="true">↗</span>Positioning</h2>
    <p class="positioning-copy">Set the base first, then place the arms using the live depth view.</p>
    <div class="positioning-grid">
      <button id="lean" class="secondary"><span class="key">Z</span><span class="label">Lean forward</span><span class="desc">Hold a bounded 4° forward lean</span></button>
      <button id="table-rest" class="action-card" data-action="table-rest"><span class="key">R</span><span class="label">Place arms on table</span><span class="desc">Place, release control, then run another action from that pose</span></button>
    </div>
  </section>
  <div class="controls">
    <button id="reconnect" class="secondary">Reconnect</button>
    <button id="stop" class="stop" disabled><span class="key">Esc</span>Stop action</button>
  </div>
  <details class="log-wrap"><summary>Technical details</summary><pre id="log">No activity yet.</pre></details>
</main>
<script>
const statusEl=document.getElementById('status'), detail=document.getElementById('detail');
const dot=document.getElementById('dot'), error=document.getElementById('error');
const stop=document.getElementById('stop'), reconnect=document.getElementById('reconnect');
const lean=document.getElementById('lean'), baseDetail=document.getElementById('base-detail');
const tableRest=document.getElementById('table-rest');
const latencyDetail=document.getElementById('latency-detail');
const log=document.getElementById('log'), catalog=document.getElementById('catalog');
// A changed server ID means the Python process hot-reloaded; fetch the new page bundle.
let current={}, buttons=[], rendered=false, previewAudio=null, loadedServerId=null;
async function post(path, body={}) {
  const response=await fetch(path,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
  return response.json();
}
function makeButton(item, type) {
  const button=document.createElement('button');
  button.className=type==='routine'?'routine-card':'action-card';
  button.dataset[type]=item.id;
  if(item.key){ const key=document.createElement('span'); key.className='key'; key.textContent=item.key; button.append(key); }
  const label=document.createElement('span'); label.className='label'; label.textContent=item.label; button.append(label);
  const desc=document.createElement('span'); desc.className='desc'; desc.textContent=item.description; button.append(desc);
  if(type==='routine') {
    const specs=item.steps.map(id=>current.actions.find(action=>action.id===id)).filter(Boolean);
    const hasMotion=specs.some(spec=>spec.channels.some(channel=>channel.includes('arm')));
    const steps=document.createElement('span'); steps.className='desc';
    steps.textContent=`${hasMotion?'Includes arm motion':'No arm motion'} · ${specs.map(spec=>spec.label).join(' → ')}`;
    button.append(steps);
  }
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
  const groupIcons={Gestures:'✦',Lights:'◉',Sounds:'♫',Music:'♪',Routines:'＋'};
  const groups=new Map();
  for(const action of current.actions) {
    if(action.category==='Positioning') continue;
    if(!groups.has(action.category)) groups.set(action.category,[]);
    groups.get(action.category).push(action);
  }
  for(const [name,items] of groups) {
    const section=document.createElement('section'); section.className='group'; section.setAttribute('aria-label',name);
    const title=document.createElement('h2');
    const icon=document.createElement('span'); icon.className='group-icon'; icon.setAttribute('aria-hidden','true'); icon.textContent=groupIcons[name]||'•';
    title.append(icon,document.createTextNode(name)); section.append(title);
    const grid=document.createElement('div'); grid.className='grid';
    items.forEach(item=>grid.append(makeButton(item,'action'))); section.append(grid); catalog.append(section);
  }
  const section=document.createElement('section'); section.className='group'; section.setAttribute('aria-label','Routines');
  const title=document.createElement('h2');
  const icon=document.createElement('span'); icon.className='group-icon'; icon.setAttribute('aria-hidden','true'); icon.textContent=groupIcons.Routines;
  title.append(icon,document.createTextNode('Routines')); section.append(title);
  const grid=document.createElement('div'); grid.className='grid';
  current.routines.forEach(item=>grid.append(makeButton(item,'routine'))); section.append(grid); catalog.append(section);
  buttons=[...catalog.querySelectorAll('button'),tableRest]; rendered=true;
}
async function refresh() {
  try {
    const next=await (await fetch('/api/status',{cache:'no-store'})).json();
    if(loadedServerId&&next.server_id&&next.server_id!==loadedServerId) {
      window.location.reload(); return;
    }
    current=next; loadedServerId=next.server_id||loadedServerId;
    renderCatalog();
    const ready=current.connected&&!current.running&&!current.checking;
    buttons.forEach(b=>b.disabled=!ready); stop.disabled=!(current.running||current.lean_enabled||current.lean_transition);
    reconnect.disabled=current.running||current.checking||current.lean_enabled||current.lean_transition;
    lean.disabled=!current.connected||current.lean_transition||current.running;
    lean.className='secondary '+(current.lean_enabled?'lean-active':'');
    lean.setAttribute('aria-pressed',String(current.lean_enabled));
    lean.innerHTML=`<span class="key">Z</span><span class="label">${current.lean_transition?'Changing base mode…':current.lean_enabled?'Return to balance':'Lean forward'}</span><span class="desc">${current.lean_enabled?'Restore upright balance mode':'Hold a bounded 4° forward lean'}</span>`;
    baseDetail.textContent=`Base: ${current.lean_phase}`;
    latencyDetail.textContent=current.last_dispatch_ms==null?'Dispatch: waiting for runner':`Dispatch: runner ready in ${current.last_dispatch_ms} ms`;
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
tableRest.addEventListener('click',()=>post('/api/run',{action:'table-rest'}).then(refresh));
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
setInterval(refresh,250); refresh();
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


def file_signature(path):
    """Return a reload signature without keeping the source file open."""
    try:
        stat = Path(path).stat()
    except OSError:
        return None
    return stat.st_mtime_ns, stat.st_size


def start_source_reloader(server, controller, source, reload_requested, stop_event):
    """Restart the dashboard after an idle-safe source change.

    A live robot pose is never interrupted for developer convenience. If an
    action or lean mode is active, the restart remains pending until the robot
    has returned to an idle/balance state.
    """
    initial = file_signature(source)

    def watch():
        pending = None
        announced_wait = False
        while not stop_event.wait(0.35):
            signature = file_signature(source)
            if signature is None or signature == initial:
                continue
            if pending != signature:
                pending = signature
                announced_wait = False
                continue

            state = controller.state.snapshot()
            busy = (
                state["running"]
                or state["checking"]
                or state["lean_enabled"]
                or state["lean_transition"]
            )
            if busy:
                if not announced_wait:
                    print(
                        "[reload] source changed; waiting for actions and lean mode to stop",
                        flush=True,
                    )
                    announced_wait = True
                continue
            print("[reload] source changed; restarting dashboard", flush=True)
            reload_requested.set()
            server.shutdown()
            return

    threading.Thread(target=watch, name="dashboard-reloader", daemon=True).start()


def main():
    parser = argparse.ArgumentParser(description="Accessible local BracketBot command dashboard")
    parser.add_argument("--bind", default="127.0.0.1", help="listen address (default: localhost only)")
    parser.add_argument("--port", type=int, default=8020)
    parser.add_argument(
        "--ssh-hosts",
        type=parse_hosts,
        default=DEFAULT_SSH_HOSTS,
        help=(
            "comma-separated SSH routes in priority order "
            "(default: Wi-Fi alias, hotspot/mDNS, USB)"
        ),
    )
    parser.add_argument(
        "--simulate",
        action="store_true",
        help="exercise actions and routines locally without SSH or robot hardware",
    )
    parser.add_argument(
        "--no-reload",
        dest="reload",
        action="store_false",
        help="disable automatic restart when robot_dashboard.py changes",
    )
    parser.set_defaults(reload=True)
    args = parser.parse_args()

    controller = RobotController(args.ssh_hosts, simulate=args.simulate)
    Handler.controller = controller
    server = ThreadingHTTPServer((args.bind, args.port), Handler)
    server.daemon_threads = True
    reload_requested = threading.Event()
    reloader_stop = threading.Event()
    if args.reload:
        start_source_reloader(
            server,
            controller,
            Path(__file__).resolve(),
            reload_requested,
            reloader_stop,
        )
    controller.start_monitor()
    print(f"BracketBot dashboard: http://{args.bind}:{args.port}", flush=True)
    if args.reload:
        print(f"Hot reload: watching {Path(__file__).name}", flush=True)
    if args.simulate:
        print("Mode: local simulation (SSH and BBOS disabled)", flush=True)
    else:
        print(f"SSH candidates: {', '.join(args.ssh_hosts)}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        reloader_stop.set()
        controller._stop_monitor.set()
        controller.stop()
        deadline = time.monotonic() + 8.0
        while time.monotonic() < deadline:
            state = controller.state.snapshot()
            if not state["running"] and not state["lean_enabled"] and not state["lean_transition"]:
                break
            time.sleep(0.1)
        server.server_close()
    if reload_requested.is_set():
        os.execv(sys.executable, [sys.executable, *sys.argv])


if __name__ == "__main__":
    main()
