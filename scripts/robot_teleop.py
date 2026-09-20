"""Bare WASD drive test: stdin twist commands -> drive.ctrl. No perception, no PID.

Runs on the robot, fed by ``scripts/teleop_dashboard.py`` over ssh. One JSON
line per command: ``{"v": m/s, "w": rad/s}``. The base stops by itself if lines
stop arriving (a closed laptop lid, a dropped ssh) or the robot tips.

bbos allows one ``drive.ctrl`` writer, and while this runs it is this one. Other
programs on the robot that need a small base move (the pick backing away from a
table) send the same JSON as UDP datagrams to 127.0.0.1:RELAY_PORT instead. The
relay gets the same speed clamps and deadman, and the person at the keys always
wins: relay commands are ignored while a key is held.
"""

from __future__ import annotations

import argparse
import json
import math
import signal
import socket
import sys
import threading
import time

import numpy as np
from bbos import Config, Reader, Type, Writer

PERIOD = 0.05
DEADMAN_S = 0.4          # no command for this long -> zero twist
UPRIGHT_DEG = 25.0
RELAY_PORT = 8765        # robot-local only (bound to 127.0.0.1)
RELAY_V_MAX = 0.15       # a program gets a gentler limit than the person driving
STOP = threading.Event()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v-max", type=float, default=0.30)
    parser.add_argument("--w-max", type=float, default=1.0)
    args = parser.parse_args()
    signal.signal(signal.SIGINT, lambda *_: STOP.set())
    signal.signal(signal.SIGTERM, lambda *_: STOP.set())

    target = {"v": 0.0, "w": 0.0, "at": 0.0}
    relayed = {"v": 0.0, "w": 0.0, "at": 0.0}

    def read():
        for line in sys.stdin:
            try:
                message = json.loads(line)
                v, w = float(message["v"]), float(message["w"])
            except (ValueError, KeyError, TypeError):
                continue
            if math.isfinite(v) and math.isfinite(w):
                target.update(
                    v=min(max(v, -args.v_max), args.v_max),
                    w=min(max(w, -args.w_max), args.w_max),
                    at=time.monotonic(),
                )
        STOP.set()               # stdin closed: the dashboard or ssh went away

    threading.Thread(target=read, name="teleop-stdin", daemon=True).start()

    def relay():
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as listener:
            try:
                listener.bind(("127.0.0.1", RELAY_PORT))
            except OSError as exc:
                print(f"[teleop] relay unavailable: {exc}", flush=True)
                return
            print(f"[teleop] relay listening on 127.0.0.1:{RELAY_PORT}", flush=True)
            while not STOP.is_set():
                try:
                    message = json.loads(listener.recv(256))
                    v, w = float(message["v"]), float(message["w"])
                except (ValueError, KeyError, TypeError, OSError):
                    continue
                if math.isfinite(v) and math.isfinite(w):
                    limit = min(RELAY_V_MAX, args.v_max)
                    relayed.update(v=min(max(v, -limit), limit),
                                   w=min(max(w, -args.w_max), args.w_max), at=time.monotonic())

    threading.Thread(target=relay, name="teleop-relay", daemon=True).start()

    drive_cfg = Config("drive")
    circumference = math.pi * float(drive_cfg.wheel_diam)
    with Reader("imu.orientation", keeptime=False) as imu, \
            Reader("drive.state", keeptime=False) as state, \
            Writer("drive.ctrl", Type("drive_ctrl"), keeptime=False) as drive:

        def send(v, w):
            with drive.buf() as frame:
                frame["twist"] = np.array([v, w], dtype=np.float32)

        print("[teleop] ready", flush=True)
        rpy, vel, last_report = [0.0, 0.0, 0.0], [0.0, 0.0], 0.0
        try:
            while not STOP.is_set():
                if imu.ready():
                    rpy = np.asarray(imu.data["rpy"], dtype=float).tolist()
                if state.ready():
                    vel = np.asarray(state.data["vel"], dtype=float).tolist()
                now = time.monotonic()
                live = now - target["at"] <= DEADMAN_S
                helped = not live and now - relayed["at"] <= DEADMAN_S
                v, w = ((target["v"], target["w"]) if live
                        else (relayed["v"], relayed["w"]) if helped else (0.0, 0.0))
                if max(abs(rpy[0]), abs(rpy[1])) > UPRIGHT_DEG:
                    print("[teleop] not upright; stopping", flush=True)
                    break
                send(v, w)
                if now - last_report >= 0.2:
                    last_report = now
                    print("TELEOP " + json.dumps({
                        "v": round(v, 3), "w": round(w, 3),
                        "wheel_v": round((vel[0] + vel[1]) / 2 * circumference, 3),
                        "pitch": round(rpy[1], 1), "live": live, "relay": helped,
                    }), flush=True)
                time.sleep(PERIOD)
        finally:
            for _ in range(6):
                send(0.0, 0.0)
                time.sleep(PERIOD)
            print("[teleop] stopped; zero twist sent", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
