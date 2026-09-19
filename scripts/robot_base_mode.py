"""Hold BracketBot lean mode until interrupted, then restore balance mode.

The BBOS base-mode request expires after roughly 0.25 seconds, so lean must be
republished continuously. If this process or SSH dies, expiration is the final
fallback; normal signals explicitly publish BALANCE before exiting.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import signal
import time

import numpy as np
from bbos import Reader, Type, Writer


BALANCE = 0
LEAN = 1
PERIOD = 0.05
UPRIGHT_DEG = 25.0
STOP_REQUESTED = False


def request_stop(*_):
    global STOP_REQUESTED
    STOP_REQUESTED = True
    print("[base] balance requested", flush=True)


def write_pid_file(path):
    if path is not None:
        path.write_text(f"{os.getpid()}\n")


def remove_pid_file(path):
    if path is None:
        return
    try:
        if path.read_text().strip() == str(os.getpid()):
            path.unlink()
    except OSError:
        pass


def fresh(reader, timeout=2.0):
    started = time.monotonic()
    while not reader.ready():
        if time.monotonic() - started > timeout:
            raise RuntimeError("no fresh IMU sample; is the daemon running?")
        time.sleep(0.002)
    return reader.data


def write_mode(writer, mode, angle):
    with writer.buf() as frame:
        frame["mode"] = np.uint8(mode)
        frame["lean_angle_deg"] = np.float32(angle)


def main():
    parser = argparse.ArgumentParser(description="Hold a bounded BracketBot lean")
    parser.add_argument("--angle", type=float, default=4.0)
    parser.add_argument("--pid-file", type=Path)
    args = parser.parse_args()
    if not 1.0 <= args.angle <= 15.0:
        parser.error("--angle must be between 1 and 15 degrees")

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    if hasattr(signal, "SIGHUP"):
        signal.signal(signal.SIGHUP, request_stop)

    write_pid_file(args.pid_file)
    try:
        with Reader("imu.orientation", keeptime=False) as imu:
            rpy = np.asarray(fresh(imu)["rpy"], dtype=float)
        print(f"[base] IMU rpy (deg): {rpy}", flush=True)
        if abs(rpy[0]) >= UPRIGHT_DEG or abs(rpy[1]) >= UPRIGHT_DEG:
            raise RuntimeError(
                f"robot is not upright; |roll| and |pitch| must be < {UPRIGHT_DEG:.0f} deg"
            )

        with Writer("base.mode", Type("base_mode"), keeptime=False) as base:
            write_mode(base, LEAN, args.angle)
            print(f"[base] lean active at {args.angle:.1f} degrees", flush=True)
            try:
                while not STOP_REQUESTED:
                    write_mode(base, LEAN, args.angle)
                    time.sleep(PERIOD)
            finally:
                print("[base] restoring balance mode", flush=True)
                for _ in range(6):
                    write_mode(base, BALANCE, 0.0)
                    time.sleep(PERIOD)
                print("[base] balance mode active", flush=True)
    finally:
        remove_pid_file(args.pid_file)


if __name__ == "__main__":
    main()
