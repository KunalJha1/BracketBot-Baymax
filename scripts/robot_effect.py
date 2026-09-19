"""Run one bounded sound or LED effect on BracketBot.

This script is copied to the robot by ``robot_dashboard.py``. It owns exactly
one BBOS writer, accepts only typed arguments, responds to SIGINT/SIGTERM, and
always releases the writer when the effect finishes.
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import signal
import time
import wave

import numpy as np
from bbos import Config, Type, Writer


STOP_REQUESTED = False


def request_stop(*_):
    global STOP_REQUESTED
    STOP_REQUESTED = True
    print("[effect] stop requested", flush=True)


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


def parse_rgb(value):
    try:
        rgb = tuple(int(part) for part in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("RGB must be three comma-separated integers") from exc
    if len(rgb) != 3 or any(component < 0 or component > 255 for component in rgb):
        raise argparse.ArgumentTypeError("RGB components must each be between 0 and 255")
    return rgb


def play_sound(path):
    cfg = Config("speaker")
    with wave.open(str(path), "rb") as source:
        if source.getsampwidth() != 2 or source.getcomptype() != "NONE":
            raise RuntimeError("sound must be uncompressed 16-bit PCM WAV")
        if source.getframerate() != cfg.sample_rate:
            raise RuntimeError(
                f"sound is {source.getframerate()} Hz; speaker requires {cfg.sample_rate} Hz"
            )

        source_channels = source.getnchannels()
        duration = source.getnframes() / source.getframerate()
        print(f"[effect] sound {path.name}: {duration:.2f}s", flush=True)
        # The schema's timing paces chunks to the speaker sample clock.
        with Writer("speaker.audio", Type("speaker_audio")) as speaker:
            time.sleep(0.25)
            while not STOP_REQUESTED:
                raw = source.readframes(cfg.chunk_size)
                if not raw:
                    break
                samples = np.frombuffer(raw, dtype="<i2").reshape(-1, source_channels)
                if source_channels == 1 and cfg.channels > 1:
                    samples = np.repeat(samples, cfg.channels, axis=1)
                elif source_channels > 1 and cfg.channels == 1:
                    samples = samples.mean(axis=1, dtype=np.float32).astype(np.int16)[:, None]
                elif source_channels != cfg.channels:
                    raise RuntimeError(
                        f"sound has {source_channels} channels; speaker requires {cfg.channels}"
                    )
                if len(samples) < cfg.chunk_size:
                    padding = np.zeros(
                        (cfg.chunk_size - len(samples), cfg.channels), dtype=np.int16
                    )
                    samples = np.concatenate((samples, padding))
                with speaker.buf() as frame:
                    frame["audio"] = samples
    print("[effect] sound complete", flush=True)


def led_scale(pattern, elapsed):
    if pattern == "solid":
        return 1.0
    if pattern == "blink":
        return 1.0 if int(elapsed * 3) % 2 == 0 else 0.08
    # A 1.6-second cosine pulse that never completely disappears.
    return 0.2 + 0.8 * (0.5 - 0.5 * math.cos(2 * math.pi * elapsed / 1.6))


def play_led(rgb, pattern, duration):
    print(f"[effect] LED {rgb} {pattern} for {duration:.2f}s", flush=True)
    started = time.monotonic()
    with Writer("led.ctrl", Type("led_ctrl"), keeptime=False) as led:
        while not STOP_REQUESTED and time.monotonic() - started < duration:
            scale = led_scale(pattern, time.monotonic() - started)
            color = np.asarray([round(value * scale) for value in rgb], dtype=np.uint8)
            with led.buf() as frame:
                frame["rgb"] = color
                frame["brightness"] = np.int16(-1)
                frame["period_ms"] = np.uint16(0)
            time.sleep(0.05)
        # Clear the temporary expression rather than leaving a stale color.
        with led.buf() as frame:
            frame["rgb"] = np.zeros(3, dtype=np.uint8)
            frame["brightness"] = np.int16(-1)
            frame["period_ms"] = np.uint16(0)
    print("[effect] LED complete", flush=True)


def build_parser():
    parser = argparse.ArgumentParser(description="Run one bounded BracketBot effect")
    parser.add_argument("--pid-file", type=Path)
    commands = parser.add_subparsers(dest="command", required=True)

    sound = commands.add_parser("sound")
    sound.add_argument("path", type=Path)

    led = commands.add_parser("led")
    led.add_argument("--rgb", type=parse_rgb, required=True)
    led.add_argument("--pattern", choices=("solid", "pulse", "blink"), default="solid")
    led.add_argument("--duration", type=float, default=3.0)
    return parser


def main():
    args = build_parser().parse_args()
    if args.command == "sound" and not args.path.is_file():
        raise SystemExit(f"sound not found: {args.path}")
    if args.command == "led" and not 0.1 <= args.duration <= 30:
        raise SystemExit("--duration must be between 0.1 and 30 seconds")

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    if hasattr(signal, "SIGHUP"):
        signal.signal(signal.SIGHUP, request_stop)
    write_pid_file(args.pid_file)
    try:
        if args.command == "sound":
            play_sound(args.path)
        else:
            play_led(args.rgb, args.pattern, args.duration)
    finally:
        remove_pid_file(args.pid_file)


if __name__ == "__main__":
    main()
