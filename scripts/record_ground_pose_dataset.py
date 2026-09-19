#!/usr/bin/env python3
"""Record labelled ground-pose telemetry from the robot vision status API.

Only numeric/status evidence is retained; this tool does not download or store
camera images. Run one short capture per staged pose while the robot is
stationary and autonomous navigation is off.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen


LABELS = ("empty", "standing", "sitting", "crouching", "kneeling", "lying")


def fetch_status(url: str, timeout: float = 2.0) -> dict[str, Any]:
    request = Request(url, headers={"Accept": "application/json"})
    with urlopen(request, timeout=timeout) as response:
        return json.load(response)


def make_record(status: dict[str, Any], label: str) -> dict[str, Any]:
    """Keep the calibration evidence and omit unrelated dashboard state."""

    mapping = status.get("map") or {}
    return {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "label": label,
        "frame": status.get("frame"),
        "people": status.get("people"),
        "track_ids": status.get("track_ids", []),
        "ground_status": status.get("ground_status"),
        "depth_aligned": status.get("depth_aligned"),
        "camera_age_ms": status.get("camera_age_ms"),
        "scan_fps": status.get("scan_fps"),
        "map_epoch": mapping.get("map_epoch"),
        "robot_position": mapping.get("robot_position"),
        "robot_heading": mapping.get("robot_heading"),
        "ground_observations": status.get("ground_observations", []),
    }


def run(args: argparse.Namespace) -> int:
    deadline = time.monotonic() + args.seconds
    next_sample = time.monotonic()
    records = 0
    errors = 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("a", encoding="utf-8") as output:
        while time.monotonic() < deadline:
            wait = next_sample - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            next_sample += args.interval
            try:
                status = fetch_status(args.url, args.timeout)
            except (OSError, URLError, TimeoutError, ValueError) as exc:
                errors += 1
                print(f"[capture] request failed: {exc}", flush=True)
                continue
            record = make_record(status, args.label)
            output.write(json.dumps(record, separators=(",", ":")) + "\n")
            output.flush()
            records += 1
            print(
                f"[capture] label={args.label} frame={record['frame']} "
                f"people={record['people']} ground={record['ground_status']} "
                f"observations={len(record['ground_observations'])}",
                flush=True,
            )
    print(
        f"[capture] wrote {records} records to {args.output} ({errors} errors)",
        flush=True,
    )
    return 0 if records else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Record labelled, image-free ground-pose calibration telemetry"
    )
    parser.add_argument("--label", required=True, choices=LABELS)
    parser.add_argument("--seconds", type=float, default=15.0)
    parser.add_argument("--interval", type=float, default=0.5)
    parser.add_argument("--timeout", type=float, default=2.0)
    parser.add_argument(
        "--url", default="http://127.0.0.1:8018/api/status"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/ground-pose-calibration.jsonl"),
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.seconds <= 0:
        raise SystemExit("--seconds must be greater than zero")
    if args.interval <= 0:
        raise SystemExit("--interval must be greater than zero")
    if args.timeout <= 0:
        raise SystemExit("--timeout must be greater than zero")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
