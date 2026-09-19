"""Build an extend-salute-wave gesture from proven arm recordings.

The hand extension comes from ``hug.json`` and the raised salute/wave comes from
``wave.json``. A smooth bounded blend joins those two recorded poses. Keeping
this generator beside the asset makes that provenance reviewable and the JSON
reproducible.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "bbapps" / "greeter" / "movements" / "wave.json"
DEFAULT_EXTENSION_SOURCE = ROOT / "bbapps" / "greeter" / "movements" / "hug.json"
DEFAULT_OUTPUT = ROOT / "bbapps" / "greeter" / "movements" / "salute.json"
LIFT_SECONDS = 1.25
EXTEND_SOURCE_SECONDS = 1.51
EXTEND_DURATION = 0.9
TRANSITION_SECONDS = 0.6
TRANSITION_TICK = 0.015
HOLD_SECONDS = 0.45


def smooth(value):
    return value * value * (3.0 - 2.0 * value)


def build_salute(frames, extension_frames):
    """Return a straight extension, salute transition, and recorded wave."""
    if not isinstance(frames, list) or len(frames) < 2:
        raise ValueError("wave recording must contain at least two frames")
    if not isinstance(extension_frames, list) or len(extension_frames) < 2:
        raise ValueError("extension recording must contain at least two frames")

    start_time = float(frames[0]["t"])
    normalized_times = [float(frame["t"]) - start_time for frame in frames]
    salute_index = max(
        (index for index, stamp in enumerate(normalized_times) if stamp <= LIFT_SECONDS),
        default=-1,
    )
    if salute_index < 1 or salute_index >= len(frames) - 1:
        raise ValueError("wave recording does not contain a distinct lift and wave")

    neutral_right = list(frames[0]["right"])
    extension_start = float(extension_frames[0]["t"])
    extension = [
        frame
        for frame in extension_frames
        if float(frame["t"]) - extension_start <= EXTEND_SOURCE_SECONDS
    ]
    if len(extension) < 2:
        raise ValueError("extension recording does not contain the straight-arm pose")
    extension_span = float(extension[-1]["t"]) - extension_start
    result = [
        {
            "t": round(
                (float(frame["t"]) - extension_start)
                / extension_span
                * EXTEND_DURATION,
                4,
            ),
            "left": list(frame["left"]),
            "right": neutral_right,
        }
        for frame in extension
    ]

    extended = np.asarray(result[-1]["left"], dtype=np.float64)
    salute = np.asarray(frames[salute_index]["left"], dtype=np.float64)
    transition_steps = round(TRANSITION_SECONDS / TRANSITION_TICK)
    for step in range(1, transition_steps + 1):
        fraction = smooth(step / transition_steps)
        pose = extended + fraction * (salute - extended)
        result.append(
            {
                "t": round(EXTEND_DURATION + step * TRANSITION_TICK, 4),
                "left": pose.tolist(),
                "right": neutral_right,
            }
        )

    hold_end = round(EXTEND_DURATION + TRANSITION_SECONDS + HOLD_SECONDS, 4)
    result.append({"t": hold_end, "left": salute.tolist(), "right": neutral_right})

    salute_time = normalized_times[salute_index]
    for frame, stamp in zip(frames[salute_index + 1 :], normalized_times[salute_index + 1 :]):
        result.append(
            {
                "t": round(hold_end + stamp - salute_time, 4),
                "left": list(frame["left"]),
                "right": neutral_right,
            }
        )
    return result


def main():
    parser = argparse.ArgumentParser(description="Regenerate the salute trajectory")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument(
        "--extension-source", type=Path, default=DEFAULT_EXTENSION_SOURCE
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    frames = json.loads(args.source.read_text())
    extension_frames = json.loads(args.extension_source.read_text())
    salute = build_salute(frames, extension_frames)
    args.output.write_text(json.dumps(salute, separators=(",", ":")) + "\n")
    print(f"wrote {len(salute)} frames to {args.output}")


if __name__ == "__main__":
    main()
