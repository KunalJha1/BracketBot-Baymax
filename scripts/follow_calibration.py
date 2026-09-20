"""Robot-specific follow calibration. Kept on the robot, never overwritten by deployment.

These values are measured with a person at the robot; a passing laptop test is
not calibration. See docs/follow-readiness.md before creating the JSON file.
"""

from dataclasses import dataclass
import json
import math
from pathlib import Path
import socket


DEFAULT_PATH = Path.home() / ".config" / "baymax" / "follow.json"


@dataclass(frozen=True)
class Calibration:
    left_sign: float = -1.0
    self_mask: tuple = ()
    wheel_order: tuple = (0, 1)
    wheel_signs: tuple = (1.0, 1.0)
    motion_speed_limit: float = 0.15


def number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("expected a finite number")
    return float(value)


def load_calibration(path, *, required=True, hostname=None):
    path = Path(path).expanduser()
    if not path.exists() and not required:
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if type(data["schema_version"]) is not int or data["schema_version"] != 1:
            raise ValueError("unsupported schema_version")
        if data["robot_id"] != (hostname or socket.gethostname()):
            raise ValueError("robot_id does not match this robot's hostname")
        if not isinstance(data["evidence"], str) or not data["evidence"].strip():
            raise ValueError("evidence must describe the physical calibration checks")
        sign = number(data["left_sign"])
        if sign not in (-1.0, 1.0):
            raise ValueError("left_sign must be -1 or 1")
        if not isinstance(data["self_mask"], list):
            raise ValueError("self_mask must be a list of reviewed boxes (or [] after inspection)")
        boxes = []
        for box in data["self_mask"]:
            if len(box) != 6:
                raise ValueError("each self_mask box needs six forward/left/up bounds")
            bounds = tuple(number(v) for v in box)
            if any(bounds[i] >= bounds[i + 1] for i in (0, 2, 4)):
                raise ValueError("self_mask minima must be below maxima")
            boxes.append(bounds)
        order = tuple(data["wheel_order"])
        if any(type(v) is not int for v in order) or sorted(order) != [0, 1]:
            raise ValueError("wheel_order must be [0, 1] or [1, 0]")
        signs = tuple(number(v) for v in data["wheel_signs"])
        if len(signs) != 2 or any(v not in (-1.0, 1.0) for v in signs):
            raise ValueError("wheel_signs must contain two signs, each -1 or 1")
        limit = number(data["motion_speed_limit"])
        if not 0 < limit <= 0.30:
            raise ValueError("motion_speed_limit must be above 0 and at most 0.30 m/s")
        return Calibration(sign, tuple(boxes), order, signs, limit)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise RuntimeError(
            f"follow calibration unavailable or invalid: {path}: {exc}. "
            "Run the read-only probe and follow docs/follow-readiness.md; motion is disabled."
        ) from exc
