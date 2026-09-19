"""Camera-target selection and geometry for the point-at-person gesture.

This module deliberately has no BBOS or OpenCV imports, so its target choice
and aiming math can be tested away from the robot.  Image ``x`` increases to
the viewer's right; robot ``y`` increases to the robot's left.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import threading
import time
from typing import Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class PersonTarget:
    """One detected person normalized into camera coordinates."""

    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    frame_width: int
    frame_height: int
    observed_at: float

    @property
    def area(self) -> int:
        return max(0, self.x2 - self.x1) * max(0, self.y2 - self.y1)

    @property
    def x_offset(self) -> float:
        """Horizontal center in ``[-1, 1]`` (left to right)."""

        center = 0.5 * (self.x1 + self.x2)
        return float(np.clip(2.0 * center / self.frame_width - 1.0, -1.0, 1.0))

    @property
    def y_offset(self) -> float:
        """Vertical center in ``[-1, 1]`` (top to bottom)."""

        center = 0.5 * (self.y1 + self.y2)
        return float(np.clip(2.0 * center / self.frame_height - 1.0, -1.0, 1.0))

    @property
    def arm(self) -> str:
        # Image-left is the robot's left when both face the same direction.
        return "left" if self.x_offset <= 0.0 else "right"


class PersonTargetTracker:
    """Thread-safe snapshot of people in the most recent detector frame."""

    def __init__(self, max_age: float = 1.0) -> None:
        self.max_age = max_age
        self._lock = threading.Lock()
        self._targets: tuple[PersonTarget, ...] = ()

    def update(
        self,
        detections: Iterable[Sequence[float]],
        frame_width: int,
        frame_height: int,
        *,
        observed_at: float | None = None,
    ) -> None:
        """Replace the snapshot from ``(x1, y1, x2, y2, confidence)`` rows."""

        if frame_width <= 0 or frame_height <= 0:
            raise ValueError("frame dimensions must be positive")
        stamp = time.monotonic() if observed_at is None else observed_at
        targets = []
        for row in detections:
            if len(row) != 5:
                raise ValueError("each detection must have five values")
            x1, y1, x2, y2, confidence = row
            target = PersonTarget(
                max(0, min(frame_width, round(float(x1)))),
                max(0, min(frame_height, round(float(y1)))),
                max(0, min(frame_width, round(float(x2)))),
                max(0, min(frame_height, round(float(y2)))),
                float(confidence),
                frame_width,
                frame_height,
                stamp,
            )
            if target.area > 0:
                targets.append(target)
        with self._lock:
            self._targets = tuple(targets)

    def select(
        self,
        preference: str = "primary",
        *,
        now: float | None = None,
    ) -> PersonTarget | None:
        """Choose a fresh target predictably.

        ``primary`` means the largest/closest-looking person, breaking ties by
        confidence and proximity to image center. ``left`` and ``right`` pick
        the leftmost or rightmost visible person.
        """

        if preference not in {"primary", "left", "right"}:
            raise ValueError(f"unknown target preference: {preference}")
        stamp = time.monotonic() if now is None else now
        with self._lock:
            targets = tuple(self._targets)
        fresh = [target for target in targets if stamp - target.observed_at <= self.max_age]
        if not fresh:
            return None
        if preference == "left":
            return min(fresh, key=lambda target: (target.x_offset, -target.area))
        if preference == "right":
            return max(fresh, key=lambda target: (target.x_offset, target.area))
        return max(
            fresh,
            key=lambda target: (target.area, target.confidence, -abs(target.x_offset)),
        )


def pointing_goal(
    target: PersonTarget,
    shoulder_height: float,
    *,
    reach: float = 0.34,
    horizontal_fov_degrees: float = 70.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a bounded IK position and pointing direction in robot base space.

    Depth is intentionally not inferred from a monocular box.  The hand stays
    inside a short, fixed-radius shell while its bearing follows the selected
    person's image location.  The vertical correction is small and clamped.
    """

    if not 0.20 <= reach <= 0.38:
        raise ValueError("pointing reach must stay between 0.20 and 0.38 m")
    half_fov = math.radians(horizontal_fov_degrees * 0.5)
    bearing = -target.x_offset * half_fov
    lateral = reach * math.sin(bearing)
    # Keep the chosen hand on its own side of the body even for a centered box.
    side_sign = 1.0 if target.arm == "left" else -1.0
    lateral = side_sign * max(0.06, abs(lateral))
    forward = math.sqrt(max(reach * reach - lateral * lateral, 0.0))
    height = float(np.clip(shoulder_height - 0.12 * target.y_offset, 0.55, 1.30))
    position = np.array([forward, lateral, height], dtype=np.float64)

    # The gripper's local Z axis points along this ray. A modest vertical term
    # mirrors the image bearing without ever aiming sharply up or down.
    direction = np.array(
        [math.cos(bearing), math.sin(bearing), -0.25 * target.y_offset],
        dtype=np.float64,
    )
    direction /= np.linalg.norm(direction)
    return position, direction


def quaternion_from_z(direction: Sequence[float]) -> np.ndarray:
    """Return an xyzw quaternion whose local Z axis follows ``direction``."""

    target = np.asarray(direction, dtype=np.float64)
    norm = float(np.linalg.norm(target))
    if not np.isfinite(target).all() or norm < 1e-9:
        raise ValueError("pointing direction must be finite and non-zero")
    target /= norm
    source = np.array([0.0, 0.0, 1.0])
    dot = float(np.clip(np.dot(source, target), -1.0, 1.0))
    if dot < -1.0 + 1e-9:
        return np.array([1.0, 0.0, 0.0, 0.0])
    cross = np.cross(source, target)
    quaternion = np.array([cross[0], cross[1], cross[2], 1.0 + dot])
    quaternion /= np.linalg.norm(quaternion)
    return quaternion


def quaternion_forward_z(quaternion: Sequence[float]) -> np.ndarray:
    """World direction of local Z for an xyzw quaternion (test/debug helper)."""

    x, y, z, w = np.asarray(quaternion, dtype=np.float64)
    return np.array(
        [2.0 * (x * z + y * w), 2.0 * (y * z - x * w), 1.0 - 2.0 * (x * x + y * y)]
    )


def quaternion_slerp(
    start: Sequence[float], end: Sequence[float], alpha: float
) -> np.ndarray:
    """Shortest-path spherical interpolation for xyzw quaternions."""

    q0 = np.asarray(start, dtype=np.float64)
    q1 = np.asarray(end, dtype=np.float64)
    q0 /= np.linalg.norm(q0)
    q1 /= np.linalg.norm(q1)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    dot = float(np.clip(dot, -1.0, 1.0))
    alpha = float(np.clip(alpha, 0.0, 1.0))
    if dot > 0.9995:
        result = q0 + alpha * (q1 - q0)
        return result / np.linalg.norm(result)
    angle = math.acos(dot)
    result = (
        math.sin((1.0 - alpha) * angle) * q0
        + math.sin(alpha * angle) * q1
    ) / math.sin(angle)
    return result / np.linalg.norm(result)
