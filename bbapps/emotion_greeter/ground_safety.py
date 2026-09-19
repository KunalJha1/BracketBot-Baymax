"""Depth-grounded person safety observations for the robot vision app.

This module intentionally reports ``possible_person_on_ground`` rather than a
fall.  A single camera frame cannot establish that somebody fell, why they are
on the floor, or whether they need medical help.  The output is evidence for a
deterministic stop/notify policy, never a diagnosis.

The deployed BracketBot depth topics use these frames:

* ``camera.rect.left`` and ``camera.points.idx_2d`` share a 512x384 pixel grid.
* ``camera.points.points`` is ``[right, forward, up]`` in the robot base frame.
* the navigation map has yaw=0 facing map +Y, so base points transform using
  right ``[cos(yaw), sin(yaw)]`` and forward ``[-sin(yaw), cos(yaw)]``.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np


# COCO pose indices emitted by YOLO11-pose.
TORSO_KEYPOINTS = (5, 6, 11, 12)  # shoulders and hips
BODY_KEYPOINTS = tuple(range(5, 17))


@dataclass(frozen=True)
class Keypoint:
    index: int
    x: float
    y: float
    confidence: float


@dataclass(frozen=True)
class GroundAssessment:
    state: str
    confidence: float
    reason: str
    depth_keypoints: int
    torso_height_m: float | None = None
    body_extent_m: float | None = None
    base_position: tuple[float, float, float] | None = None
    map_position: tuple[float, float] | None = None

    @property
    def suspected(self) -> bool:
        return self.state == "possible_person_on_ground"


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _pixel_point_lookup(
    indices: np.ndarray,
    points: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:
    """Return a flat pixel -> point-row lookup, with -1 for missing depth."""

    lookup = np.full(width * height, -1, dtype=np.int32)
    indices = np.asarray(indices, dtype=np.int64).reshape(-1)
    points = np.asarray(points)
    count = min(len(indices), len(points))
    if count == 0:
        return lookup
    valid = (
        (indices[:count] >= 0)
        & (indices[:count] < width * height)
        & np.isfinite(points[:count]).all(axis=1)
    )
    rows = np.nonzero(valid)[0].astype(np.int32)
    lookup[indices[:count][valid]] = rows
    return lookup


def keypoints_in_base_frame(
    keypoints: Iterable[Keypoint],
    point_indices: np.ndarray,
    points: np.ndarray,
    image_width: int,
    image_height: int,
    *,
    keypoint_confidence: float = 0.35,
    search_radius_px: int = 5,
) -> dict[int, np.ndarray]:
    """Associate confident 2D pose joints with aligned base-frame depth.

    Depth is sparse because ``camera.points`` omits invalid pixels.  For each
    joint we take the median of a small local patch, which is more stable than
    selecting one edge pixel and limits background leakage around limbs.
    """

    point_indices = np.asarray(point_indices).reshape(-1)
    points = np.asarray(points, dtype=np.float32)
    lookup = _pixel_point_lookup(point_indices, points, image_width, image_height)
    associated: dict[int, np.ndarray] = {}
    radius = max(0, int(search_radius_px))
    radius_sq = radius * radius

    for keypoint in keypoints:
        if keypoint.confidence < keypoint_confidence:
            continue
        cx, cy = int(round(keypoint.x)), int(round(keypoint.y))
        if not (0 <= cx < image_width and 0 <= cy < image_height):
            continue
        x0, x1 = max(0, cx - radius), min(image_width - 1, cx + radius)
        y0, y1 = max(0, cy - radius), min(image_height - 1, cy + radius)
        rows = []
        for y in range(y0, y1 + 1):
            for x in range(x0, x1 + 1):
                if (x - cx) ** 2 + (y - cy) ** 2 > radius_sq:
                    continue
                row = int(lookup[y * image_width + x])
                if row >= 0:
                    rows.append(row)
        if not rows:
            continue
        patch = points[np.asarray(rows, dtype=np.int32)]
        # Points behind the camera or wildly outside the robot workspace are
        # malformed for this purpose, even if finite.
        patch = patch[
            (patch[:, 1] > 0.1)
            & (patch[:, 1] < 8.0)
            & (patch[:, 2] > -0.25)
            & (patch[:, 2] < 2.5)
        ]
        if len(patch):
            associated[keypoint.index] = np.median(patch, axis=0)
    return associated


def base_to_map(
    base_position: np.ndarray,
    robot_position: np.ndarray,
    robot_yaw: float,
) -> tuple[float, float]:
    """Transform ``[right, forward, up]`` into the navigation map frame."""

    right, forward = float(base_position[0]), float(base_position[1])
    cosine, sine = math.cos(robot_yaw), math.sin(robot_yaw)
    return (
        float(robot_position[0]) + cosine * right - sine * forward,
        float(robot_position[1]) + sine * right + cosine * forward,
    )


def assess_ground_pose(
    keypoints_3d: dict[int, np.ndarray],
    *,
    robot_position: np.ndarray | None = None,
    robot_yaw: float | None = None,
) -> GroundAssessment:
    """Assess whether depth-grounded pose evidence is consistent with floor level.

    Three torso joints are required.  The conservative decision requires a low
    torso, most visible body joints near the floor, and meaningful horizontal
    body extent.  Sitting, crouching, missing depth, and upper-body-only views
    therefore remain clear/unknown instead of becoming emergency alerts.
    """

    torso = [keypoints_3d[index] for index in TORSO_KEYPOINTS if index in keypoints_3d]
    body = [keypoints_3d[index] for index in BODY_KEYPOINTS if index in keypoints_3d]
    if len(torso) < 3 or len(body) < 4:
        return GroundAssessment(
            "unknown",
            0.0,
            "need at least three torso and four body keypoints with aligned depth",
            len(body),
        )

    torso_array = np.asarray(torso, dtype=np.float32)
    body_array = np.asarray(body, dtype=np.float32)
    torso_height = float(np.median(torso_array[:, 2]))
    low_fraction = float(np.mean(body_array[:, 2] <= 0.55))
    height_span = float(np.ptp(body_array[:, 2]))

    planar = body_array[:, :2]
    deltas = planar[:, None, :] - planar[None, :, :]
    body_extent = float(np.sqrt(np.sum(deltas * deltas, axis=2)).max())

    height_score = _clamp01((0.65 - torso_height) / 0.35)
    extent_score = _clamp01((body_extent - 0.35) / 0.55)
    flat_score = _clamp01((0.75 - height_span) / 0.45)
    score = (
        0.45 * height_score
        + 0.25 * low_fraction
        + 0.20 * extent_score
        + 0.10 * flat_score
    )
    suspected = (
        torso_height <= 0.55
        and low_fraction >= 0.60
        and body_extent >= 0.45
        and score >= 0.62
    )

    base_position_array = np.median(body_array, axis=0)
    base_position = tuple(float(value) for value in base_position_array)
    map_position = None
    if robot_position is not None and robot_yaw is not None:
        map_position = base_to_map(base_position_array, robot_position, robot_yaw)

    state = "possible_person_on_ground" if suspected else "clear"
    reason = (
        f"torso={torso_height:.2f}m low={low_fraction:.0%} "
        f"extent={body_extent:.2f}m height_span={height_span:.2f}m"
    )
    return GroundAssessment(
        state,
        round(score, 4),
        reason,
        len(body),
        round(torso_height, 3),
        round(body_extent, 3),
        base_position,
        map_position,
    )


@dataclass
class _TrackState:
    first_suspected_at: float | None = None
    first_clear_at: float | None = None
    confirmed: bool = False
    last_seen_at: float = 0.0


class GroundAlertTracker:
    """Require sustained evidence and sustained clearing for each visual track."""

    def __init__(self, hold_seconds: float = 2.0, clear_seconds: float = 2.0) -> None:
        self.hold_seconds = hold_seconds
        self.clear_seconds = clear_seconds
        self.tracks: dict[int, _TrackState] = {}

    def update(
        self,
        assessments: dict[int, GroundAssessment],
        now: float,
    ) -> dict[int, str]:
        statuses: dict[int, str] = {}
        for track_id, assessment in assessments.items():
            state = self.tracks.setdefault(track_id, _TrackState())
            state.last_seen_at = now
            if assessment.suspected:
                state.first_clear_at = None
                if state.first_suspected_at is None:
                    state.first_suspected_at = now
                if now - state.first_suspected_at >= self.hold_seconds:
                    state.confirmed = True
                statuses[track_id] = "alert" if state.confirmed else "checking"
            elif assessment.state == "clear":
                state.first_suspected_at = None
                if state.confirmed:
                    if state.first_clear_at is None:
                        state.first_clear_at = now
                    if now - state.first_clear_at >= self.clear_seconds:
                        state.confirmed = False
                        state.first_clear_at = None
                statuses[track_id] = "alert" if state.confirmed else "clear"
            else:
                # Missing depth is not evidence that a previously confirmed
                # person got up. Keep the alert latched until positive clearing.
                statuses[track_id] = "alert" if state.confirmed else "unknown"

        for track_id, state in list(self.tracks.items()):
            if track_id in assessments:
                continue
            if state.confirmed:
                statuses[track_id] = "alert"
            elif now - state.last_seen_at > max(2.0, self.clear_seconds):
                del self.tracks[track_id]
        return statuses
