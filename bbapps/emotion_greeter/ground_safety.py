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

from dataclasses import dataclass, replace
import math
from typing import Iterable

import numpy as np


# COCO pose indices emitted by YOLO11-pose.
TORSO_KEYPOINTS = (5, 6, 11, 12)  # shoulders and hips
SHOULDER_KEYPOINTS = (5, 6)
HIP_KEYPOINTS = (11, 12)
HEAD_KEYPOINTS = (0, 1, 2, 3, 4)  # nose, eyes, ears
BODY_KEYPOINTS = tuple(range(5, 17))

# Lying down means the trunk itself is horizontal and at floor level. Somebody
# sitting on the floor with their legs out also has low hips, low legs and a
# long footprint, so those cues alone cannot tell the two apart; the shoulders
# and head can. A seated adult's shoulders are ~0.55-0.65 m up and ~0.45 m
# above their hips, and their head is ~0.8 m up.
LYING_SHOULDER_MAX_M = 0.45
LYING_TORSO_RISE_MAX_M = 0.30
LYING_HEAD_MAX_M = 0.55


# Base-frame [right, forward, up, 1] -> camera.rect pixel, fitted on the robot
# from 128k camera.points samples (0.02 px median reprojection error). It puts
# the camera 1.55 m above the wheels, pitched 57 degrees below the horizon.
RECT_SIZE = (512, 384)
RECT_PROJECTION = np.array(
    [
        [132.277289, 199.880301, -127.052585, 196.931253],
        [2.308924, 92.878442, -218.04224, 337.96241],
        [0.0, 0.838664, -0.544649, 0.844193],
    ],
    dtype=np.float64,
)
_RECT_RAY = np.linalg.inv(RECT_PROJECTION[:, :3])
CAMERA_CENTRE = -_RECT_RAY @ RECT_PROJECTION[:, 3]

# The depth cloud ends ~1.7 m out, so most people on the floor have no depth on
# their joints. Bone lengths give a depth-free test instead: slide every joint
# down its camera ray onto the floor. A body that really is lying there keeps
# human proportions; any raised joint lands far behind where it is, so a
# standing, sitting or crouching body comes out metres long.
MONOCULAR_FLOOR_HEIGHT_M = 0.10
MONOCULAR_MAX_FORWARD_M = 6.0
MONOCULAR_SEGMENT_MAX_M = {
    (5, 11): 0.72, (6, 12): 0.72,    # shoulder-hip
    (11, 13): 0.63, (12, 14): 0.63,  # hip-knee
    (13, 15): 0.60, (14, 16): 0.60,  # knee-ankle
    (5, 7): 0.52, (6, 8): 0.52,      # shoulder-elbow
    (5, 6): 0.60, (11, 12): 0.50,    # across shoulders, across hips
}
MONOCULAR_TORSO_MIN_M = 0.22


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
    body_radius_m: float | None = None

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


def box_position_in_base_frame(
    box: tuple[int, int, int, int],
    point_indices: np.ndarray,
    points: np.ndarray,
    image_width: int,
    *,
    min_points: int = 25,
    body_depth_m: float = 0.4,
) -> tuple[float, float, float] | None:
    """Where a detected person stands (right, forward, up), from the depth inside their box.

    Pose joints need a visible torso, so a person who turns their back or is cut
    off by the frame gets no ``base_position``. The box is there from any side.
    Only its central part is used, and only the nearest surface in it: the
    corners and the far side of the box are floor and wall behind the person.
    """

    x1, y1, x2, y2 = box
    indices = np.asarray(point_indices, dtype=np.int64).reshape(-1)
    points = np.asarray(points, dtype=np.float32)
    count = min(len(indices), len(points))
    if count == 0 or x2 <= x1 or y2 <= y1:
        return None
    indices, points = indices[:count], points[:count]
    xs, ys = indices % image_width, indices // image_width
    width, height = x2 - x1, y2 - y1
    inside = (
        (indices >= 0)
        & (xs >= x1 + 0.25 * width) & (xs <= x2 - 0.25 * width)
        & (ys >= y1 + 0.15 * height) & (ys <= y2 - 0.30 * height)
        & np.isfinite(points).all(axis=1)
        & (points[:, 1] > 0.1) & (points[:, 1] < 8.0)
        & (points[:, 2] > -0.25) & (points[:, 2] < 2.5)
    )
    body = points[inside]
    if len(body) < min_points:
        return None
    nearest = np.percentile(body[:, 1], 20)
    body = body[body[:, 1] <= nearest + body_depth_m]
    if len(body) < min_points:
        return None
    right, forward, up = np.median(body, axis=0)
    return float(right), float(forward), float(up)


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


def floor_points_from_pixels(
    pixels: np.ndarray, floor_height_m: float = MONOCULAR_FLOOR_HEIGHT_M
) -> np.ndarray:
    """Where each camera.rect pixel's ray meets the floor; NaN above the horizon."""

    pixels = np.asarray(pixels, dtype=np.float64).reshape(-1, 2)
    rays = np.column_stack([pixels, np.ones(len(pixels))]) @ _RECT_RAY.T
    points = np.full((len(pixels), 3), np.nan)
    down = rays[:, 2] < -1e-6
    travel = (floor_height_m - CAMERA_CENTRE[2]) / rays[down, 2]
    points[down] = CAMERA_CENTRE + travel[:, None] * rays[down]
    return points


def monocular_floor_pose(
    keypoints: Iterable[Keypoint],
    image_width: int,
    image_height: int,
    *,
    keypoint_confidence: float = 0.35,
) -> tuple[dict[int, np.ndarray] | None, str]:
    """Return the pose laid on the floor if it has human proportions there.

    ``(pose, reason)``: a pose means the skeleton is consistent with a body
    lying on the floor. ``(None, reason)`` means it is not, or cannot be told.
    """

    if (image_width, image_height) != RECT_SIZE:
        return None, f"no camera model for {image_width}x{image_height}"
    usable = [
        kp for kp in keypoints
        if kp.confidence >= keypoint_confidence
        # Joints mapped in from the raw-resolution floor crop can fall just past
        # the rect border; the pinhole model still holds there.
        and -0.5 * image_width <= kp.x < 1.5 * image_width
        and -0.5 * image_height <= kp.y < 1.5 * image_height
    ]
    body = [kp for kp in usable if kp.index in BODY_KEYPOINTS]
    torso = [kp for kp in usable if kp.index in TORSO_KEYPOINTS]
    if len(torso) < 3 or len(body) < 6:
        return None, "need three torso and six body keypoints"
    floor = floor_points_from_pixels([(kp.x, kp.y) for kp in usable])
    return floor_pose_if_human({kp.index: point for kp, point in zip(usable, floor)})


def floor_pose_if_human(
    pose: dict[int, np.ndarray],
) -> tuple[dict[int, np.ndarray] | None, str]:
    """Keep a pose already laid on the floor plane only if its bones are human-sized.

    Shared by every camera model: the caller intersects its own pixel rays with
    the floor (NaN for rays above the horizon) and this judges the result.
    """

    body = [i for i in pose if i in BODY_KEYPOINTS]
    torso = [i for i in pose if i in TORSO_KEYPOINTS]
    if len(torso) < 3 or len(body) < 6:
        return None, "need three torso and six body keypoints"
    for index in body:
        point = pose[index]
        if not np.isfinite(point).all() or not 0.2 < point[1] <= MONOCULAR_MAX_FORWARD_M:
            return None, f"joint {index} is not on floor within {MONOCULAR_MAX_FORWARD_M:.0f} m"
    torso_lengths = []
    for (first, second), limit in MONOCULAR_SEGMENT_MAX_M.items():
        if first in pose and second in pose:
            length = float(np.linalg.norm(pose[first] - pose[second]))
            if length > limit:
                return None, (
                    f"joints {first}-{second} would span {length:.2f}m on the floor "
                    f"(max {limit:.2f}m): body is raised, not lying"
                )
            if (first, second) in ((5, 11), (6, 12)):
                torso_lengths.append(length)
    if not torso_lengths or max(torso_lengths) < MONOCULAR_TORSO_MIN_M:
        return None, "torso too short to be a person on the floor"
    # Head joints only matter when they are finite; an unseen head is not evidence.
    return {i: pt for i, pt in pose.items() if np.isfinite(pt).all()}, "floor-consistent"


def assess_ground_pose_monocular(
    keypoints: Iterable[Keypoint],
    image_width: int,
    image_height: int,
    *,
    keypoint_confidence: float = 0.35,
    robot_position: np.ndarray | None = None,
    robot_yaw: float | None = None,
) -> GroundAssessment:
    """Depth-free fallback for people beyond the depth cloud."""

    keypoints = list(keypoints)
    pose, reason = monocular_floor_pose(
        keypoints, image_width, image_height, keypoint_confidence=keypoint_confidence
    )
    if pose is None:
        raised = "not lying" in reason
        return GroundAssessment("clear" if raised else "unknown", 0.0, f"mono: {reason}", 0)
    assessment = assess_ground_pose(pose, robot_position=robot_position, robot_yaw=robot_yaw)
    return replace(assessment, reason=f"mono: {assessment.reason}")


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
    torso, most visible body joints near the floor, meaningful horizontal body
    extent, and a lying posture: shoulders at floor level, a near-horizontal
    trunk, and a low head when one is visible.  Sitting (on a chair or on the
    floor), crouching, kneeling, missing depth, and upper-body-only views
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
    # Three torso joints guarantee at least one shoulder and one hip.
    shoulders = [keypoints_3d[i][2] for i in SHOULDER_KEYPOINTS if i in keypoints_3d]
    hips = [keypoints_3d[i][2] for i in HIP_KEYPOINTS if i in keypoints_3d]
    heads = [keypoints_3d[i][2] for i in HEAD_KEYPOINTS if i in keypoints_3d]
    shoulder_height = float(max(shoulders))
    torso_rise = abs(float(np.mean(shoulders)) - float(np.mean(hips)))
    head_height = float(np.median(heads)) if heads else None
    lying = (
        shoulder_height <= LYING_SHOULDER_MAX_M
        and torso_rise <= LYING_TORSO_RISE_MAX_M
        and (head_height is None or head_height <= LYING_HEAD_MAX_M)
    )
    low = (
        torso_height <= 0.55
        and low_fraction >= 0.60
        and body_extent >= 0.45
        and score >= 0.62
    )
    suspected = low and lying

    base_position_array = np.median(body_array, axis=0)
    base_position = tuple(float(value) for value in base_position_array)
    # Include every visible joint and a margin for unobserved hands/head/clothing.
    # Approaching a torso centre alone could put the wheels over outstretched legs.
    all_joints = np.asarray(list(keypoints_3d.values()), dtype=np.float32)
    body_radius = float(np.linalg.norm(all_joints[:, :2] - base_position_array[:2], axis=1).max()) + 0.25
    map_position = None
    if robot_position is not None and robot_yaw is not None:
        map_position = base_to_map(base_position_array, robot_position, robot_yaw)

    state = "possible_person_on_ground" if suspected else "clear"
    reason = (
        f"torso={torso_height:.2f}m low={low_fraction:.0%} "
        f"extent={body_extent:.2f}m height_span={height_span:.2f}m "
        f"shoulders={shoulder_height:.2f}m rise={torso_rise:.2f}m"
    )
    if head_height is not None:
        reason += f" head={head_height:.2f}m"
    if low and not lying:
        reason += " | low but trunk upright: sitting or crouching, not lying"
    return GroundAssessment(
        state,
        round(score, 4),
        reason,
        len(body),
        round(torso_height, 3),
        round(body_extent, 3),
        base_position,
        map_position,
        body_radius,
    )


@dataclass
class _TrackState:
    first_suspected_at: float | None = None
    last_suspected_at: float = 0.0
    first_clear_at: float | None = None
    confirmed: bool = False
    last_seen_at: float = 0.0


class GroundAlertTracker:
    """Require sustained evidence and sustained clearing for each visual track."""

    def __init__(
        self,
        hold_seconds: float = 2.0,
        clear_seconds: float = 2.0,
        gap_seconds: float = 0.7,
        lost_seconds: float = 15.0,
    ) -> None:
        self.hold_seconds = hold_seconds
        self.clear_seconds = clear_seconds
        # A body on the floor is a hard pose: the detector drops it for a frame
        # or two. Gaps this short do not restart the confirmation hold.
        self.gap_seconds = gap_seconds
        # A confirmed person nobody has seen for this long is gone (or the
        # tracker renamed them); holding the alert forever would flash the
        # emergency lights and block every later check-in.
        self.lost_seconds = lost_seconds
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
                state.last_suspected_at = now
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
                if now - state.last_suspected_at > self.gap_seconds:
                    state.first_suspected_at = None
                state.first_clear_at = None

        for track_id, state in list(self.tracks.items()):
            if track_id in assessments:
                continue
            if state.confirmed:
                if now - state.last_seen_at > self.lost_seconds:
                    del self.tracks[track_id]
                    continue
                state.first_clear_at = None
                statuses[track_id] = "alert"
            else:
                if now - state.last_suspected_at > self.gap_seconds:
                    state.first_suspected_at = None
                if now - state.last_seen_at > max(2.0, self.clear_seconds):
                    del self.tracks[track_id]
        return statuses
