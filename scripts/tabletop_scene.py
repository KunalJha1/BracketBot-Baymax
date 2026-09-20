"""Depth-only tabletop scene: tilted table plane plus the objects standing on it.

Pure numpy, so it is unit-testable and can be uploaded to ``/tmp`` beside the
robot runners. Input clouds use the depth daemon's ``camera.points`` frame
(lateral right, forward, up); every output is in the arm IK frame
(forward, left, up), matching ``scripts/table_rest.py``.

The table is fitted as a *tilted* plane: the robot balances, so its body (and
therefore the base frame) pitches several degrees relative to a level table.
A live capture on bracketbot-184 measured a 9-11 degree apparent tilt.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, asdict
import math

import numpy as np


PLANE_INLIER_METRES = 0.012
MAX_TABLE_TILT_DEGREES = 20.0
RANSAC_SCORE_POINTS = 15000
REGION_X = (0.05, 1.20)
REGION_Y = 0.80
REGION_Z = (0.45, 1.15)
OBJECT_MIN_HEIGHT = 0.02
OBJECT_MAX_HEIGHT = 0.45
CLUSTER_CELL_METRES = 0.02
MIN_CLUSTER_POINTS = 25
TABLE_NEIGHBOURHOOD_CELLS = 3


@dataclass(frozen=True)
class TablePlane:
    centroid: tuple[float, float, float]
    normal: tuple[float, float, float]
    inliers: int
    tilt_degrees: float
    near_edge: float
    edge_samples: np.ndarray = None  # (N, 2) subsampled inlier x, y

    def near_edge_at(self, y: float, half_width: float = 0.12) -> float:
        """Nearest table edge in the strip around lateral ``y``.

        A round table's edge is much closer in front of the robot's centre
        than out where an arm hangs, so a single global edge is wrong for
        clearance checks at a particular lateral offset.
        """

        if self.edge_samples is None or not len(self.edge_samples):
            return self.near_edge
        strip = self.edge_samples[np.abs(self.edge_samples[:, 1] - y) <= half_width]
        if len(strip) < 30:
            return self.near_edge
        return float(np.quantile(strip[:, 0], 0.03))

    def height_at(self, x: float, y: float) -> float:
        """Table surface height (arm z) below arm-frame ``(x, y)``."""

        cx, cy, cz = self.centroid
        nx, ny, nz = self.normal
        return cz - (nx * (x - cx) + ny * (y - cy)) / nz

    def height_above(self, points) -> np.ndarray:
        """Signed distance of arm-frame points above the plane."""

        return (np.asarray(points, dtype=np.float64) - self.centroid) @ np.asarray(self.normal)


@dataclass(frozen=True)
class TableObject:
    center: tuple[float, float, float]  # footprint centre on the table surface
    top: float                          # height of the highest point above the table
    length: float                       # footprint major axis, metres
    width: float                        # footprint minor axis, metres
    yaw: float                          # major-axis heading in the arm frame, radians
    points: int

    def as_dict(self) -> dict:
        return asdict(self)


def points_to_arm(points) -> np.ndarray:
    values = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    return np.column_stack((values[:, 1], -values[:, 0], values[:, 2]))


def _region(arm: np.ndarray) -> np.ndarray:
    return (
        np.isfinite(arm).all(axis=1)
        & (arm[:, 0] >= REGION_X[0]) & (arm[:, 0] <= REGION_X[1])
        & (np.abs(arm[:, 1]) <= REGION_Y)
        & (arm[:, 2] >= REGION_Z[0]) & (arm[:, 2] <= REGION_Z[1])
    )


def fit_table_plane(arm_points, iterations=300, seed=0) -> TablePlane:
    """RANSAC the dominant near-horizontal plane, then refine it by SVD."""

    arm = np.asarray(arm_points, dtype=np.float64).reshape(-1, 3)
    candidates = arm[_region(arm)]
    if len(candidates) < 200:
        raise RuntimeError(f"too few depth points near table height ({len(candidates)})")
    rng = np.random.default_rng(seed)
    # Score hypotheses on a subsample: counting inliers over every depth point
    # 300 times was most of a scan. The refit below still uses all points.
    scored = candidates
    if len(candidates) > RANSAC_SCORE_POINTS:
        scored = candidates[rng.choice(len(candidates), RANSAC_SCORE_POINTS, replace=False)]
    min_nz = math.cos(math.radians(MAX_TABLE_TILT_DEGREES))
    best_count, best = 0, None
    for _ in range(iterations):
        sample = candidates[rng.choice(len(candidates), 3, replace=False)]
        normal = np.cross(sample[1] - sample[0], sample[2] - sample[0])
        length = float(np.linalg.norm(normal))
        if length < 1e-9:
            continue
        normal /= length
        if normal[2] < 0:
            normal = -normal
        if normal[2] < min_nz:
            continue
        count = int(np.count_nonzero(
            np.abs(scored @ normal - sample[0] @ normal) < PLANE_INLIER_METRES))
        if count > best_count:
            best_count, best = count, (sample[0], normal)
    if best is None:
        raise RuntimeError("no near-horizontal plane found")
    origin, normal = best
    inliers = candidates[np.abs((candidates - origin) @ normal) < PLANE_INLIER_METRES]
    centroid = inliers.mean(axis=0)
    normal = np.linalg.svd(inliers - centroid, full_matrices=False)[2][2]
    if normal[2] < 0:
        normal = -normal
    inliers = candidates[np.abs((candidates - centroid) @ normal) < PLANE_INLIER_METRES]
    if len(inliers) < 500:
        raise RuntimeError(f"table plane is too small ({len(inliers)} inliers)")
    return TablePlane(
        tuple(float(v) for v in centroid),
        tuple(float(v) for v in normal),
        int(len(inliers)),
        float(math.degrees(math.acos(float(np.clip(normal[2], -1.0, 1.0))))),
        float(np.quantile(inliers[:, 0], 0.03)),
        inliers[:: max(1, len(inliers) // 4000), :2].astype(np.float32).copy(),
    )


_CELL_OFFSET = 1 << 20  # keeps packed cell coordinates non-negative


def _cell_keys(cells: np.ndarray) -> np.ndarray:
    """Pack integer (cx, cy) grid cells into one sortable int64 each."""

    cells = np.asarray(cells, dtype=np.int64)
    return ((cells[:, 0] + _CELL_OFFSET) << 22) | (cells[:, 1] + _CELL_OFFSET)


def _clusters(cells: np.ndarray, allowed_keys: np.ndarray) -> list[np.ndarray]:
    """8-connected components over occupied grid cells, as point-index arrays.

    Only the flood fill over *distinct* cells runs in Python (a few hundred);
    everything per-point is vectorised.
    """

    keys = _cell_keys(cells)
    kept = np.flatnonzero(np.isin(keys, allowed_keys))
    if not len(kept):
        return []
    unique, inverse = np.unique(keys[kept], return_inverse=True)
    number = {int(key): index for index, key in enumerate(unique)}
    steps = [(dx << 22) + dy for dx in (-1, 0, 1) for dy in (-1, 0, 1) if dx or dy]
    label = np.full(len(unique), -1, dtype=np.int64)
    count = 0
    for start in range(len(unique)):
        if label[start] >= 0:
            continue
        label[start] = count
        queue = deque([start])
        while queue:
            key = int(unique[queue.popleft()])
            for step in steps:
                neighbour = number.get(key + step)
                if neighbour is not None and label[neighbour] < 0:
                    label[neighbour] = count
                    queue.append(neighbour)
        count += 1
    point_label = label[inverse]
    order = np.argsort(point_label, kind="stable")
    bounds = np.searchsorted(point_label[order], np.arange(count + 1))
    return [kept[order[bounds[i]:bounds[i + 1]]] for i in range(count)]


def find_objects(arm_points, plane: TablePlane) -> list[TableObject]:
    """Group points standing on the table into objects, largest first."""

    arm = np.asarray(arm_points, dtype=np.float64).reshape(-1, 3)
    arm = arm[np.isfinite(arm).all(axis=1)]
    height = plane.height_above(arm)
    on_table = np.abs(height) < PLANE_INLIER_METRES
    above = (height > OBJECT_MIN_HEIGHT) & (height < OBJECT_MAX_HEIGHT)
    table_cells = np.unique(
        np.floor(arm[on_table, :2] / CLUSTER_CELL_METRES).astype(np.int64), axis=0)
    reach = TABLE_NEIGHBOURHOOD_CELLS
    # Only cells over (or right beside) observed tabletop count, so people and
    # chairs beyond the table edge are not reported as objects.
    spread = np.arange(-reach, reach + 1)
    offsets = np.stack(np.meshgrid(spread, spread, indexing="ij"), axis=-1).reshape(-1, 2)
    near_table = np.unique(_cell_keys(
        (table_cells[:, None, :] + offsets[None, :, :]).reshape(-1, 2)))
    points = arm[above]
    heights = height[above]
    cells = np.floor(points[:, :2] / CLUSTER_CELL_METRES).astype(int)
    objects = []
    for indices in _clusters(cells, near_table):
        if len(indices) < MIN_CLUSTER_POINTS:
            continue
        xy = points[indices, :2]
        centre_xy = 0.5 * (xy.min(axis=0) + xy.max(axis=0))
        offsets = xy - xy.mean(axis=0)
        values, vectors = np.linalg.eigh(np.cov(offsets.T) + 1e-12 * np.eye(2))
        major = vectors[:, 1]
        projected = offsets @ vectors
        extent = np.ptp(projected, axis=0)
        objects.append(
            TableObject(
                (
                    float(centre_xy[0]),
                    float(centre_xy[1]),
                    float(plane.height_at(*centre_xy)),
                ),
                float(np.quantile(heights[indices], 0.95)),
                float(extent[1]),
                float(extent[0]),
                float(math.atan2(major[1], major[0])),
                int(len(indices)),
            )
        )
    objects.sort(key=lambda item: item.points, reverse=True)
    return objects


def select_graspable(objects, near=None, max_width=0.10, max_reach=0.70,
                     min_top=0.05, max_top=0.30):
    """Pick one object a single gripper can plausibly take.

    With ``near`` (arm-frame ``(x, y)``) the closest candidate to that hint
    wins; otherwise the nearest candidate to the robot does.
    """

    candidates = [
        item for item in objects
        if item.width <= max_width
        and item.length <= 2.0 * max_width
        and min_top <= item.top <= max_top
        and 0.15 <= item.center[0] <= max_reach
        and abs(item.center[1]) <= 0.45
    ]
    if not candidates:
        return None
    if near is not None:
        return min(candidates, key=lambda item: math.hypot(
            item.center[0] - near[0], item.center[1] - near[1]))
    return min(candidates, key=lambda item: math.hypot(*item.center[:2]))


def object_near(arm_points, plane: TablePlane, xy, radius=0.06):
    """Measure the object standing at a hinted ``(x, y)`` from local points only.

    Fallback for when depth smoothing joins a small object to a neighbour and
    ``find_objects`` reports one large blob. Returns ``None`` if nothing of at
    least ``OBJECT_MIN_HEIGHT`` stands there.
    """

    arm = np.asarray(arm_points, dtype=np.float64).reshape(-1, 3)
    arm = arm[np.isfinite(arm).all(axis=1)]
    height = plane.height_above(arm)
    local = (
        (np.hypot(arm[:, 0] - xy[0], arm[:, 1] - xy[1]) <= radius)
        & (height > OBJECT_MIN_HEIGHT) & (height < OBJECT_MAX_HEIGHT)
    )
    if np.count_nonzero(local) < MIN_CLUSTER_POINTS:
        return None
    # Recentre on the local points once, then re-crop, so a hint a few
    # centimetres off still measures the whole object.
    centre = np.median(arm[local, :2], axis=0)
    local = (
        (np.hypot(arm[:, 0] - centre[0], arm[:, 1] - centre[1]) <= radius)
        & (height > OBJECT_MIN_HEIGHT) & (height < OBJECT_MAX_HEIGHT)
    )
    xy_points = arm[local, :2]
    centre_xy = 0.5 * (xy_points.min(axis=0) + xy_points.max(axis=0))
    extent = np.ptp(xy_points, axis=0)
    return TableObject(
        (float(centre_xy[0]), float(centre_xy[1]), float(plane.height_at(*centre_xy))),
        float(np.quantile(height[local], 0.95)),
        float(extent.max()),
        float(extent.min()),
        0.0,
        int(np.count_nonzero(local)),
    )


def find_box(arm_points, plane: TablePlane, objects, exclude=None, side_of=None,
             min_length=0.15, max_length=0.60, max_reach=0.60):
    """The open container on the table: a broad rim with a hollow middle.

    Depth sees a box as its walls; the floor inside sits at table level. So a
    real container has points around its edge and almost nothing standing in
    the middle, which is what separates it from a laptop or a pile of clutter.
    """

    arm = np.asarray(arm_points, dtype=np.float64).reshape(-1, 3)
    arm = arm[np.isfinite(arm).all(axis=1)]
    height = plane.height_above(arm)
    best = None
    for item in objects:
        if exclude is not None and math.dist(item.center[:2], exclude.center[:2]) < 0.05:
            continue
        if not min_length <= item.length <= max_length or not 0.03 <= item.top <= 0.22:
            continue
        reach = min(math.hypot(item.center[0], item.center[1] - 0.0975),
                    math.hypot(item.center[0], item.center[1] + 0.0975))
        if reach > max_reach:
            continue
        if side_of is not None and exclude is not None:
            if side_of * (item.center[1] - exclude.center[1]) <= 0:
                continue
        half = np.array([item.length, item.width]) / 2.0
        offset = np.abs(arm[:, :2] - np.asarray(item.center[:2]))
        inside = np.all(offset <= half * 0.45, axis=1)
        around = np.all(offset <= half * 1.05, axis=1) & ~inside
        standing_inside = np.count_nonzero(inside & (height > 0.03))
        rim = np.count_nonzero(around & (height > 0.03))
        if rim < 40 or standing_inside > 0.55 * rim:
            continue
        score = (rim, -math.dist(item.center[:2], exclude.center[:2]) if exclude else 0)
        if best is None or score > best[0]:
            best = (score, item)
    return None if best is None else best[1]
