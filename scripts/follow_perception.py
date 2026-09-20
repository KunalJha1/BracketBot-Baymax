"""Geometric person candidates with optional aligned point-colour appearance.

Converts ``camera.points`` to robot-local (forward, left, up) and finds
person-sized clusters on a floor grid. Pure numpy, so it runs unchanged in the
laptop tests.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# camera.points uses base +y forward, +x lateral. -1 means +x points to the robot's
# right (right-handed, z up). Unverified diagnostic default; motion uses the robot's calibration.
BASE_LEFT_SIGN = -1.0


@dataclass(frozen=True)
class ClusterConfig:
    cell: float = 0.10  # floor-grid cell size (m)
    min_cell_points: int = 3  # a cell with fewer points is empty
    z_min: float = 0.10  # drops the floor
    z_max: float = 2.0  # drops the ceiling
    forward_min: float = 0.3
    forward_max: float = 3.5
    left_max: float = 2.5
    max_footprint: float = 0.8  # wider or deeper than this is not a person (m)
    min_footprint: float = 0.15  # the larger extent must reach this (m)
    min_top: float = 1.2  # highest point must reach this (m)
    min_points: int = 60
    torso_z: tuple[float, float] = (0.8, 1.6)
    min_torso_points: int = 10


@dataclass(frozen=True)
class Cluster:
    forward: float  # torso median (m); the surface the camera sees
    left: float
    points: int
    top: float  # highest point (m)
    depth: float  # footprint extent along forward (m)
    width: float  # footprint extent along left (m)
    hist: np.ndarray | None = None


def base_to_local(points_base, left_sign=BASE_LEFT_SIGN):
    """camera.points base frame (x lateral, y forward, z up) -> (forward, left, up)."""
    p = np.asarray(points_base, dtype=np.float64)
    return np.column_stack([p[:, 1], left_sign * p[:, 0], p[:, 2]])


def outside_self_mask(points_local, boxes):
    """Selection shared by geometry and its aligned colours."""
    p = np.asarray(points_local, dtype=np.float64).reshape(-1, 3)
    keep = np.ones(len(p), dtype=bool)
    for f0, f1, l0, l1, z0, z1 in boxes:
        keep &= ~((p[:, 0] >= f0) & (p[:, 0] <= f1)
                  & (p[:, 1] >= l0) & (p[:, 1] <= l1)
                  & (p[:, 2] >= z0) & (p[:, 2] <= z1))
    return keep


def without_self(points_local, boxes):
    """Remove only physically reviewed robot-body boxes."""
    p = np.asarray(points_local, dtype=np.float64).reshape(-1, 3)
    return p[outside_self_mask(p, boxes)]


def clothing_histogram(colors):
    """Coarse chromaticity/brightness cue, not a person ID or a learned ReID model.

    BBOS point colours must be aligned uint8 RGB triplets. Unknown formats are
    ignored, so a device schema change cannot manufacture a false colour cue.
    """
    if colors is None:
        return None
    rgb = np.asarray(colors)
    if rgb.dtype != np.uint8 or rgb.ndim != 2 or rgb.shape[1] != 3 or len(rgb) < 10:
        return None
    rgb = rgb.astype(float)
    total = np.maximum(rgb.sum(axis=1), 1)
    features = np.column_stack([rgb[:, 0] / total, rgb[:, 1] / total, rgb.max(axis=1) / 256])
    # Soft bins avoid changing the entire descriptor when exposure crosses one
    # brightness-bin boundary. Keep chromaticity separate from brightness.
    coordinates = np.clip(features * 4 - .5, 0, 3)
    lower = np.floor(coordinates).astype(int)
    fraction = coordinates - lower
    hist = np.zeros(64)
    for dr in (0, 1):
        for dg in (0, 1):
            for dv in (0, 1):
                offset = np.array([dr, dg, dv])
                bins = np.minimum(lower + offset, 3)
                weight = np.prod(np.where(offset, fraction, 1 - fraction), axis=1)
                hist += np.bincount(bins[:, 0] * 16 + bins[:, 1] * 4 + bins[:, 2], weights=weight, minlength=64)
    return hist / hist.sum()


def _label(occupied):
    """8-connected component labels of a boolean grid; 0 is background."""
    labels = np.zeros(occupied.shape, dtype=int)
    rows, cols = occupied.shape
    current = 0
    for start in zip(*np.nonzero(occupied)):
        if labels[start]:
            continue
        current += 1
        labels[start] = current
        stack = [start]
        while stack:
            i, j = stack.pop()
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    ni, nj = i + di, j + dj
                    if 0 <= ni < rows and 0 <= nj < cols and occupied[ni, nj] and not labels[ni, nj]:
                        labels[ni, nj] = current
                        stack.append((ni, nj))
    return labels


def find_people(points_local, cfg=ClusterConfig(), colors=None):
    """Person-sized clusters in a robot-local cloud, nearest-first."""
    p = np.asarray(points_local, dtype=np.float64).reshape(-1, 3)
    if colors is not None:
        colors = np.asarray(colors)
        if colors.dtype != np.uint8 or colors.shape != p.shape:
            colors = None
    f, l, z = p[:, 0], p[:, 1], p[:, 2]
    keep = (
        (z >= cfg.z_min) & (z <= cfg.z_max)
        & (f >= cfg.forward_min) & (f <= cfg.forward_max) & (np.abs(l) <= cfg.left_max)
    )
    p = p[keep]
    if colors is not None:
        colors = colors[keep]
    if len(p) == 0:
        return []
    rows = int(np.ceil((cfg.forward_max - cfg.forward_min) / cfg.cell)) + 1
    cols = int(np.ceil(2 * cfg.left_max / cfg.cell)) + 1
    gi = np.clip(((p[:, 0] - cfg.forward_min) / cfg.cell).astype(int), 0, rows - 1)
    gj = np.clip(((p[:, 1] + cfg.left_max) / cfg.cell).astype(int), 0, cols - 1)
    counts = np.zeros((rows, cols), dtype=int)
    np.add.at(counts, (gi, gj), 1)
    labels = _label(counts >= cfg.min_cell_points)
    point_labels = labels[gi, gj]

    people = []
    for k in range(1, labels.max() + 1):
        sel = p[point_labels == k]
        if len(sel) < cfg.min_points:
            continue
        depth = float(sel[:, 0].max() - sel[:, 0].min())
        width = float(sel[:, 1].max() - sel[:, 1].min())
        if depth > cfg.max_footprint or width > cfg.max_footprint or max(depth, width) < cfg.min_footprint:
            continue
        top = float(sel[:, 2].max())
        if top < cfg.min_top:
            continue
        torso_mask = (sel[:, 2] >= cfg.torso_z[0]) & (sel[:, 2] <= cfg.torso_z[1])
        torso = sel[torso_mask]
        body = torso if len(torso) >= cfg.min_torso_points else sel
        hist = None if colors is None else clothing_histogram(colors[point_labels == k][torso_mask])
        people.append(Cluster(float(np.median(body[:, 0])), float(np.median(body[:, 1])),
                              len(sel), top, depth, width, hist))
    return sorted(people, key=lambda c: np.hypot(c.forward, c.left))
