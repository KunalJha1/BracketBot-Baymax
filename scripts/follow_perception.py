"""Depth-only person finding for person-follow: no neural network, no camera image.

Converts ``camera.points`` to robot-local (forward, left, up) and finds
person-sized clusters on a floor grid. Pure numpy, so it runs unchanged in the
laptop tests.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# camera.points uses base +y forward, +x lateral. -1 means +x points to the robot's
# right (right-handed, z up). Verified at gate G0; flip here if G0 shows otherwise.
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


def base_to_local(points_base, left_sign=BASE_LEFT_SIGN):
    """camera.points base frame (x lateral, y forward, z up) -> (forward, left, up)."""
    p = np.asarray(points_base, dtype=np.float64)
    return np.column_stack([p[:, 1], left_sign * p[:, 0], p[:, 2]])


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


def find_people(points_local, cfg=ClusterConfig()):
    """Person-sized clusters in a robot-local cloud, nearest-first."""
    p = np.asarray(points_local, dtype=np.float64).reshape(-1, 3)
    f, l, z = p[:, 0], p[:, 1], p[:, 2]
    keep = (
        (z >= cfg.z_min) & (z <= cfg.z_max)
        & (f >= cfg.forward_min) & (f <= cfg.forward_max) & (np.abs(l) <= cfg.left_max)
    )
    p = p[keep]
    if len(p) == 0:
        return []
    rows = int(np.ceil((cfg.forward_max - cfg.forward_min) / cfg.cell)) + 1
    cols = int(np.ceil(2 * cfg.left_max / cfg.cell)) + 1
    gi = np.clip(((p[:, 0] - cfg.forward_min) / cfg.cell).astype(int), 0, rows - 1)
    gj = np.clip(((p[:, 1] + cfg.left_max) / cfg.cell).astype(int), 0, cols - 1)
    counts = np.bincount(gi * cols + gj, minlength=rows * cols).reshape(rows, cols)
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
        torso = sel[(sel[:, 2] >= cfg.torso_z[0]) & (sel[:, 2] <= cfg.torso_z[1])]
        body = torso if len(torso) >= cfg.min_torso_points else sel
        people.append(Cluster(float(np.median(body[:, 0])), float(np.median(body[:, 1])),
                              len(sel), top, depth, width))
    return sorted(people, key=lambda c: np.hypot(c.forward, c.left))
