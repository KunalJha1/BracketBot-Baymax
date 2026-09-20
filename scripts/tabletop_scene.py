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

from dataclasses import dataclass, asdict
import math

import numpy as np


PLANE_INLIER_METRES = 0.012
MAX_TABLE_TILT_DEGREES = 20.0
RANSAC_SCORE_POINTS = 15000
RANSAC_BATCH = 50            # hypotheses scored per matrix product (keeps it ~6 MB)
REGION_X = (0.05, 1.20)
REGION_Y = 0.80
REGION_Z = (0.45, 1.15)
OBJECT_MIN_HEIGHT = 0.02
OBJECT_MAX_HEIGHT = 0.45
CLUSTER_CELL_METRES = 0.02
MIN_CLUSTER_POINTS = 25
# Stereo depth smears a low skirt (2-4 cm) around and between things standing on
# the table. It joins neighbouring cans into one blob and stretches every
# footprint toward the camera, so objects are told apart and measured by the
# part of them above this share of their height, which the skirt never reaches.
UPPER_SHARE = 0.5
MIN_UPPER_POINTS = 12
SKIRT_METRES = 0.04
SPLIT_STEP_METRES = 0.005
MIN_PROMINENCE_METRES = 0.015
MIN_PART_THICKNESS_METRES = 0.025
MAX_SMALL_THING_METRES = 0.15
MAX_ELONGATION = 2.2             # a lying can is 1.8; a stretch of box rim is 3 or more
# Finer than the ground clustering: lids of cans standing a finger apart must
# not touch, and up here there is no skirt to bridge sparse depth.
UPPER_CELL_METRES = 0.01
OUTLINE_MARGIN_METRES = 0.015
# A lower object left behind when taller ones are cut out of a cluster (the box
# the cans lean on) has to be this substantial, or it is only their skirt.
MIN_LEFT_BEHIND_POINTS = 60
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
    # Draw every hypothesis first (the same draws as one at a time), then
    # score them together: one matrix product per batch, not one per plane.
    picks = np.array([rng.choice(len(candidates), 3, replace=False) for _ in range(iterations)])
    corners = candidates[picks]
    normals = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    lengths = np.linalg.norm(normals, axis=1)
    usable = lengths >= 1e-9
    normals[usable] /= lengths[usable, None]
    normals[normals[:, 2] < 0] *= -1.0
    usable &= normals[:, 2] >= min_nz
    if not usable.any():
        raise RuntimeError("no near-horizontal plane found")
    origins, normals = corners[usable, 0], normals[usable]
    offsets = np.einsum("ij,ij->i", origins, normals)
    counts = np.concatenate([
        np.count_nonzero(np.abs(scored @ normals[first:first + RANSAC_BATCH].T
                                - offsets[first:first + RANSAC_BATCH]) < PLANE_INLIER_METRES, axis=0)
        for first in range(0, len(normals), RANSAC_BATCH)])
    winner = int(np.argmax(counts))  # the first of equals, as a running best would keep
    if counts[winner] == 0:
        raise RuntimeError("no near-horizontal plane found")
    origin, normal = origins[winner], normals[winner]
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


def _label_cells(unique: np.ndarray) -> tuple[np.ndarray, int]:
    """Component number per distinct (sorted) cell key, numbered by first cell."""

    # A scan labels ~300 cell sets, and a Python flood fill over them was a
    # quarter of it. Each cell looks up its four "forward" neighbours in the sorted keys, then
    # every cell takes the smallest index it is joined to until nothing changes.
    size = len(unique)
    ahead = np.array([1, (1 << 22) - 1, 1 << 22, (1 << 22) + 1], dtype=np.int64)
    wanted = unique[:, None] + ahead[None, :]
    found = np.minimum(np.searchsorted(unique, wanted), size - 1)
    joined = unique[found] == wanted
    near = np.repeat(np.arange(size), len(ahead))[joined.ravel()]
    far = found.ravel()[joined.ravel()]
    label = np.arange(size)
    while True:
        lowest = np.minimum(label[near], label[far])
        updated = label.copy()
        np.minimum.at(updated, near, lowest)
        np.minimum.at(updated, far, lowest)
        updated = updated[updated]  # follow each cell's pointer one step further
        if np.array_equal(updated, label):
            break
        label = updated
    names, label = np.unique(label, return_inverse=True)
    return label, len(names)


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
    label, count = _label_cells(unique)
    point_label = label[inverse]
    order = np.argsort(point_label, kind="stable")
    bounds = np.searchsorted(point_label[order], np.arange(count + 1))
    return [kept[order[bounds[i]:bounds[i + 1]]] for i in range(count)]


def _extents(xy: np.ndarray) -> tuple[float, float]:
    """Footprint (major, minor) extent along its principal axes."""

    offsets = xy - xy.mean(axis=0)
    vectors = np.linalg.eigh(np.cov(offsets.T) + 1e-12 * np.eye(2))[1]
    spans = np.ptp(offsets @ vectors, axis=0)
    return float(spans[1]), float(spans[0])


def _convex_outline(xy: np.ndarray) -> np.ndarray:
    """Counter-clockwise convex hull (monotone chain) of 2-D points."""

    points = np.unique(np.round(xy, 3), axis=0)
    if len(points) < 3:
        return points

    def half(sequence):
        # Plain floats: two-element numpy arithmetic per point was a quarter of a scan.
        chain = []
        for x, y in sequence:
            while len(chain) >= 2:
                (px, py), (qx, qy) = chain[-2], chain[-1]
                if (qx - px) * (y - py) - (qy - py) * (x - px) > 0:
                    break
                chain.pop()
            chain.append((x, y))
        return chain[:-1]

    ordered = points.tolist()
    return np.array(half(ordered) + half(ordered[::-1]))


def _footprint(xy: np.ndarray) -> tuple[np.ndarray, float, float, float]:
    """Centre, length, width and heading of the smallest rectangle around ``xy``.

    Principal axes are arbitrary for anything near square, which is what the
    box is: they turned it by whatever the noise said and overstated its size.
    The smallest enclosing rectangle always lies along one edge of the outline.
    """

    outline = _convex_outline(xy)
    if len(outline) < 3:
        centre = 0.5 * (xy.min(axis=0) + xy.max(axis=0))
        spans = np.ptp(xy, axis=0)
        return centre, float(spans.max()), float(spans.min()), 0.0 if spans[0] >= spans[1] else math.pi / 2
    edges = np.roll(outline, -1, axis=0) - outline
    headings = np.unique(np.round(np.arctan2(edges[:, 1], edges[:, 0]) % (math.pi / 2), 4))
    best = None
    for heading in headings:
        axes = np.array([[math.cos(heading), math.sin(heading)],
                         [-math.sin(heading), math.cos(heading)]])
        along = outline @ axes.T
        low, high = along.min(axis=0), along.max(axis=0)
        spans = high - low
        if best is None or spans[0] * spans[1] < best[0]:
            best = (spans[0] * spans[1], heading, spans, 0.5 * (low + high) @ axes)
    _, heading, spans, centre = best
    if spans[1] > spans[0]:
        heading += math.pi / 2
    return centre, float(spans.max()), float(spans.min()), float(heading)


def _outside_by(outline: np.ndarray, spot: np.ndarray) -> float:
    """Distance of ``spot`` outside a convex outline; negative when inside."""

    if len(outline) < 3:
        return float("inf")
    edges = np.roll(outline, -1, axis=0) - outline
    outward = np.column_stack((edges[:, 1], -edges[:, 0]))
    outward /= np.maximum(np.linalg.norm(outward, axis=1, keepdims=True), 1e-12)
    return float(np.max(np.sum(outward * (spot - outline), axis=1)))


def _upper_parts(xy, heights, candidates, level):
    """Connected groups of ``candidates`` standing above ``level``."""

    high = candidates[heights[candidates] > level]
    if len(high) < MIN_UPPER_POINTS:
        return []
    cells = np.floor(xy[high] / UPPER_CELL_METRES).astype(np.int64)
    return [
        high[part] for part in _clusters(cells, np.unique(_cell_keys(cells)))
        if len(part) >= MIN_UPPER_POINTS
    ]


def _small_thing(xy, heights, part, level) -> bool:
    """Whether ``part`` is a can-sized thing of its own rather than a piece of something.

    It has to stand clearly above the cut, which a box rim breaking up near its
    top does not, and have a can's footprint: not a sliver of wall seen edge-on,
    not a long run of it either.
    """

    if np.quantile(heights[part], 0.95) - level < MIN_PROMINENCE_METRES:
        return False
    major, minor = _extents(xy[part])
    return (
        minor >= MIN_PART_THICKNESS_METRES
        and major <= MAX_SMALL_THING_METRES
        and major <= MAX_ELONGATION * minor
    )


def _split_by_upper_parts(xy: np.ndarray, heights: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """Free the small things standing in one ground-level cluster.

    Returns ``(members, upper)`` index arrays per thing: every point that belongs
    to it, and the ones high enough to measure it by. The cut height is raised
    a step at a time; cans joined by skirt come apart at half their height, cans
    leaning on the box only above its rim. A freed thing that turns out to be
    two is replaced by them. Whatever is left (the box) stays one object when
    it is more than skirt. A cluster with nothing to free comes back whole.
    """

    everything = np.arange(len(xy))
    top = float(np.quantile(heights, 0.95))
    base = max(OBJECT_MIN_HEIGHT, UPPER_SHARE * top)
    whole = _upper_parts(xy, heights, everything, base)
    # A lone can is already one small thing at the base cut: nothing to free.
    if len(whole) == 1 and _small_thing(xy, heights, whole[0], base):
        return [(everything, whole[0])]
    owner = np.full(len(xy), -1)          # which freed thing each upper point is in
    things: dict[int, np.ndarray] = {}
    for level in np.arange(base, top - SPLIT_STEP_METRES, SPLIT_STEP_METRES):
        inside: dict[int, list[np.ndarray]] = {}
        for part in _upper_parts(xy, heights, everything, level):
            if not _small_thing(xy, heights, part, level):
                continue
            inside.setdefault(int(owner[part[0]]), []).append(part)
        for parent, parts in inside.items():
            # One part inside a thing is that thing again, seen higher up and
            # so with less of whatever it leans on still attached; two are
            # two things. Either way the higher look replaces the lower one.
            things.pop(parent, None)
            for part in parts:
                number = max(things, default=-1) + 1
                things[number] = part
                owner[part] = number
    if not things:
        return [(everything, np.concatenate(whole) if whole else everything)]
    taken = np.zeros(len(xy), dtype=bool)
    pieces = []
    for upper in things.values():
        low, high = xy[upper].min(axis=0), xy[upper].max(axis=0)
        under = np.all(
            (xy >= low - CLUSTER_CELL_METRES) & (xy <= high + CLUSTER_CELL_METRES), axis=1)
        pieces.append((np.flatnonzero(under & ~taken), upper))
        taken |= under
    rest = np.flatnonzero(~taken)
    rest_top = float(np.quantile(heights[rest], 0.95)) if len(rest) else 0.0
    if rest_top <= SKIRT_METRES:
        return pieces
    left = _upper_parts(xy, heights, rest, max(SKIRT_METRES, UPPER_SHARE * rest_top))
    if sum(len(part) for part in left) < MIN_LEFT_BEHIND_POINTS:
        return pieces
    # A taller stretch of the box's own rim looks like a small thing too. What
    # stands within the outline of what is left belongs to it: a can leaning on
    # the box has its centre a radius outside the walls, a piece of wall is on them.
    body = np.concatenate(left)
    outline = _convex_outline(xy[body])
    free = []
    for members, upper in pieces:
        spot = 0.5 * (xy[upper].min(axis=0) + xy[upper].max(axis=0))
        if _outside_by(outline, spot) <= OUTLINE_MARGIN_METRES:
            rest = np.union1d(rest, members)
            body = np.union1d(body, upper)
        else:
            free.append((members, upper))
    return free + [(rest, body)]


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
    for cluster in _clusters(cells, near_table):
        if len(cluster) < MIN_CLUSTER_POINTS:
            continue
        for members, upper in _split_by_upper_parts(points[cluster, :2], heights[cluster]):
            if len(members) < MIN_CLUSTER_POINTS:
                continue
            centre_xy, length, width, yaw = _footprint(points[cluster[upper], :2])
            objects.append(
                TableObject(
                    (
                        float(centre_xy[0]),
                        float(centre_xy[1]),
                        float(plane.height_at(*centre_xy)),
                    ),
                    float(np.quantile(heights[cluster[members]], 0.95)),
                    length,
                    width,
                    yaw,
                    int(len(members)),
                )
            )
    objects.sort(key=lambda item: item.points, reverse=True)
    return objects


def graspable_candidates(objects, max_width=0.10, max_reach=0.70, min_top=0.05, max_top=0.30):
    """Every object a single gripper can plausibly take, nearest the robot first."""

    candidates = [
        item for item in objects
        if item.width <= max_width
        and item.length <= 2.0 * max_width
        and min_top <= item.top <= max_top
        and 0.15 <= item.center[0] <= max_reach
        and abs(item.center[1]) <= 0.45
    ]
    return sorted(candidates, key=lambda item: math.hypot(*item.center[:2]))


def select_graspable(objects, near=None, max_width=0.10, max_reach=0.70,
                     min_top=0.05, max_top=0.30):
    """Pick one object a single gripper can plausibly take.

    With ``near`` (arm-frame ``(x, y)``) the closest candidate to that hint
    wins; otherwise the nearest candidate to the robot does.
    """

    candidates = graspable_candidates(objects, max_width, max_reach, min_top, max_top)
    if not candidates:
        return None
    if near is not None:
        return min(candidates, key=lambda item: math.hypot(
            item.center[0] - near[0], item.center[1] - near[1]))
    return candidates[0]


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
        # Measure along the box's own axes: its long side is rarely along x.
        major = np.array([math.cos(item.yaw), math.sin(item.yaw)])
        axes = np.column_stack((major, [-major[1], major[0]]))
        offset = np.abs((arm[:, :2] - np.asarray(item.center[:2])) @ axes)
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
