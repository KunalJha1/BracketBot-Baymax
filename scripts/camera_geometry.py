"""Head-camera geometry shared by pointing (and later, follow and pick) runners.

Pure numpy so it can be unit-tested away from the robot and uploaded to
``/tmp`` next to the runners that import it.

Frames:

* **camera** (optical): x right, y down, z out of the lens.
* **points**: the depth daemon's ``camera.points`` base frame, (lateral right,
  forward, up). ``Config("depth").camera_to_base_3x4`` maps camera to this.
* **arm**: the IK frame, (forward, left, up) = (points_y, -points_x, points_z).
  ``scripts/table_rest.py`` already relies on this mapping.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np


# Documented left-eye fisheye calibration and mounting (docs/robot-facts.md).
# The robot runner prefers live values from Config; these keep tests and
# fallbacks deterministic.
DEFAULT_K = np.array(
    [[447.13, 0.0, 618.11], [0.0, 447.13, 497.87], [0.0, 0.0, 1.0]], dtype=np.float64
)
DEFAULT_D = np.array([0.1287, -0.0281, 0.0, 0.0], dtype=np.float64)
DEFAULT_T_POINTS_CAMERA = np.array(
    [
        [1.000, 0.017, 0.000, 0.0],
        [0.010, -0.545, 0.839, 0.0],
        [0.015, -0.839, -0.545, 1.55],
    ],
    dtype=np.float64,
)
CHEST_HEIGHT_METRES = 1.25


def fisheye_pixel_to_ray(u: float, v: float, K=DEFAULT_K, D=DEFAULT_D) -> np.ndarray:
    """Unit camera-frame ray for a raw (distorted) fisheye pixel.

    Inverts the OpenCV equidistant model
    ``theta_d = theta * (1 + k1 t^2 + k2 t^4 + k3 t^6 + k4 t^8)`` by Newton
    iteration, so no OpenCV dependency is needed.
    """

    K = np.asarray(K, dtype=np.float64)
    k = np.zeros(4)
    coefficients = np.asarray(D, dtype=np.float64).reshape(-1)[:4]
    k[: len(coefficients)] = coefficients
    x = (float(u) - K[0, 2]) / K[0, 0]
    y = (float(v) - K[1, 2]) / K[1, 1]
    theta_d = math.hypot(x, y)
    if theta_d < 1e-12:
        return np.array([0.0, 0.0, 1.0])
    theta = theta_d
    for _ in range(20):
        t2 = theta * theta
        f = theta * (1 + t2 * (k[0] + t2 * (k[1] + t2 * (k[2] + t2 * k[3])))) - theta_d
        df = 1 + t2 * (3 * k[0] + t2 * (5 * k[1] + t2 * (7 * k[2] + t2 * 9 * k[3])))
        step = f / df
        theta -= step
        if abs(step) < 1e-12:
            break
    theta = float(np.clip(theta, 0.0, math.pi * 0.5 - 1e-6))
    scale = math.sin(theta) / theta_d
    return np.array([x * scale, y * scale, math.cos(theta)])


def ray_to_fisheye_pixel(ray: Sequence[float], K=DEFAULT_K, D=DEFAULT_D) -> tuple[float, float]:
    """Project a camera-frame ray to a raw fisheye pixel (inverse of the above)."""

    K = np.asarray(K, dtype=np.float64)
    k = np.zeros(4)
    coefficients = np.asarray(D, dtype=np.float64).reshape(-1)[:4]
    k[: len(coefficients)] = coefficients
    x, y, z = (float(value) for value in ray)
    radius = math.hypot(x, y)
    if radius < 1e-12:
        return float(K[0, 2]), float(K[1, 2])
    theta = math.atan2(radius, z)
    t2 = theta * theta
    theta_d = theta * (1 + t2 * (k[0] + t2 * (k[1] + t2 * (k[2] + t2 * k[3]))))
    return (
        float(K[0, 0] * theta_d * x / radius + K[0, 2]),
        float(K[1, 1] * theta_d * y / radius + K[1, 2]),
    )


def camera_ray_to_points(ray, T_points_camera=DEFAULT_T_POINTS_CAMERA, rectify=None):
    """Return ``(origin, unit direction)`` of a camera ray in the points frame.

    ``rectify`` is the stereo rectification rotation (``R1``) when the
    extrinsic describes the rectified camera rather than the raw eye.
    """

    T = np.asarray(T_points_camera, dtype=np.float64)
    direction = np.asarray(ray, dtype=np.float64)
    if rectify is not None:
        direction = np.asarray(rectify, dtype=np.float64) @ direction
    direction = T[:3, :3] @ direction
    return T[:3, 3].copy(), direction / np.linalg.norm(direction)


def points_to_arm(points) -> np.ndarray:
    """Convert (lateral right, forward, up) rows to arm (forward, left, up)."""

    values = np.asarray(points, dtype=np.float64)
    return np.stack((values[..., 1], -values[..., 0], values[..., 2]), axis=-1)


def range_along_ray(origin, direction, cloud, cone_degrees=3.0, min_points=8,
                    min_range=0.3, max_range=6.0):
    """Robust range to the nearest surface along a ray, or ``None``.

    ``cloud`` is an ``(N, 3)`` array in the same frame as the ray. Points inside
    the cone are histogrammed by range and the nearest well-supported 10 cm bin
    wins, so a person in front of a wall picks the person, not the wall.
    """

    cloud = np.asarray(cloud, dtype=np.float64).reshape(-1, 3)
    cloud = cloud[np.isfinite(cloud).all(axis=1)]
    if not len(cloud):
        return None
    offsets = cloud - np.asarray(origin, dtype=np.float64)
    ranges = np.linalg.norm(offsets, axis=1)
    valid = (ranges >= min_range) & (ranges <= max_range)
    offsets, ranges = offsets[valid], ranges[valid]
    if not len(ranges):
        return None
    cosine = offsets @ np.asarray(direction, dtype=np.float64) / ranges
    inside = ranges[cosine >= math.cos(math.radians(cone_degrees))]
    if len(inside) < min_points:
        return None
    bins = np.floor(inside / 0.10).astype(np.int64)
    for bin_id in np.unique(bins):
        members = inside[np.abs(bins - bin_id) <= 1]
        if len(members) >= min_points:
            return float(np.median(members))
    return None


def intersect_height(origin, direction, height):
    """Distance along a ray to the horizontal plane ``z = height``, or ``None``."""

    dz = float(direction[2])
    if abs(dz) < 1e-6:
        return None
    distance = (height - float(origin[2])) / dz
    return distance if distance > 0 else None


def target_point(origin, direction, cloud=None, fallback_height=CHEST_HEIGHT_METRES,
                 fallback_range=2.0):
    """Best 3D point for a ray: depth hit, else a chest-height plane, else 2 m.

    Returns ``(point, source)`` where ``source`` names the evidence used.
    """

    origin = np.asarray(origin, dtype=np.float64)
    direction = np.asarray(direction, dtype=np.float64)
    if cloud is not None:
        distance = range_along_ray(origin, direction, cloud)
        if distance is not None:
            return origin + distance * direction, "depth"
    distance = intersect_height(origin, direction, fallback_height)
    if distance is not None and distance <= 6.0:
        return origin + distance * direction, "plane-fallback"
    return origin + fallback_range * direction, "range-fallback"


def quaternion_from_z(direction) -> np.ndarray:
    """Return an xyzw quaternion whose local Z axis follows ``direction``."""

    target = np.asarray(direction, dtype=np.float64)
    norm = float(np.linalg.norm(target))
    if not np.isfinite(target).all() or norm < 1e-9:
        raise ValueError("pointing direction must be finite and non-zero")
    target = target / norm
    source = np.array([0.0, 0.0, 1.0])
    dot = float(np.clip(np.dot(source, target), -1.0, 1.0))
    if dot < -1.0 + 1e-9:
        return np.array([1.0, 0.0, 0.0, 0.0])
    cross = np.cross(source, target)
    quaternion = np.array([cross[0], cross[1], cross[2], 1.0 + dot])
    return quaternion / np.linalg.norm(quaternion)


def quaternion_slerp(start, end, alpha) -> np.ndarray:
    """Shortest-path spherical interpolation for xyzw quaternions."""

    q0 = np.asarray(start, dtype=np.float64)
    q1 = np.asarray(end, dtype=np.float64)
    q0 = q0 / np.linalg.norm(q0)
    q1 = q1 / np.linalg.norm(q1)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1, dot = -q1, -dot
    dot = float(np.clip(dot, -1.0, 1.0))
    alpha = float(np.clip(alpha, 0.0, 1.0))
    if dot > 0.9995:
        result = q0 + alpha * (q1 - q0)
        return result / np.linalg.norm(result)
    angle = math.acos(dot)
    result = (
        math.sin((1.0 - alpha) * angle) * q0 + math.sin(alpha * angle) * q1
    ) / math.sin(angle)
    return result / np.linalg.norm(result)


def point_along(shoulder, target, reach):
    """Hand position ``reach`` metres from ``shoulder`` toward ``target``."""

    shoulder = np.asarray(shoulder, dtype=np.float64)
    ray = np.asarray(target, dtype=np.float64) - shoulder
    distance = float(np.linalg.norm(ray))
    if distance < 1e-6:
        raise ValueError("target coincides with the shoulder")
    direction = ray / distance
    return shoulder + reach * direction, direction
