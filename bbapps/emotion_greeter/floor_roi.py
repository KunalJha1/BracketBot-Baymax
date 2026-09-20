"""Second look at the floor ahead, at the raw camera's resolution.

``camera.rect`` is 512x384 and the pose model shrinks it to 320 px, so a body on
the floor a few metres out is a handful of pixels and is simply not detected:
on the robot the detector reported ``people=0`` while the same model, fed the
raw 1280x960 eye, found the person. Running the model on the whole raw frame
costs ~0.5 s on this CPU. A 640x480 crop of the floor 1.6-6 m ahead shrinks to
320x240 instead, so it has the raw frame's pixel density at the small model's
price, and covers exactly the range the depth cloud (<= 1.7 m) cannot.

Detections are mapped into ``camera.rect`` pixels, so tracking, depth lookup,
the floor-plane test and the approach target all work on them unchanged.
"""

from __future__ import annotations

import cv2
import numpy as np

# Left eye of camera.head.rgb (docs/robot-facts.md).
RAW_EYE_SIZE = (1280, 960)
RAW_CAMERA_MATRIX = np.array(
    [[447.13, 0.0, 618.11], [0.0, 447.13, 497.87], [0.0, 0.0, 1.0]], dtype=np.float64
)
RAW_FISHEYE_D = np.array([0.1287, -0.0281, 0.0, 0.0], dtype=np.float64)
# Undistorted normalised raw coords [x, y, 1] -> camera.rect pixel. Fitted with
# RANSAC on 3.6k SIFT matches between simultaneous raw and rect frames: 0.56 px
# median error, with inliers from row 13 to row 379.
RAW_TO_RECT = np.array(
    [
        [133.01909, -7.66663, 235.06428],
        [2.9198, 125.94783, 201.60119],
        [0.00799, -0.02641, 1.0],
    ],
    dtype=np.float64,
)
# x0, y0, x1, y1 in the raw left eye: floor from ~1.6 m to ~6 m ahead.
FLOOR_ROI = (320, 130, 960, 610)


def left_eye(head_rgb: np.ndarray) -> np.ndarray:
    """The left 1280x960 eye of the side-by-side stereo frame."""

    return head_rgb[:, : head_rgb.shape[1] // 2]


def floor_crop(eye: np.ndarray) -> np.ndarray:
    x0, y0, x1, y1 = FLOOR_ROI
    return eye[y0:y1, x0:x1]


def raw_pixels_to_rect(pixels: np.ndarray) -> np.ndarray:
    """Raw left-eye pixels -> camera.rect pixels (NaN behind the rect camera)."""

    pixels = np.asarray(pixels, dtype=np.float64).reshape(-1, 1, 2)
    if not len(pixels):
        return np.empty((0, 2))
    normalised = cv2.fisheye.undistortPoints(pixels, RAW_CAMERA_MATRIX, RAW_FISHEYE_D)
    homogeneous = np.column_stack([normalised.reshape(-1, 2), np.ones(len(pixels))])
    projected = homogeneous @ RAW_TO_RECT.T
    rect = np.full((len(pixels), 2), np.nan)
    front = projected[:, 2] > 1e-6
    rect[front] = projected[front, :2] / projected[front, 2:3]
    return rect


def crop_pixels_to_rect(pixels: np.ndarray) -> np.ndarray:
    """Pixels of ``floor_crop`` -> camera.rect pixels."""

    pixels = np.asarray(pixels, dtype=np.float64).reshape(-1, 2)
    return raw_pixels_to_rect(pixels + np.array(FLOOR_ROI[:2], dtype=np.float64))
