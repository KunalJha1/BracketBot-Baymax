"""Robot-side perception for person-follow: pose engine, decoding, and torso position.

Everything here is numpy except ``PoseEngine``, which imports TensorRT and
PyCUDA only when constructed on the Jetson. Decode and geometry are unit-tested
on a laptop.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

NOSE, L_SHOULDER, R_SHOULDER, L_WRIST, R_WRIST, L_HIP, R_HIP = 0, 5, 6, 9, 10, 11, 12
PAD_VALUE = 114
# camera.points uses base +y forward, +x lateral. -1 means +x points to the robot's
# right (right-handed, z up). Verified at gate G0; flip here if G0 shows otherwise.
BASE_LEFT_SIGN = -1.0


@dataclass(frozen=True, eq=False)
class PoseDetection:
    box: np.ndarray  # (4,) x1, y1, x2, y2 in source-image pixels
    score: float
    keypoints: np.ndarray  # (17, 3) x, y, confidence in source-image pixels


@dataclass(frozen=True)
class Letterbox:
    scale: float
    pad_x: int
    pad_y: int


def _resize(image, width, height):
    try:
        import cv2
    except ImportError:  # laptop without OpenCV: nearest neighbour is enough for tests
        rows = np.minimum((np.arange(height) + 0.5) * image.shape[0] / height, image.shape[0] - 1).astype(int)
        cols = np.minimum((np.arange(width) + 0.5) * image.shape[1] / width, image.shape[1] - 1).astype(int)
        return image[rows][:, cols]
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)


def letterbox(image, size=640):
    """RGB uint8 (H, W, 3) -> float32 (1, 3, size, size) in [0, 1], and the mapping back."""
    h, w = image.shape[:2]
    scale = min(size / h, size / w)
    nh, nw = round(h * scale), round(w * scale)
    pad_y, pad_x = (size - nh) // 2, (size - nw) // 2
    canvas = np.full((size, size, 3), PAD_VALUE, dtype=np.uint8)
    canvas[pad_y:pad_y + nh, pad_x:pad_x + nw] = _resize(image, nw, nh)
    tensor = canvas.transpose(2, 0, 1)[None].astype(np.float32) / 255.0
    return np.ascontiguousarray(tensor), Letterbox(scale, pad_x, pad_y)


def nms(boxes, scores, iou_threshold):
    order = np.argsort(-scores)
    keep = []
    while len(order):
        i = order[0]
        keep.append(int(i))
        rest = order[1:]
        x1 = np.maximum(boxes[i, 0], boxes[rest, 0])
        y1 = np.maximum(boxes[i, 1], boxes[rest, 1])
        x2 = np.minimum(boxes[i, 2], boxes[rest, 2])
        y2 = np.minimum(boxes[i, 3], boxes[rest, 3])
        inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
        area = lambda b: (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])  # noqa: E731
        iou = inter / (area(boxes[i]) + area(boxes[rest]) - inter + 1e-9)
        order = rest[iou <= iou_threshold]
    return keep


def decode_pose(raw, lb, conf=0.40, iou=0.5):
    """YOLO pose output (1, 56, N) -> detections in source-image pixels, best first."""
    preds = np.asarray(raw, dtype=np.float32)
    if preds.ndim == 3:
        preds = preds[0]
    if preds.shape[0] == 56:
        preds = preds.T
    preds = preds[preds[:, 4] >= conf]
    if len(preds) == 0:
        return []
    cx, cy, w, h = preds[:, 0], preds[:, 1], preds[:, 2], preds[:, 3]
    boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)
    keypoints = preds[:, 5:].reshape(-1, 17, 3).copy()
    offset = np.array([lb.pad_x, lb.pad_y, lb.pad_x, lb.pad_y], dtype=np.float32)
    detections = []
    for i in nms(boxes, preds[:, 4], iou):
        kp = keypoints[i]
        kp[:, 0] = (kp[:, 0] - lb.pad_x) / lb.scale
        kp[:, 1] = (kp[:, 1] - lb.pad_y) / lb.scale
        detections.append(PoseDetection((boxes[i] - offset) / lb.scale, float(preds[i, 4]), kp))
    return detections


def hand_raised(det, min_conf=0.5, margin_frac=0.10):
    """Either wrist above the nose by at least ``margin_frac`` of the person's box height."""
    kp = det.keypoints
    if kp[NOSE, 2] < min_conf:
        return False
    margin = margin_frac * (det.box[3] - det.box[1])
    return any(kp[w, 2] >= min_conf and kp[NOSE, 1] - kp[w, 1] >= margin for w in (L_WRIST, R_WRIST))


def torso_rect(det, min_conf=0.5):
    """Shoulder-hip rectangle, or the middle third of the box when keypoints are unsure."""
    kp = det.keypoints
    idx = [L_SHOULDER, R_SHOULDER, L_HIP, R_HIP]
    x1, y1, x2, y2 = (float(v) for v in det.box)
    if all(kp[i, 2] >= min_conf for i in idx):
        tx1, tx2 = float(kp[idx, 0].min()), float(kp[idx, 0].max())
        ty1, ty2 = float(kp[idx, 1].min()), float(kp[idx, 1].max())
        min_width = 0.25 * (x2 - x1)  # side-on people have overlapping shoulders
        if tx2 - tx1 < min_width:
            mid = (tx1 + tx2) / 2
            tx1, tx2 = mid - min_width / 2, mid + min_width / 2
        return tx1, ty1, tx2, ty2
    w, h = x2 - x1, y2 - y1
    return x1 + w / 3, y1 + h / 3, x2 - w / 3, y2 - h / 3


def torso_histogram(image, rect):
    """4x4x4 HSV histogram (64 bins, L1-normalised) of the torso; None if the rect is empty."""
    height, width = image.shape[:2]
    x1, y1, x2, y2 = (int(round(v)) for v in rect)
    x1, x2 = max(0, x1), min(width, x2)
    y1, y2 = max(0, y1), min(height, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    rgb = image[y1:y2, x1:x2].reshape(-1, 3).astype(np.float32) / 255.0
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    value = rgb.max(axis=1)
    delta = value - rgb.min(axis=1)
    sat = np.where(value > 0, delta / np.maximum(value, 1e-6), 0.0)
    hue = np.zeros_like(value)
    chroma = delta > 1e-6
    rmax = chroma & (value == r)
    gmax = chroma & (value == g) & ~rmax
    bmax = chroma & ~rmax & ~gmax
    hue[rmax] = ((g - b)[rmax] / delta[rmax]) % 6
    hue[gmax] = (b - r)[gmax] / delta[gmax] + 2
    hue[bmax] = (r - g)[bmax] / delta[bmax] + 4
    hist, _ = np.histogramdd(
        np.column_stack([hue / 6.0, sat, value]), bins=(4, 4, 4), range=((0, 1), (0, 1), (0, 1))
    )
    return (hist / hist.sum()).ravel()


def base_to_local(points_base, left_sign=BASE_LEFT_SIGN):
    """camera.points base frame (x lateral, y forward, z up) -> (forward, left, up)."""
    p = np.asarray(points_base, dtype=np.float64)
    return np.column_stack([p[:, 1], left_sign * p[:, 0], p[:, 2]])


def mask_to_image_pixels(mask, depth_shape, image_shape):
    """camera.points ``mask`` (flat depth-pixel indices) -> (N, 2) u, v in the detection image."""
    depth_h, depth_w = depth_shape[:2]
    image_h, image_w = image_shape[:2]
    mask = np.asarray(mask, dtype=np.int64)
    rows, cols = mask // depth_w, mask % depth_w
    return np.column_stack([cols * (image_w / depth_w), rows * (image_h / depth_h)])


def person_position(points_local, pixels, rect, min_points=40, z_range=(0.2, 2.0)):
    """Median (forward, left) of the points that land on the torso; None if too few."""
    u, v = pixels[:, 0], pixels[:, 1]
    x1, y1, x2, y2 = rect
    z = points_local[:, 2]
    inside = (u >= x1) & (u <= x2) & (v >= y1) & (v <= y2) & (z >= z_range[0]) & (z <= z_range[1])
    if np.count_nonzero(inside) < min_points:
        return None
    selected = points_local[inside]
    return float(np.median(selected[:, 0])), float(np.median(selected[:, 1]))


class PoseEngine:
    """yolo11n-pose TensorRT engine on the Jetson (TensorRT >= 8.5 tensor-address API)."""

    def __init__(self, engine_path, conf=0.40, iou=0.5):
        import pycuda.driver as cuda
        import tensorrt as trt

        cuda.init()
        self._cuda = cuda
        self._ctx = cuda.Device(0).make_context()
        logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as source, trt.Runtime(logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(source.read())
        if self.engine is None:
            raise RuntimeError(f"could not deserialize TensorRT engine {engine_path}")
        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()
        self.buffers = {}
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = tuple(self.engine.get_tensor_shape(name))
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            host = cuda.pagelocked_empty(int(np.prod(shape)), dtype)
            device = cuda.mem_alloc(host.nbytes)
            self.context.set_tensor_address(name, int(device))
            is_input = self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
            self.buffers[name] = (host, device, shape, is_input)
        self.input_name = next(n for n, b in self.buffers.items() if b[3])
        self.output_name = next(n for n, b in self.buffers.items() if not b[3])
        self.size = int(self.buffers[self.input_name][2][-1])
        self.conf, self.iou = conf, iou

    def infer(self, image):
        tensor, lb = letterbox(image, self.size)
        host, device, _, _ = self.buffers[self.input_name]
        np.copyto(host, tensor.ravel().astype(host.dtype))
        self._cuda.memcpy_htod_async(device, host, self.stream)
        self.context.execute_async_v3(stream_handle=self.stream.handle)
        out_host, out_device, out_shape, _ = self.buffers[self.output_name]
        self._cuda.memcpy_dtoh_async(out_host, out_device, self.stream)
        self.stream.synchronize()
        return decode_pose(out_host.reshape(out_shape), lb, self.conf, self.iou)

    def close(self):
        self._ctx.pop()
        self._ctx.detach()
