"""Robot-native YOLO person + facial-expression greeter.

This app reads the left eye from ``camera.head.rgb``, runs the repository's
YOLO11 pose model as a person detector, classifies the largest visible face,
and after a sustained distress cue starts a short spoken check-in (see
``check_in.py``). Vision stays local; only the check-in reply uses the
OpenRouter LLM, and the recorded ``sad_prompt.wav`` is used when speech or
the network is unavailable.

On the robot:
  python3 main.py --models-dir models
  python3 main.py --models-dir models --speak-on-start --max-frames 1
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
import uuid
import wave
from typing import Any

import cv2
import numpy as np

from ground_safety import (
    GroundAlertTracker,
    GroundAssessment,
    Keypoint,
    assess_ground_pose,
    keypoints_in_base_frame,
)


SCRIPT_DIR = Path(__file__).resolve().parent
VISION_SESSION_ID = uuid.uuid4().hex
PROJECT_ROOT = SCRIPT_DIR.parents[1]
# The speech helpers live in bbapps/greeter and are imported rather than
# duplicated, the same way check_in.py reaches them.
_GREETER_DIR = SCRIPT_DIR.parent / "greeter"
if str(_GREETER_DIR) not in sys.path:
    sys.path.append(str(_GREETER_DIR))
import speech_relay  # noqa: E402
EMOTION_LABELS = (
    "anger",
    "contempt",
    "disgust",
    "fear",
    "happiness",
    "neutral",
    "sadness",
    "surprise",
)
DISTRESS_LABELS = ("sadness", "anger", "disgust", "fear")
EXPRESSION_MODELS = ("enet_b0_8_best_afew.onnx", "enet_b0_8_va_mtl.onnx")


@dataclass(frozen=True)
class Detection:
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    keypoints: tuple[Keypoint, ...] = ()


@dataclass(frozen=True)
class Expression:
    x1: int
    y1: int
    x2: int
    y2: int
    label: str
    confidence: float
    # Summed sadness+anger+disgust+fear. The single "sadness" class is a poor
    # stand-in for "this person looks upset": AffectNet scores a plain frown as
    # disgust=60%/sadness=20%, so a sadness-only gate never fires on the cue
    # people actually give the robot. The four negative classes together
    # separate cleanly (87% on a frown, 19-49% otherwise).
    distress: float = 0.0


@dataclass(frozen=True)
class TrackedDetection:
    track_id: int
    detection: Detection


def intersection_over_union(left: Detection, right: Detection) -> float:
    x1 = max(left.x1, right.x1)
    y1 = max(left.y1, right.y1)
    x2 = min(left.x2, right.x2)
    y2 = min(left.y2, right.y2)
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    left_area = max(0, left.x2 - left.x1) * max(0, left.y2 - left.y1)
    right_area = max(0, right.x2 - right.x1) * max(0, right.y2 - right.y1)
    union = left_area + right_area - intersection
    return intersection / union if union else 0.0


def tracking_similarity(left: Detection, right: Detection) -> float:
    """Blend box overlap with center proximity for low-frame-rate tracking."""

    left_center = ((left.x1 + left.x2) / 2, (left.y1 + left.y2) / 2)
    right_center = ((right.x1 + right.x2) / 2, (right.y1 + right.y2) / 2)
    distance = (
        (left_center[0] - right_center[0]) ** 2
        + (left_center[1] - right_center[1]) ** 2
    ) ** 0.5
    scale = max(
        left.x2 - left.x1,
        left.y2 - left.y1,
        right.x2 - right.x1,
        right.y2 - right.y1,
        1,
    )
    proximity = max(0.0, 1.0 - distance / (2.0 * scale))
    return max(intersection_over_union(left, right), proximity * 0.5)


class PersonTracker:
    """Assign short-lived, session-local IDs to overlapping person boxes."""

    def __init__(self, minimum_iou: float = 0.15, max_missed: int = 2) -> None:
        self.minimum_iou = minimum_iou
        self.max_missed = max_missed
        self.next_id = 1
        self.tracks: dict[int, tuple[Detection, int]] = {}

    def update(self, detections: list[Detection]) -> list[TrackedDetection]:
        candidates = sorted(
            (
                (tracking_similarity(previous, detection), track_id, index)
                for track_id, (previous, _) in self.tracks.items()
                for index, detection in enumerate(detections)
            ),
            reverse=True,
        )
        assignments: dict[int, int] = {}
        used_tracks: set[int] = set()
        used_detections: set[int] = set()
        for score, track_id, index in candidates:
            if score < self.minimum_iou:
                break
            if track_id in used_tracks or index in used_detections:
                continue
            assignments[index] = track_id
            used_tracks.add(track_id)
            used_detections.add(index)

        updated: dict[int, tuple[Detection, int]] = {}
        tracked = []
        for index, detection in enumerate(detections):
            track_id = assignments.get(index)
            if track_id is None:
                track_id = self.next_id
                self.next_id += 1
            updated[track_id] = (detection, 0)
            tracked.append(TrackedDetection(track_id, detection))

        for track_id, (detection, missed) in self.tracks.items():
            if track_id not in used_tracks and track_id not in updated:
                missed += 1
                if missed <= self.max_missed:
                    updated[track_id] = (detection, missed)
        self.tracks = updated
        return tracked


SAD_LABELS = frozenset({"sad", "sadness"})
# Classes that may lead a reading that starts a check-in.
TRIGGER_LABELS = SAD_LABELS | frozenset(DISTRESS_LABELS)


def leans_sad(expression: Expression | None) -> bool:
    """True when sadness is the leading reading, at any confidence."""
    return expression is not None and expression.label in SAD_LABELS


@dataclass
class SadVoiceTrigger:
    """Decide when a visible frown is worth speaking about.

    A borderline cue still has to persist for ``hold_seconds`` so a passing
    grimace does not start a conversation, but an unmistakable one
    (``instant_confidence`` or higher) fires on the first reading: waiting a
    further second and a half on a face the models are already sure about is
    the difference between the robot feeling attentive and feeling laggy.

    The summed distress score alone is not enough: a resting face often reads
    as a diffuse spread ("neutral 25%", distress 55-60%), which used to start
    check-ins on people who were not frowning at all. So a negative emotion
    also has to be the leading class, and the bar sits above that noise band
    (a held frown scores 87-100%).
    """

    hold_seconds: float = 1.0
    cooldown_seconds: float = 30.0
    reset_seconds: float = 1.5
    confidence: float = 0.72
    instant_confidence: float = 0.9
    first_sad_at: float | None = None
    first_clear_at: float | None = None
    last_triggered_at: float | None = None
    armed: bool = True

    @property
    def warming(self) -> bool:
        """True while sad evidence is building toward a trigger."""
        return self.armed and self.first_sad_at is not None

    def update(
        self,
        expression: Expression | None,
        person_present: bool,
        now: float,
    ) -> bool:
        sad_visible = (
            person_present
            and expression is not None
            and expression.distress >= self.confidence
            and expression.label in TRIGGER_LABELS
        )
        if sad_visible:
            self.first_clear_at = None
            if self.first_sad_at is None:
                self.first_sad_at = now
            cooldown_over = (
                self.last_triggered_at is None
                or now - self.last_triggered_at >= self.cooldown_seconds
            )
            held = now - self.first_sad_at >= self.hold_seconds
            certain = expression.distress >= self.instant_confidence
            if self.armed and cooldown_over and (held or certain):
                self.armed = False
                self.last_triggered_at = now
                return True
            return False

        self.first_sad_at = None
        if not self.armed:
            if self.first_clear_at is None:
                self.first_clear_at = now
            elif now - self.first_clear_at >= self.reset_seconds:
                self.armed = True
                self.first_clear_at = None
        return False


def _letterbox(frame: np.ndarray, size: int) -> tuple[np.ndarray, float, int, int]:
    height, width = frame.shape[:2]
    scale = min(size / width, size / height)
    resized_width = round(width * scale)
    resized_height = round(height * scale)
    resized = cv2.resize(frame, (resized_width, resized_height))
    left = (size - resized_width) // 2
    top = (size - resized_height) // 2
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    canvas[top : top + resized_height, left : left + resized_width] = resized
    return canvas, scale, left, top


def yolo_detections(
    output: Any,
    scale: float,
    pad_x: int,
    pad_y: int,
    frame_width: int,
    frame_height: int,
    confidence: float,
    nms_threshold: float,
) -> list[Detection]:
    """Decode the one-class YOLO11 pose ONNX output into person boxes."""

    predictions = np.asarray(output)
    if predictions.ndim == 3 and predictions.shape[0] == 1:
        predictions = predictions[0]
    else:
        predictions = predictions.squeeze()
    if predictions.ndim == 1 and predictions.shape[0] == 56:
        predictions = predictions.reshape(56, 1)
    if predictions.ndim != 2:
        raise RuntimeError(f"Unexpected YOLO output shape: {np.asarray(output).shape}")
    if predictions.shape[0] == 56:
        predictions = predictions.T
    if predictions.shape[1] != 56:
        raise RuntimeError(f"Unexpected YOLO output shape: {np.asarray(output).shape}")

    boxes: list[list[int]] = []
    scores: list[float] = []
    rows: list[np.ndarray] = []
    for row in predictions:
        score = float(row[4])
        if score < confidence:
            continue
        center_x, center_y, width, height = (float(value) for value in row[:4])
        boxes.append(
            [
                round(center_x - width / 2),
                round(center_y - height / 2),
                round(width),
                round(height),
            ]
        )
        scores.append(score)
        rows.append(row)

    if not boxes:
        return []
    kept = cv2.dnn.NMSBoxes(boxes, scores, confidence, nms_threshold)
    detections = []
    for index in np.asarray(kept).reshape(-1):
        x, y, width, height = boxes[int(index)]
        x1 = round((x - pad_x) / scale)
        y1 = round((y - pad_y) / scale)
        x2 = round((x + width - pad_x) / scale)
        y2 = round((y + height - pad_y) / scale)
        pose = rows[int(index)][5:].reshape(17, 3)
        keypoints = tuple(
            Keypoint(
                keypoint_index,
                (float(point[0]) - pad_x) / scale,
                (float(point[1]) - pad_y) / scale,
                float(point[2]),
            )
            for keypoint_index, point in enumerate(pose)
            if float(point[2]) > 0.0
        )
        detections.append(
            Detection(
                max(0, min(frame_width, x1)),
                max(0, min(frame_height, y1)),
                max(0, min(frame_width, x2)),
                max(0, min(frame_height, y2)),
                scores[int(index)],
                keypoints,
            )
        )
    return detections


class PersonDetector:
    def __init__(self, model_path: Path, size: int = 320) -> None:
        if not model_path.exists():
            raise FileNotFoundError(f"YOLO ONNX model not found: {model_path}")
        self.net = cv2.dnn.readNetFromONNX(str(model_path))
        self.size = size

    def detect(
        self,
        frame: np.ndarray,
        confidence: float = 0.4,
        nms_threshold: float = 0.45,
    ) -> list[Detection]:
        canvas, scale, pad_x, pad_y = _letterbox(frame, self.size)
        blob = cv2.dnn.blobFromImage(
            canvas,
            scalefactor=1 / 255.0,
            size=(self.size, self.size),
            swapRB=True,
            crop=False,
        )
        self.net.setInput(blob)
        output = self.net.forward()
        height, width = frame.shape[:2]
        return yolo_detections(
            output,
            scale,
            pad_x,
            pad_y,
            width,
            height,
            confidence,
            nms_threshold,
        )


def square_face_box(
    x: float,
    y: float,
    face_width: float,
    face_height: float,
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    """Square, unpadded crop around a YuNet box.

    On RAF-DB this framing beat the previous 12%-padded rectangle by 3-5
    points of balanced accuracy: the classifiers were trained on square
    tight faces, and resizing a tall rectangle to 224x224 squashes features.
    """
    side = max(face_width, face_height)
    center_x = x + face_width / 2
    center_y = y + face_height / 2
    x1 = max(0, round(center_x - side / 2))
    y1 = max(0, round(center_y - side / 2))
    x2 = min(width, round(center_x + side / 2))
    y2 = min(height, round(center_y + side / 2))
    return x1, y1, x2, y2


def expression_probabilities(logits: np.ndarray) -> np.ndarray:
    """Softmax over the eight AffectNet classes with contempt removed.

    Multi-task models append valence/arousal after the eight logits. Contempt
    is rare, hard to read from a robot's camera, and mostly steals mass from
    neutral, so it is dropped before the argmax.
    """
    logits = np.asarray(logits, dtype=np.float64).reshape(-1)[: len(EMOTION_LABELS)]
    probabilities = np.exp(logits - logits.max())
    probabilities[EMOTION_LABELS.index("contempt")] = 0.0
    return probabilities / probabilities.sum()


def expression_runner(path: Path) -> tuple[Any, str]:
    """Return a callable running one expression ONNX, and the backend name.

    OpenCV 4.8's DNN module silently miscomputes these EfficientNet-B0 graphs
    on the robot: every input, including a photo of the floor and uniform
    noise, comes back as the same near-uniform distribution whose peak never
    exceeds ~25%. The 60% sadness gate is then unreachable by construction, so
    the greeter can never speak. onnxruntime runs the identical file correctly
    (the frowning reference face goes from 22% surprise to 53% disgust), so it
    is used whenever it imports and cv2.dnn is kept only as a last resort.
    """

    try:
        import onnxruntime
    except ImportError:
        net = cv2.dnn.readNetFromONNX(str(path))

        def run_cv2(blob: np.ndarray) -> np.ndarray:
            net.setInput(blob)
            return net.forward()

        return run_cv2, "cv2.dnn"

    options = onnxruntime.SessionOptions()
    # Two threads measured fastest on the Jetson's six cores (103 ms versus
    # 120 ms at one and 197 ms at six); more threads contend with YOLO.
    options.intra_op_num_threads = 2
    options.graph_optimization_level = (
        onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    )
    session = onnxruntime.InferenceSession(
        str(path), sess_options=options, providers=["CPUExecutionProvider"]
    )
    input_name = session.get_inputs()[0].name

    def run_ort(blob: np.ndarray) -> np.ndarray:
        return session.run(None, {input_name: blob})[0]

    return run_ort, "onnxruntime"


class ExpressionAnalyzer:
    def __init__(
        self,
        face_model: Path,
        expression_models: Path | list[Path],
        smoothing: float = 0.25,
        attack: float = 0.55,
        face_confidence: float = 0.75,
        min_face_size: int = 32,
    ) -> None:
        if isinstance(expression_models, Path):
            expression_models = [expression_models]
        for path in (face_model, *expression_models):
            if not path.exists():
                raise FileNotFoundError(f"Vision model not found: {path}")
        self.face_detector = cv2.FaceDetectorYN.create(
            str(face_model), "", (320, 320), face_confidence, 0.3, 50
        )
        # Averaging two EfficientNet-B0 heads costs one extra ~100 ms pass per
        # classified face and lifts RAF-DB balanced accuracy from 53% to 59%
        # while keeping the sadness precision of the original model.
        runners = [expression_runner(path) for path in expression_models]
        self.expression_nets = [runner for runner, _ in runners]
        self.backends = sorted({backend for _, backend in runners})
        if "cv2.dnn" in self.backends:
            print(
                "[expression] WARNING: onnxruntime is missing, so expression "
                "cues fall back to cv2.dnn. On OpenCV 4.8 that returns the "
                "same near-uniform scores for every input and the greeter will "
                "never reach its sadness threshold. Install onnxruntime.",
                flush=True,
            )
        self.smoothing = smoothing
        # A symmetric filter is the single biggest source of lag in the frown
        # path: at 0.25 it takes four classifications to cross 0.6 from a cold
        # start. Rising evidence is followed quickly and falling evidence
        # slowly, so a frown registers in a reading or two while one noisy
        # frame cannot cancel it.
        self.attack = max(smoothing, attack)
        self.min_face_size = min_face_size
        self.scores: np.ndarray | None = None
        self.missing_frames = 0

    def classify(self, face_bgr: np.ndarray) -> np.ndarray:
        face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(face_rgb, (224, 224)).astype(np.float32) / 255.0
        resized -= np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
        resized /= np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
        blob = np.ascontiguousarray(
            resized.transpose(2, 0, 1)[None, ...], dtype=np.float32
        )
        probabilities = [
            expression_probabilities(run(blob)) for run in self.expression_nets
        ]
        return np.mean(probabilities, axis=0)

    def missing(self) -> None:
        self.missing_frames += 1
        if self.missing_frames > 15:
            self.scores = None

    def analyze(
        self,
        frame: np.ndarray,
        classify: bool = True,
        offset_x: int = 0,
        offset_y: int = 0,
    ) -> Expression | None:
        height, width = frame.shape[:2]
        self.face_detector.setInputSize((width, height))
        _, faces = self.face_detector.detect(frame)
        if faces is None or len(faces) == 0:
            self.missing()
            return None

        # YuNet occasionally returns a box with NaN coordinates. NaN loses
        # every comparison, so such a row slips past the min-face-size check
        # below and then raises "cannot convert float NaN to integer" in
        # square_face_box, which killed the whole app mid-run. Drop those rows
        # before choosing the largest face, so one bad box costs one frame.
        usable = [
            row
            for row in faces
            if np.isfinite(np.asarray(row[:4], dtype=np.float64)).all()
        ]
        if not usable:
            self.missing()
            return None

        self.missing_frames = 0
        face = max(usable, key=lambda row: float(row[2] * row[3]))
        x, y, face_width, face_height = (float(value) for value in face[:4])
        if max(face_width, face_height) < self.min_face_size:
            # Upscaling a tiny face to 224 px mostly produces noise (RAF-DB
            # accuracy falls from 57% at 64 px to 44% at 24 px).
            self.missing()
            return None
        x1, y1, x2, y2 = square_face_box(
            x, y, face_width, face_height, width, height
        )

        if classify or self.scores is None:
            probabilities = self.classify(frame[y1:y2, x1:x2])
            if self.scores is None:
                self.scores = probabilities
            else:
                rate = np.where(probabilities > self.scores, self.attack, self.smoothing)
                blended = (1 - rate) * self.scores + rate * probabilities
                # Per-class rates break the unit sum; renormalise so the
                # reported confidence stays a probability.
                self.scores = blended / blended.sum()

        best_index = int(np.argmax(self.scores))
        distress = float(
            sum(self.scores[EMOTION_LABELS.index(name)] for name in DISTRESS_LABELS)
        )
        return Expression(
            x1 + offset_x,
            y1 + offset_y,
            x2 + offset_x,
            y2 + offset_y,
            EMOTION_LABELS[best_index],
            float(self.scores[best_index]),
            distress,
        )


def face_belongs_to_person(
    expression: Expression | None,
    detections: list[Detection],
) -> bool:
    if expression is None:
        return False
    center_x = (expression.x1 + expression.x2) / 2
    center_y = (expression.y1 + expression.y2) / 2
    return any(
        detection.x1 <= center_x <= detection.x2
        and detection.y1 <= center_y <= detection.y2
        for detection in detections
    )


class DashboardState:
    def __init__(self) -> None:
        self.condition = threading.Condition()
        self.sequence = 0
        self.jpeg: bytes | None = None
        # Open /stream.mjpg connections; frames are only JPEG-encoded while
        # somebody is watching.
        self.viewers = 0
        self.targets: tuple[Detection, ...] = ()
        self.targets_at = 0.0
        self.frame_width = 0
        self.frame_height = 0
        self.metrics: dict[str, Any] = {
            "ready": False,
            "frame": 0,
            "people": 0,
            "track_ids": [],
            "ground_status": "starting",
            "ground_alerts": [],
            "ground_observations": [],
            "depth_aligned": False,
            "map": {"ready": False},
            "expression": None,
            "expression_confidence": 0.0,
            "check_in": "off",
            "pipeline_ms": 0.0,
            "yolo_ms": 0.0,
            "expression_ms": 0.0,
            "camera_age_ms": 0.0,
            "scan_fps": 0.0,
        }

    def add_viewer(self, delta: int) -> None:
        with self.condition:
            self.viewers += delta

    def publish(self, frame: np.ndarray, metrics: dict[str, Any]) -> None:
        with self.condition:
            watched = self.viewers > 0
        jpeg = None
        if watched:
            ok, encoded = cv2.imencode(
                ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 82]
            )
            if not ok:
                return
            jpeg = encoded.tobytes()
        with self.condition:
            self.jpeg = jpeg
            self.metrics = {"ready": True, **metrics}
            self.sequence += 1
            self.condition.notify_all()

    def wait_for_frame(
        self,
        previous_sequence: int,
        timeout: float = 10.0,
    ) -> tuple[int, bytes | None]:
        with self.condition:
            self.condition.wait_for(
                lambda: self.sequence != previous_sequence,
                timeout=timeout,
            )
            return self.sequence, self.jpeg

    def status(self) -> dict[str, Any]:
        with self.condition:
            return dict(self.metrics)

    def update_targets(
        self,
        detections: list[Detection],
        frame_width: int,
        frame_height: int,
        observed_at: float,
    ) -> None:
        with self.condition:
            self.targets = tuple(detections)
            self.targets_at = observed_at
            self.frame_width = frame_width
            self.frame_height = frame_height

    def select_target(self, preference: str, max_age: float = 1.0):
        with self.condition:
            if time.monotonic() - self.targets_at > max_age or not self.targets:
                return None
            targets = self.targets
            width = self.frame_width
            height = self.frame_height
        if preference == "left":
            target = min(targets, key=lambda item: item.x1 + item.x2)
        elif preference == "right":
            target = max(targets, key=lambda item: item.x1 + item.x2)
        else:
            target = max(
                targets,
                key=lambda item: (
                    (item.x2 - item.x1) * (item.y2 - item.y1),
                    item.confidence,
                ),
            )
        return (
            2.0 * (target.x1 + target.x2) / (2.0 * width) - 1.0,
            2.0 * (target.y1 + target.y2) / (2.0 * height) - 1.0,
        )


class CameraActionController:
    """Run at most one camera-selected motion in an isolated robot process."""

    def __init__(self, runner=Path("/tmp/camera_point_motion.py")) -> None:
        self.runner = runner
        self.lock = threading.Lock()
        self.process: subprocess.Popen | None = None
        self.last_error: str | None = None
        self.stage = "idle"
        self.logs: list[str] = []

    def _log(self, line: str) -> None:
        line = line.strip()
        if not line:
            return
        with self.lock:
            self.logs.append(line)
            del self.logs[:-80]
        print(line, flush=True)

    def status(self) -> dict[str, Any]:
        with self.lock:
            return {
                "movement_running": self.process is not None,
                "movement_error": self.last_error,
                "movement_stage": self.stage,
                "movement_log": list(self.logs),
            }

    def running(self) -> bool:
        with self.lock:
            return self.process is not None

    def error(self) -> str | None:
        self.running()
        with self.lock:
            return self.last_error

    def start(self, target: tuple[float, float]) -> tuple[bool, str]:
        with self.lock:
            if self.process is not None and self.process.poll() is None:
                return False, "Another camera action is already running"
            if not self.runner.is_file():
                return False, "Pointing runner is not installed; reconnect the gesture dashboard"
            x_offset, y_offset = target
            self.last_error = None
            self.stage = "launching"
            self.logs = []
            self.logs.append(
                f"[vision-action] selected target x={x_offset:+.3f} y={y_offset:+.3f}"
            )
            self.process = subprocess.Popen(
                [
                    sys.executable,
                    str(self.runner),
                    "--x-offset", f"{x_offset:.6f}",
                    "--y-offset", f"{y_offset:.6f}",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            process = self.process
        threading.Thread(
            target=self._monitor,
            args=(process,),
            name="camera-action-monitor",
            daemon=True,
        ).start()
        return True, "Started camera-guided pointing"

    def _monitor(self, process: subprocess.Popen) -> None:
        lines = []
        assert process.stdout is not None
        for line in process.stdout:
            line = line.strip()
            if line:
                lines.append(line)
                self._log(line)
                if "stage=" in line:
                    with self.lock:
                        self.stage = line.split("stage=", 1)[1].split()[0]
        return_code = process.wait()
        with self.lock:
            if self.process is process:
                if return_code:
                    detail = lines[-1] if lines else f"exit status {return_code}"
                    self.last_error = detail
                    self.stage = "failed"
                else:
                    self.stage = "complete"
                self.process = None

    def stop(self) -> tuple[bool, str]:
        with self.lock:
            process = self.process
            if process is None or process.poll() is not None:
                return False, "No camera action is running"
            process.send_signal(signal.SIGINT)
        return True, "Stop requested; returning the arm safely"


DASHBOARD_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>BracketBot Vision</title>
  <style>
    :root { color-scheme: dark; font-family: ui-sans-serif, system-ui, sans-serif; }
    body { margin: 0; background: #090d12; color: #edf4fa; }
    main { width: min(1200px, 96vw); margin: 24px auto; }
    h1 { margin: 0 0 4px; font-size: clamp(24px, 4vw, 40px); }
    .note { color: #9fb0bf; margin: 0 0 18px; }
    .stats { display: grid; grid-template-columns: repeat(auto-fit,minmax(145px,1fr)); gap: 10px; margin-bottom: 14px; }
    .card { background: #131c25; border: 1px solid #263746; border-radius: 12px; padding: 12px 14px; }
    .label { color: #8fa4b5; font-size: 12px; text-transform: uppercase; letter-spacing: .08em; }
    .value { margin-top: 4px; font-size: 24px; font-variant-numeric: tabular-nums; }
    .feed { display: block; width: 100%; background: #000; border: 1px solid #263746; border-radius: 14px; }
    .ok { color: #63e6a5; }
    .warn { color: #ffca58; }
    .alert { color: #ff6262; }
  </style>
</head>
<body><main>
  <h1>Robot vision</h1>
  <p class="note">Live left-eye view. Person numbers are temporary tracking IDs, not face recognition.</p>
  <section class="stats">
    <div class="card"><div class="label">Status</div><div class="value ok" id="status">Starting</div></div>
    <div class="card"><div class="label">People</div><div class="value" id="people">0</div></div>
    <div class="card"><div class="label">Ground safety</div><div class="value" id="ground">Starting</div></div>
    <div class="card"><div class="label">SLAM map</div><div class="value" id="map">Starting</div></div>
    <div class="card"><div class="label">Expression cue</div><div class="value" id="expression">—</div></div>
    <div class="card"><div class="label">Sad check-in</div><div class="value" id="checkin">—</div></div>
    <div class="card"><div class="label">Pipeline</div><div class="value" id="pipeline">—</div></div>
    <div class="card"><div class="label">YOLO</div><div class="value" id="yolo">—</div></div>
    <div class="card"><div class="label">Face + expression</div><div class="value" id="face">—</div></div>
    <div class="card"><div class="label">Camera age</div><div class="value" id="age">—</div></div>
    <div class="card"><div class="label">Scan rate</div><div class="value" id="fps">—</div></div>
  </section>
  <img class="feed" src="/stream.mjpg" alt="Annotated robot head-camera stream">
</main>
<script>
// An MJPEG <img> never reconnects on its own: after the app restarts it keeps
// showing its last frame while the stats below stay live.
const feed = document.querySelector('.feed');
let lastFrame = 0, wasDown = false;
function reconnectFeed() { feed.src = `/stream.mjpg?t=${Date.now()}`; }
feed.onerror = () => setTimeout(reconnectFeed, 1000);
async function refresh() {
  try {
    const response = await fetch('/api/status', {cache: 'no-store'});
    const s = await response.json();
    if (wasDown || s.frame < lastFrame) reconnectFeed();
    wasDown = false;
    lastFrame = s.frame;
    document.getElementById('status').textContent = s.ready ? `Live · frame ${s.frame}` : 'Starting';
    document.getElementById('people').textContent = s.track_ids?.length ? `${s.people} · #${s.track_ids.join(', #')}` : s.people;
    const ground = document.getElementById('ground');
    ground.textContent = s.ground_status === 'alert' ? `STOP · possible person on ground (#${s.ground_alerts.map(a => a.track_id).join(', #')})` :
      s.ground_status === 'checking' ? 'Checking low pose…' : s.depth_aligned ? 'Clear' : 'Depth unavailable';
    ground.className = `value ${s.ground_status === 'alert' ? 'alert' : s.ground_status === 'checking' ? 'warn' : 'ok'}`;
    const map = s.map || {};
    document.getElementById('map').textContent = map.ready ? `${map.known_area_m2.toFixed(1)} m² known` : 'Unavailable';
    document.getElementById('expression').textContent = s.expression ? `${s.expression} ${Math.round(s.expression_confidence * 100)}%` : 'none';
    document.getElementById('checkin').textContent = s.check_in || 'off';
    document.getElementById('pipeline').textContent = `${Math.round(s.pipeline_ms)} ms`;
    document.getElementById('yolo').textContent = `${Math.round(s.yolo_ms)} ms`;
    document.getElementById('face').textContent = `${Math.round(s.expression_ms)} ms`;
    document.getElementById('age').textContent = `${Math.round(s.camera_age_ms)} ms`;
    document.getElementById('fps').textContent = `${s.scan_fps.toFixed(2)} FPS`;
  } catch (_) { wasDown = true; document.getElementById('status').textContent = 'Disconnected'; }
}
setInterval(refresh, 750); refresh();
</script></body></html>"""


def dashboard_handler(
    state: DashboardState,
    actions: CameraActionController,
) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path == "/":
                self._send(DASHBOARD_HTML.encode(), "text/html; charset=utf-8")
            elif self.path == "/api/status":
                status = state.status()
                status.update(actions.status())
                target = state.select_target("primary")
                status["point_target"] = (
                    None
                    if target is None
                    else {"x_offset": target[0], "y_offset": target[1]}
                )
                self._send(
                    json.dumps(status).encode(),
                    "application/json",
                )
            elif self.path == "/healthz":
                self._send(b"ok\n", "text/plain; charset=utf-8")
            elif self.path.split("?", 1)[0] == "/stream.mjpg":
                self._stream()
            else:
                self.send_error(404)

        def do_POST(self) -> None:
            if self.client_address[0] not in {"127.0.0.1", "::1"}:
                self._json(403, False, "Camera actions are robot-local only")
                return
            if self.path == "/api/action/stop":
                ok, message = actions.stop()
                self._json(200 if ok else 409, ok, message)
                return
            if self.path != "/api/action":
                self.send_error(404)
                return
            try:
                length = min(int(self.headers.get("Content-Length", "0")), 4096)
                body = json.loads(self.rfile.read(length) or b"{}")
            except (ValueError, json.JSONDecodeError):
                self._json(400, False, "Invalid JSON body")
                return
            action = str(body.get("action", ""))
            preferences = {
                "point": "primary",
                "point-left": "left",
                "point-right": "right",
            }
            if action not in preferences:
                self._json(404, False, "Unknown camera action")
                return
            target = state.select_target(preferences[action])
            if target is None:
                self._json(409, False, "No fresh person detection is available")
                return
            ok, message = actions.start(target)
            self._json(202 if ok else 409, ok, message)

        def _json(self, status: int, ok: bool, message: str) -> None:
            body = json.dumps({"ok": ok, "message": message}).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _send(self, body: bytes, content_type: str) -> None:
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _stream(self) -> None:
            self.send_response(200)
            self.send_header(
                "Content-Type", "multipart/x-mixed-replace; boundary=frame"
            )
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            sequence = -1
            state.add_viewer(1)
            try:
                while True:
                    sequence, jpeg = state.wait_for_frame(sequence)
                    if jpeg is None:
                        continue
                    self.wfile.write(
                        b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                        + str(len(jpeg)).encode()
                        + b"\r\n\r\n"
                        + jpeg
                        + b"\r\n"
                    )
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                state.add_viewer(-1)

        def log_message(self, format: str, *args: Any) -> None:
            return

    return Handler


def start_dashboard(
    state: DashboardState,
    actions: CameraActionController,
    host: str,
    port: int,
) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer((host, port), dashboard_handler(state, actions))
    server.daemon_threads = True
    threading.Thread(
        target=server.serve_forever,
        name="vision-dashboard",
        daemon=True,
    ).start()
    print(f"[dashboard] http://{host}:{port}", flush=True)
    return server


def camera_age_ms(timestamp: Any) -> float:
    try:
        captured_ns = int(np.datetime64(timestamp, "ns").astype(np.int64))
        return max(0.0, (time.time_ns() - captured_ns) / 1_000_000)
    except (TypeError, ValueError, OverflowError):
        return 0.0


def timestamp_ns(timestamp: Any) -> int:
    try:
        return int(np.datetime64(timestamp, "ns").astype(np.int64))
    except (TypeError, ValueError, OverflowError):
        return 0


def planar_yaw(quaternion: Any) -> float:
    quaternion = np.asarray(quaternion, dtype=np.float64).reshape(-1)
    if len(quaternion) < 4:
        raise ValueError("SLAM quaternion must contain XYZW")
    return 2.0 * np.arctan2(quaternion[2], quaternion[3])


def publish_ground_safety_file(
    path: Path,
    status: str,
    alerts: list[dict[str, Any]],
    *,
    camera_timestamp_ns: int,
    map_epoch: int | None,
    observations: list[dict[str, Any]] | None = None,
    depth_aligned: bool = False,
) -> None:
    """Atomically publish a small interlock state for the navigation owner."""

    payload = {
        "schema_version": 1,
        "approach_schema_version": 1,
        "status": status,
        "possible_person_on_ground": status == "alert",
        "alerts": alerts,
        "camera_timestamp_ns": camera_timestamp_ns,
        "map_epoch": map_epoch,
        "published_at": time.time(),
        "session_id": VISION_SESSION_ID,
        "depth_aligned": depth_aligned,
        "observations": observations or [],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, separators=(",", ":")) + "\n")
    temporary.replace(path)


def draw_overlay(
    frame: np.ndarray,
    detections: list[TrackedDetection],
    ground_assessments: dict[int, GroundAssessment],
    ground_statuses: dict[int, str],
    expression: Expression | None,
    pipeline_ms: float,
    end_to_end_ms: float,
    scan_fps: float,
) -> None:
    for tracked in detections:
        detection = tracked.detection
        ground_status = ground_statuses.get(tracked.track_id, "unknown")
        color = (
            (40, 40, 255)
            if ground_status == "alert"
            else (0, 190, 255)
            if ground_status == "checking"
            else (0, 220, 80)
        )
        cv2.rectangle(
            frame,
            (detection.x1, detection.y1),
            (detection.x2, detection.y2),
            color,
            2,
        )
        cv2.putText(
            frame,
            (
                f"STOP: possible person on ground #{tracked.track_id}"
                if ground_status == "alert"
                else f"Checking ground pose #{tracked.track_id}"
                if ground_status == "checking"
                else f"Person #{tracked.track_id} {detection.confidence:.0%}"
            ),
            (detection.x1, max(20, detection.y1 - 7)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )
        assessment = ground_assessments.get(tracked.track_id)
        if assessment is not None and assessment.map_position is not None:
            cv2.putText(
                frame,
                f"map ({assessment.map_position[0]:+.2f}, {assessment.map_position[1]:+.2f})",
                (detection.x1, min(frame.shape[0] - 8, detection.y2 + 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
                cv2.LINE_AA,
            )
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 42), (10, 15, 20), -1)
    cv2.putText(
        frame,
        f"pipeline {pipeline_ms:.0f} ms  camera-to-screen {end_to_end_ms:.0f} ms  scan {scan_fps:.2f} FPS",
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    if expression is not None:
        cv2.rectangle(
            frame,
            (expression.x1, expression.y1),
            (expression.x2, expression.y2),
            (255, 80, 220),
            2,
        )
        cv2.putText(
            frame,
            f"{expression.label} {expression.confidence:.0%}",
            (expression.x1, min(frame.shape[0] - 8, expression.y2 + 22)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 80, 220),
            2,
            cv2.LINE_AA,
        )


def _load_bbos() -> tuple[Any, Any, Any, Any]:
    """Bridge Jetson system OpenCV with the bot's existing BBOS install."""

    bbos_root = Path.home() / "bbos"
    venv_packages = bbos_root / ".venv/lib/python3.10/site-packages"
    if str(bbos_root) not in sys.path:
        sys.path.insert(0, str(bbos_root))
    if str(venv_packages) not in sys.path:
        # Append after system packages so OpenCV keeps its NumPy 1.x ABI.
        sys.path.append(str(venv_packages))
    from bbos import Config, Reader, Type, Writer

    return Config, Reader, Type, Writer


class RobotSpeaker:
    def __init__(self, wav_path: Path) -> None:
        if not wav_path.exists():
            raise FileNotFoundError(f"Voice prompt not found: {wav_path}")
        self.wav_path = wav_path
        self.lock = threading.Lock()

    def play(self) -> None:
        if not self.lock.acquire(blocking=False):
            return
        try:
            Config, _, Type, Writer = _load_bbos()
            config = Config("speaker")
            with wave.open(str(self.wav_path), "rb") as source:
                if (
                    source.getsampwidth() != 2
                    or source.getnchannels() != config.channels
                    or source.getframerate() != config.sample_rate
                ):
                    raise RuntimeError(
                        "Voice prompt must be uncompressed 16-bit PCM matching "
                        f"the robot speaker ({config.sample_rate} Hz, "
                        f"{config.channels} channel(s))"
                    )
                try:
                    speaker_writer = Writer("speaker.audio", Type("speaker_audio"))
                except RuntimeError as error:
                    # The always-on voice assistant owns the single
                    # speaker.audio writer; ask it to play this instead.
                    print(
                        f"[speaker] Speaker owned elsewhere ({error}); relaying",
                        flush=True,
                    )
                    if not speech_relay.request(wav=self.wav_path):
                        print(
                            "[speaker] Nothing served the speech relay; the "
                            "sadness prompt was not played",
                            flush=True,
                        )
                    return
                with speaker_writer as speaker:
                    time.sleep(0.25)
                    while raw := source.readframes(config.chunk_size):
                        samples = np.frombuffer(raw, dtype="<i2")
                        if len(samples) < config.chunk_size:
                            samples = np.pad(
                                samples, (0, config.chunk_size - len(samples))
                            )
                        with speaker.buf() as frame:
                            frame["audio"] = samples.reshape(-1, config.channels)
            print("[speaker] Finished sadness prompt", flush=True)
        finally:
            self.lock.release()

    def play_async(self) -> None:
        threading.Thread(target=self.play, name="sad-voice", daemon=True).start()


def model_path(models_dir: Path, filename: str) -> Path:
    candidates = (
        models_dir / filename,
        PROJECT_ROOT / "assets/models" / filename,
        PROJECT_ROOT / filename,
    )
    return next((path for path in candidates if path.exists()), candidates[0])


def start_check_in(args: argparse.Namespace):
    """Return the spoken sad check-in, or None to fall back to the WAV prompt."""
    if args.no_check_in:
        return None
    try:
        from check_in import build_check_in

        check_in = build_check_in(args, *_load_bbos())
    except Exception as exc:
        print(
            f"[check-in] Disabled ({type(exc).__name__}: {exc}); "
            "using the recorded prompt",
            flush=True,
        )
        return None
    # Render the openers now, off the hot path, so the cue-to-voice gap at
    # trigger time is just the speaker buffer rather than a TTS round trip.
    check_in.prewarm_async()
    print("[check-in] Ready: sadness cues start a spoken check-in", flush=True)
    return check_in


def run(args: argparse.Namespace) -> int:
    Config, Reader, _, _ = _load_bbos()
    detector = PersonDetector(model_path(args.models_dir, "yolo11n-pose.onnx"))
    expression_models = [
        model_path(args.models_dir, filename) for filename in EXPRESSION_MODELS
    ]
    if not all(path.exists() for path in expression_models[1:]):
        print(
            "[vision] enet_b0_8_va_mtl.onnx not found; using the single "
            "expression model (less accurate)",
            flush=True,
        )
        expression_models = expression_models[:1]
    analyzer = ExpressionAnalyzer(
        model_path(args.models_dir, "face_detection_yunet_2026may.onnx"),
        expression_models,
        smoothing=args.expression_smoothing,
        attack=args.expression_attack,
        face_confidence=args.face_confidence,
        min_face_size=args.min_face_size,
    )
    print(
        f"[expression] {len(expression_models)} model(s) on "
        f"{', '.join(analyzer.backends)}",
        flush=True,
    )
    speaker = RobotSpeaker(args.voice_prompt)
    check_in = start_check_in(args)
    trigger = SadVoiceTrigger(
        hold_seconds=args.sad_hold_seconds,
        cooldown_seconds=args.sad_cooldown,
        reset_seconds=args.sad_reset_seconds,
        confidence=args.sad_confidence,
        instant_confidence=args.sad_instant_confidence,
    )
    tracker = PersonTracker()
    ground_tracker = GroundAlertTracker(
        hold_seconds=args.ground_hold_seconds,
        clear_seconds=args.ground_clear_seconds,
    )
    dashboard = DashboardState()
    actions = CameraActionController()
    server = start_dashboard(dashboard, actions, args.dashboard_host, args.dashboard_port)

    if args.speak_on_start:
        print("[speaker] Playing startup speaker test", flush=True)
        speaker.play()

    try:
        map_resolution = float(Config("mapping").voxel_size_m)
    except Exception:
        map_resolution = 0.05
    map_metrics: dict[str, Any] = {"ready": False}
    last_map_scan = 0.0
    last_safety_publish = 0.0
    last_safety_status = None
    last_ground_assessments: dict[int, GroundAssessment] = {}

    print(
        "[vision] Reading camera.rect + timestamp-aligned camera.points; "
        "map pose from slam.pose",
        flush=True,
    )
    frame_count = 0
    last_frame = None
    last_log_at = 0.0
    sad_leaning = False
    previous_started = None
    try:
        with Reader("camera.rect", keeptime=False) as camera, \
             Reader("camera.points", keeptime=False, aligned_to=camera) as point_reader, \
             Reader("slam.pose", keeptime=False) as slam_reader, \
             Reader("mapping.grid2d", keeptime=False) as map_reader:
            while not camera.ready():
                time.sleep(0.02)
            while not args.max_frames or frame_count < args.max_frames:
                if not camera.ready():
                    time.sleep(0.005)
                    continue
                rect_rgb = camera.data["left"].copy()
                timestamp = camera.data["timestamp"].copy()
                frame = cv2.cvtColor(rect_rgb, cv2.COLOR_RGB2BGR)
                camera_timestamp_ns = timestamp_ns(timestamp)
                point_indices = None
                depth_points = None
                depth_aligned = False
                # Read the timestamp-aligned depth sample before inference. The
                # IPC history window may no longer contain this RGB frame after
                # the ~100-300 ms model pass.
                if point_reader.ready():
                    point_timestamp_ns = timestamp_ns(point_reader.data["timestamp"])
                    depth_aligned = (
                        camera_timestamp_ns > 0
                        and point_timestamp_ns == camera_timestamp_ns
                    )
                    if depth_aligned:
                        point_count = int(point_reader.data["num_points"])
                        point_indices = point_reader.data["idx_2d"][:point_count].copy()
                        depth_points = point_reader.data["points"][:point_count].copy()
                robot_position = None
                robot_yaw = None
                map_epoch = None
                if slam_reader.ready():
                    robot_position = slam_reader.data["pos"].copy()
                    robot_yaw = float(planar_yaw(slam_reader.data["quat"]))
                    map_epoch = int(slam_reader.data["pgo_count"])
                started = time.monotonic()
                scan_fps = (
                    0.0
                    if previous_started is None
                    else 1.0 / max(started - previous_started, 1e-9)
                )
                previous_started = started
                detections = detector.detect(frame, confidence=args.yolo_confidence)
                yolo_finished = time.monotonic()
                dashboard.update_targets(
                    detections,
                    frame.shape[1],
                    frame.shape[0],
                    yolo_finished,
                )
                tracked = tracker.update(detections)
                ground_assessments: dict[int, GroundAssessment] = {}
                if depth_aligned and point_indices is not None and depth_points is not None:
                    for item in tracked:
                        pose_3d = keypoints_in_base_frame(
                            item.detection.keypoints,
                            point_indices,
                            depth_points,
                            frame.shape[1],
                            frame.shape[0],
                            keypoint_confidence=args.pose_confidence,
                            search_radius_px=args.depth_search_radius,
                        )
                        assessment = assess_ground_pose(
                            pose_3d,
                            robot_position=robot_position,
                            robot_yaw=robot_yaw,
                        )
                        ground_assessments[item.track_id] = assessment
                        if assessment.state != "unknown":
                            last_ground_assessments[item.track_id] = assessment
                else:
                    ground_assessments = {
                        item.track_id: GroundAssessment(
                            "unknown", 0.0, "camera/depth timestamps are not aligned", 0
                        )
                        for item in tracked
                    }
                ground_statuses = ground_tracker.update(
                    ground_assessments, time.monotonic()
                )
                if any(status == "alert" for status in ground_statuses.values()):
                    ground_status = "alert"
                elif any(status == "checking" for status in ground_statuses.values()):
                    ground_status = "checking"
                else:
                    ground_status = "clear" if depth_aligned else "unavailable"
                ground_alerts = []
                for track_id, status in ground_statuses.items():
                    if status != "alert":
                        continue
                    assessment = ground_assessments.get(
                        track_id, last_ground_assessments.get(track_id)
                    )
                    alert = {"track_id": track_id}
                    if assessment is not None:
                        alert.update(
                            {
                                "confidence": assessment.confidence,
                                "reason": assessment.reason,
                                "torso_height_m": assessment.torso_height_m,
                                "map_position": assessment.map_position,
                            }
                        )
                    ground_alerts.append(alert)
                ground_observations = []
                for item in tracked:
                    assessment = ground_assessments.get(item.track_id)
                    if assessment is None:
                        continue
                    ground_observations.append(
                        {
                            "track_id": item.track_id,
                            "state": assessment.state,
                            "latch_status": ground_statuses.get(
                                item.track_id, "unknown"
                            ),
                            "confidence": assessment.confidence,
                            "reason": assessment.reason,
                            "depth_keypoints": assessment.depth_keypoints,
                            "torso_height_m": assessment.torso_height_m,
                            "body_extent_m": assessment.body_extent_m,
                            "body_radius_m": assessment.body_radius_m,
                            "base_position": assessment.base_position,
                            "map_position": assessment.map_position,
                        }
                    )

                now = time.monotonic()
                if now - last_map_scan >= 5.0 and map_reader.ready():
                    grid = map_reader.data["grid"]
                    floor_cells = int(np.count_nonzero(grid == 1))
                    obstacle_cells = int(np.count_nonzero(grid == 2))
                    map_metrics = {
                        "ready": True,
                        "known_area_m2": round(
                            (floor_cells + obstacle_cells) * map_resolution**2, 2
                        ),
                        "floor_area_m2": round(floor_cells * map_resolution**2, 2),
                        "obstacle_cells": obstacle_cells,
                        "robot_position": [
                            round(float(value), 3)
                            for value in map_reader.data["robot_pos"]
                        ],
                        "robot_heading": round(
                            float(map_reader.data["robot_heading"]), 4
                        ),
                        "map_epoch": map_epoch,
                    }
                    last_map_scan = now

                if (
                    ground_status != last_safety_status
                    or now - last_safety_publish >= 0.1
                ):
                    publish_ground_safety_file(
                        args.ground_alert_file,
                        ground_status,
                        ground_alerts,
                        camera_timestamp_ns=camera_timestamp_ns,
                        map_epoch=map_epoch,
                        observations=ground_observations,
                        depth_aligned=depth_aligned,
                    )
                    last_safety_status = ground_status
                    last_safety_publish = now
                expression = None
                if detections:
                    primary = max(
                        detections,
                        key=lambda item: (item.x2 - item.x1)
                        * (item.y2 - item.y1),
                    )
                    person_crop = frame[
                        primary.y1 : primary.y2,
                        primary.x1 : primary.x2,
                    ]
                    if person_crop.size:
                        # Skipping frames saves CPU while nothing is happening,
                        # but from the first sad-leaning reading onward every
                        # frame is classified, so the evidence needed to speak
                        # accumulates at the full scan rate instead of a third
                        # of it. This is what the interval used to cost.
                        expression = analyzer.analyze(
                            person_crop,
                            classify=(
                                trigger.warming
                                or sad_leaning
                                or frame_count % args.expression_interval == 0
                            ),
                            offset_x=primary.x1,
                            offset_y=primary.y1,
                        )
                    else:
                        analyzer.missing()
                else:
                    analyzer.missing()
                expression_finished = time.monotonic()
                # A frown usually leads with disgust, not sadness, so building
                # distress counts as sad-leaning too.
                sad_leaning = leans_sad(expression) or (
                    expression is not None
                    and expression.distress >= args.sad_confidence / 2
                )
                associated = face_belongs_to_person(expression, detections)
                if trigger.update(expression, associated, time.monotonic()):
                    cue = f"sadness cue ({expression.confidence:.0%})"
                    if check_in is None:
                        print(f"[emotion] {cue}; playing prompt", flush=True)
                        speaker.play_async()
                    elif check_in.start_async():
                        print(f"[emotion] {cue}; starting check-in", flush=True)

                elapsed = time.monotonic() - started
                age_ms = camera_age_ms(timestamp)
                draw_overlay(
                    frame,
                    tracked,
                    ground_assessments,
                    ground_statuses,
                    expression,
                    pipeline_ms=elapsed * 1000,
                    end_to_end_ms=age_ms,
                    scan_fps=scan_fps,
                )
                metrics = {
                    "frame": frame_count + 1,
                    "people": len(tracked),
                    "track_ids": [item.track_id for item in tracked],
                    "ground_status": ground_status,
                    "ground_alerts": ground_alerts,
                    "ground_observations": ground_observations,
                    "depth_aligned": depth_aligned,
                    "map": map_metrics,
                    "expression": None if expression is None else expression.label,
                    "expression_confidence": (
                        0.0 if expression is None else expression.confidence
                    ),
                    "distress": 0.0 if expression is None else expression.distress,
                    "pipeline_ms": elapsed * 1000,
                    "yolo_ms": (yolo_finished - started) * 1000,
                    "expression_ms": (expression_finished - yolo_finished) * 1000,
                    "camera_age_ms": age_ms,
                    "scan_fps": scan_fps,
                    "check_in": "off" if check_in is None else check_in.status,
                }
                dashboard.publish(frame, metrics)
                label = (
                    "none"
                    if expression is None
                    else f"{expression.label} {expression.confidence:.0%} "
                    f"distress={expression.distress:.0%}"
                )
                now = time.monotonic()
                if args.max_frames or now - last_log_at >= args.log_interval:
                    ids = ",".join(str(item.track_id) for item in tracked) or "none"
                    ground_summary = ";".join(
                        f"#{item['track_id']}:{item['state']}:{item['reason']}"
                        for item in ground_observations
                    ) or "none"
                    print(
                        f"[vision] frame={frame_count + 1} people={len(tracked)} "
                        f"ids={ids} expression={label} associated={associated} "
                        f"ground={ground_status} depth_aligned={depth_aligned} "
                        f"ground_observations={ground_summary} "
                        f"pipeline={elapsed:.3f}s "
                        f"yolo={(yolo_finished - started) * 1000:.0f}ms "
                        f"face={(expression_finished - yolo_finished) * 1000:.0f}ms "
                        f"age={age_ms:.0f}ms",
                        flush=True,
                    )
                    last_log_at = now
                frame_count += 1
                last_frame = frame
                time.sleep(max(0.0, args.scan_interval - elapsed))
    finally:
        actions.stop()
        server.shutdown()
        server.server_close()

    if args.snapshot is not None and last_frame is not None:
        args.snapshot.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(args.snapshot), last_frame):
            raise RuntimeError(f"Could not write snapshot: {args.snapshot}")
        print(f"[vision] Saved {args.snapshot}", flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run YOLO and visible-expression cues on the robot head camera"
    )
    parser.add_argument("--models-dir", type=Path, default=SCRIPT_DIR / "models")
    parser.add_argument(
        "--voice-prompt",
        type=Path,
        default=SCRIPT_DIR / "sad_prompt.wav",
    )
    parser.add_argument("--yolo-confidence", type=float, default=0.4)
    parser.add_argument("--face-confidence", type=float, default=0.75)
    parser.add_argument("--expression-smoothing", type=float, default=0.25)
    parser.add_argument(
        "--expression-attack",
        type=float,
        default=0.55,
        help="faster smoothing applied to rising expression evidence",
    )
    parser.add_argument("--expression-interval", type=int, default=3)
    parser.add_argument(
        "--min-face-size",
        type=int,
        default=32,
        help="skip expression cues for faces smaller than this many pixels",
    )
    parser.add_argument(
        "--pose-confidence",
        type=float,
        default=0.35,
        help="minimum YOLO confidence for a keypoint used with depth",
    )
    parser.add_argument(
        "--depth-search-radius",
        type=int,
        default=5,
        help="pixel radius used to associate sparse aligned depth to a pose joint",
    )
    parser.add_argument(
        "--ground-hold-seconds",
        type=float,
        default=2.0,
        help="sustained low 3D pose evidence required before raising an alert",
    )
    parser.add_argument(
        "--ground-clear-seconds",
        type=float,
        default=2.0,
        help="sustained clear 3D pose evidence required to release an alert",
    )
    parser.add_argument(
        "--ground-alert-file",
        type=Path,
        default=Path("/tmp/bracketbot_ground_alert.json"),
        help="atomic safety observation consumed by the navigation app",
    )
    parser.add_argument(
        "--sad-confidence",
        type=float,
        default=0.72,
        help="summed sadness+anger+disgust+fear needed to start a check-in",
    )
    parser.add_argument(
        "--sad-instant-confidence",
        type=float,
        default=0.9,
        help="summed distress that starts the check-in without waiting out the hold",
    )
    parser.add_argument("--sad-hold-seconds", type=float, default=1.0)
    parser.add_argument("--sad-cooldown", type=float, default=30.0)
    parser.add_argument("--sad-reset-seconds", type=float, default=1.5)
    parser.add_argument("--speak-on-start", action="store_true")
    parser.add_argument(
        "--no-check-in",
        action="store_true",
        help="play the recorded prompt instead of a spoken LLM check-in",
    )
    parser.add_argument("--check-in-turns", type=int, default=3)
    parser.add_argument(
        "--check-in-answer-timeout",
        type=float,
        default=8.0,
        help="seconds to wait for the person to start answering",
    )
    parser.add_argument(
        "--check-in-trailing-silence",
        type=float,
        default=0.7,
        help="silence that ends an answer and starts Baymax's reply",
    )
    parser.add_argument("--env", type=Path, default=SCRIPT_DIR.parent / ".env")
    parser.add_argument("--mic-gain", type=float, default=3.0)
    parser.add_argument(
        "--whisper-bin",
        default="/home/bracketbot/.local/share/whisper.cpp/build/bin/whisper-cli",
    )
    parser.add_argument(
        "--whisper-model",
        default="/home/bracketbot/.local/share/whisper.cpp/models/ggml-base.en.bin",
    )
    parser.add_argument("--whisper-threads", type=int, default=4)
    parser.add_argument("--espeak-bin", default="espeak-ng")
    parser.add_argument(
        "--tts-url",
        default="",
        help="natural-voice service; defaults to $LOCAL_TTS_URL",
    )
    parser.add_argument("--dashboard-host", default="0.0.0.0")
    parser.add_argument("--dashboard-port", type=int, default=8018)
    parser.add_argument(
        "--scan-interval",
        type=float,
        default=0.25,
        help="minimum seconds between frame-processing starts",
    )
    parser.add_argument(
        "--log-interval",
        type=float,
        default=5.0,
        help="seconds between status lines in continuous mode",
    )
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--snapshot", type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    for name in (
        "yolo_confidence",
        "face_confidence",
        "sad_confidence",
        "sad_instant_confidence",
        "pose_confidence",
    ):
        if not 0 <= getattr(args, name) <= 1:
            raise SystemExit(f"--{name.replace('_', '-')} must be between 0 and 1")
    if args.expression_interval < 1:
        raise SystemExit("--expression-interval must be at least 1")
    if args.min_face_size < 0:
        raise SystemExit("--min-face-size cannot be negative")
    if args.check_in_turns < 1:
        raise SystemExit("--check-in-turns must be at least 1")
    if args.check_in_answer_timeout <= 0:
        raise SystemExit("--check-in-answer-timeout must be greater than zero")
    if not 0 < args.expression_smoothing <= 1:
        raise SystemExit("--expression-smoothing must be greater than 0 and at most 1")
    if not 0 < args.expression_attack <= 1:
        raise SystemExit("--expression-attack must be greater than 0 and at most 1")
    if args.sad_instant_confidence < args.sad_confidence:
        raise SystemExit(
            "--sad-instant-confidence cannot be below --sad-confidence"
        )
    if args.check_in_trailing_silence <= 0:
        raise SystemExit("--check-in-trailing-silence must be greater than zero")
    for name in ("sad_hold_seconds", "sad_cooldown", "sad_reset_seconds"):
        if getattr(args, name) < 0:
            raise SystemExit(f"--{name.replace('_', '-')} cannot be negative")
    for name in ("ground_hold_seconds", "ground_clear_seconds"):
        if getattr(args, name) < 0:
            raise SystemExit(f"--{name.replace('_', '-')} cannot be negative")
    if args.depth_search_radius < 0 or args.depth_search_radius > 30:
        raise SystemExit("--depth-search-radius must be between 0 and 30")
    if args.log_interval <= 0:
        raise SystemExit("--log-interval must be greater than zero")
    if args.scan_interval <= 0:
        raise SystemExit("--scan-interval must be greater than zero")
    if not 1 <= args.dashboard_port <= 65535:
        raise SystemExit("--dashboard-port must be between 1 and 65535")
    try:
        return run(args)
    except (FileNotFoundError, RuntimeError) as exc:
        raise SystemExit(f"error: {exc}") from exc


if __name__ == "__main__":
    raise SystemExit(main())
