"""Run person and facial-expression detection on a laptop camera or video stream.

The default model is the YOLO pose checkpoint already committed to this
repository.  A pose checkpoint only has a ``person`` class, so it is also a
small, convenient person detector for this first local prototype. OpenCV
YuNet finds the primary face and EmotiEffLib estimates its visible expression.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import platform
import shutil
import subprocess
import time
from typing import Any


@dataclass(frozen=True)
class Detection:
    """One person bounding box in pixel coordinates."""

    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float


@dataclass(frozen=True)
class Expression:
    """Smoothed visible-expression estimate for the primary face."""

    x1: int
    y1: int
    x2: int
    y2: int
    label: str
    confidence: float


@dataclass
class SadVoiceTrigger:
    """Debounce a visible-sadness cue before asking a friendly question.

    A cue must remain present for ``hold_seconds`` before it triggers. After
    speaking, it must clear for ``reset_seconds`` and the cooldown must expire
    before another cue can trigger. This keeps noisy frame-level predictions
    from repeatedly talking to the same person.
    """

    hold_seconds: float
    cooldown_seconds: float
    reset_seconds: float
    confidence: float
    first_sad_at: float | None = None
    first_clear_at: float | None = None
    last_triggered_at: float | None = None
    armed: bool = True

    def update(
        self,
        expression: Expression | None,
        person_present: bool,
        now: float,
    ) -> bool:
        sad_visible = (
            person_present
            and expression is not None
            and expression.label in {"sad", "sadness"}
            and expression.confidence >= self.confidence
        )

        if sad_visible:
            self.first_clear_at = None
            if self.first_sad_at is None:
                self.first_sad_at = now
            cooldown_over = (
                self.last_triggered_at is None
                or now - self.last_triggered_at >= self.cooldown_seconds
            )
            if (
                self.armed
                and cooldown_over
                and now - self.first_sad_at >= self.hold_seconds
            ):
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


class SystemSpeaker:
    """Speak text asynchronously through an installed system TTS command."""

    def __init__(self, command: str) -> None:
        self.command = command
        self.process: subprocess.Popen[bytes] | None = None

    @classmethod
    def create(cls) -> SystemSpeaker:
        candidates = (
            ("say", "spd-say", "espeak-ng", "espeak")
            if platform.system() == "Darwin"
            else ("spd-say", "espeak-ng", "espeak", "say")
        )
        command = next((path for name in candidates if (path := shutil.which(name))), None)
        if command is None:
            raise RuntimeError(
                "Sad-voice response is enabled, but no supported text-to-speech "
                "command was found (say, spd-say, espeak-ng, or espeak). Use "
                "--no-sad-voice to run without speech."
            )
        return cls(command)

    def speak(self, text: str) -> bool:
        """Start speaking unless an earlier utterance is still playing."""

        if self.process is not None and self.process.poll() is None:
            return False
        self.process = subprocess.Popen(
            [self.command, text],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return True

    def close(self) -> None:
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()


def smooth_scores(previous: Any, current: Any, alpha: float) -> Any:
    """Exponential smoothing that keeps webcam predictions from flickering."""

    if previous is None:
        return current.copy()
    return (1.0 - alpha) * previous + alpha * current


class ExpressionAnalyzer:
    """Detect the largest face and classify its visible expression locally."""

    def __init__(
        self,
        face_model: Path,
        expression_model: Path,
        smoothing: float,
        face_confidence: float,
    ) -> None:
        import cv2
        import emotiefflib.facial_analysis as facial_analysis
        from unittest.mock import patch

        if not face_model.exists():
            raise FileNotFoundError(f"Face detector model not found: {face_model}")
        if not expression_model.exists():
            raise FileNotFoundError(f"Expression model not found: {expression_model}")

        self.face_detector = cv2.FaceDetectorYN.create(
            str(face_model), "", (320, 320), face_confidence, 0.3, 50
        )
        # EmotiEffLib normally downloads this model into a user cache. Point its
        # ONNX loader at our checked-in asset so startup is deterministic/offline.
        with patch.object(
            facial_analysis,
            "get_model_path_onnx",
            return_value=str(expression_model),
        ):
            self.recognizer = facial_analysis.EmotiEffLibRecognizer(
                engine="onnx", model_name="enet_b0_8_best_afew"
            )
        self.smoothing = smoothing
        self.scores = None
        self.missing_frames = 0

    def analyze(self, frame: Any, classify: bool = True) -> Expression | None:
        import cv2
        import numpy as np

        height, width = frame.shape[:2]
        self.face_detector.setInputSize((width, height))
        _, faces = self.face_detector.detect(frame)
        if faces is None or len(faces) == 0:
            self.missing_frames += 1
            if self.missing_frames > 15:
                self.scores = None
            return None

        self.missing_frames = 0
        face = max(faces, key=lambda row: float(row[2] * row[3]))
        x, y, face_width, face_height = (float(value) for value in face[:4])
        # Square, unpadded crops match the classifier's training framing; on
        # RAF-DB they beat the old 12%-padded rectangle by ~3 accuracy points.
        side = max(face_width, face_height)
        center_x, center_y = x + face_width / 2, y + face_height / 2
        x1 = max(0, round(center_x - side / 2))
        y1 = max(0, round(center_y - side / 2))
        x2 = min(width, round(center_x + side / 2))
        y2 = min(height, round(center_y + side / 2))

        if classify or self.scores is None:
            face_rgb = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
            _, scores = self.recognizer.predict_emotions(face_rgb, logits=False)
            current = np.asarray(scores[0], dtype=float)
            self.scores = smooth_scores(self.scores, current, self.smoothing)

        emotion_count = len(self.recognizer.idx_to_emotion_class)
        emotion_scores = self.scores[:emotion_count].copy()
        # Contempt is rarely right on a live camera and mostly steals from
        # neutral, so it never wins the label.
        for index, name in self.recognizer.idx_to_emotion_class.items():
            if name.lower() == "contempt":
                emotion_scores[index] = 0.0
        emotion_scores /= max(float(emotion_scores.sum()), 1e-9)
        best_index = int(np.argmax(emotion_scores))
        return Expression(
            x1,
            y1,
            x2,
            y2,
            self.recognizer.idx_to_emotion_class[best_index].lower(),
            float(emotion_scores[best_index]),
        )


def detections_from_result(result: Any) -> list[Detection]:
    """Convert an Ultralytics result into plain person detections.

    Keeping this conversion separate from camera handling makes the detector
    straightforward to reuse with BracketBot frames in the next milestone.
    """

    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return []

    coordinates = boxes.xyxy.detach().cpu().numpy()
    confidences = boxes.conf.detach().cpu().numpy()
    classes = boxes.cls.detach().cpu().numpy().astype(int)
    return [
        Detection(*(round(float(value)) for value in xyxy), float(confidence))
        for xyxy, confidence, class_id in zip(coordinates, confidences, classes)
        if class_id == 0
    ]


def draw_overlay(
    frame: Any,
    detections: list[Detection],
    fps: float,
    expression: Expression | None = None,
) -> Any:
    """Draw person boxes, the primary expression, and live status in place."""

    import cv2

    for detection in detections:
        color = (0, 220, 80)
        cv2.rectangle(
            frame,
            (detection.x1, detection.y1),
            (detection.x2, detection.y2),
            color,
            2,
        )
        label = f"person {detection.confidence:.0%}"
        (text_width, text_height), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1
        )
        label_top = max(0, detection.y1 - text_height - 9)
        cv2.rectangle(
            frame,
            (detection.x1, label_top),
            (detection.x1 + text_width + 8, detection.y1),
            color,
            -1,
        )
        cv2.putText(
            frame,
            label,
            (detection.x1 + 4, max(text_height, detection.y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

    if expression is not None:
        color = (255, 80, 220)
        cv2.rectangle(
            frame,
            (expression.x1, expression.y1),
            (expression.x2, expression.y2),
            color,
            2,
        )
        label = f"expression: {expression.label} {expression.confidence:.0%}"
        (text_width, text_height), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.58, 1
        )
        if expression.y2 + text_height + 10 < frame.shape[0]:
            label_top = expression.y2
            label_y = expression.y2 + text_height + 6
        else:
            label_top = max(0, expression.y1 - text_height - 9)
            label_y = max(text_height, expression.y1 - 5)
        cv2.rectangle(
            frame,
            (expression.x1, label_top),
            (min(frame.shape[1], expression.x1 + text_width + 8), label_y + 4),
            color,
            -1,
        )
        cv2.putText(
            frame,
            label,
            (expression.x1 + 4, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

    status = f"people: {len(detections)}   FPS: {fps:.1f}   Q/Esc: quit"
    cv2.putText(
        frame,
        status,
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return frame


def parse_source(value: str) -> int | str:
    """Treat a numeric source as a camera index and everything else as a path/URL."""

    return int(value) if value.isdigit() else value


def open_capture(source: int | str, width: int, height: int) -> Any:
    import cv2

    if isinstance(source, int) and platform.system() == "Darwin":
        capture = cv2.VideoCapture(source, cv2.CAP_AVFOUNDATION)
    else:
        capture = cv2.VideoCapture(source)

    if isinstance(source, int):
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    if not capture.isOpened():
        capture.release()
        hint = ""
        if platform.system() == "Darwin" and isinstance(source, int):
            hint = (
                " On macOS, allow camera access for Terminal (or your IDE) in "
                "System Settings > Privacy & Security > Camera."
            )
        raise RuntimeError(f"Could not open video source {source!r}.{hint}")
    return capture


def run(args: argparse.Namespace) -> int:
    import cv2
    from ultralytics import YOLO

    if not args.model.exists():
        raise FileNotFoundError(
            f"Model not found: {args.model}. The default yolo11n-pose.pt should "
            "be present at the repository root."
        )

    print(f"Loading {args.model} ...", flush=True)
    model = YOLO(str(args.model))
    expression_analyzer = None
    sad_trigger = None
    speaker = None
    if not args.no_expression:
        print("Loading YuNet + EmotiEffLib expression models ...", flush=True)
        expression_analyzer = ExpressionAnalyzer(
            args.face_model,
            args.expression_model,
            args.expression_smoothing,
            args.face_confidence,
        )
        if not args.no_sad_voice:
            speaker = SystemSpeaker.create()
            sad_trigger = SadVoiceTrigger(
                hold_seconds=args.sad_hold_seconds,
                cooldown_seconds=args.sad_cooldown,
                reset_seconds=args.sad_reset_seconds,
                confidence=args.sad_confidence,
            )
            print(
                f"Sad-expression voice cue enabled via {speaker.command!r}.",
                flush=True,
            )
    source = parse_source(args.source)
    capture = open_capture(source, args.width, args.height)
    writer = None
    frame_count = 0
    smoothed_fps = 0.0
    previous_time = time.perf_counter()

    try:
        print(f"Detecting people from {source!r}. Press Q or Esc to stop.", flush=True)
        while True:
            ok, frame = capture.read()
            if not ok:
                if frame_count == 0:
                    raise RuntimeError(f"Opened video source {source!r}, but received no frames.")
                break

            result = model.predict(
                frame,
                classes=[0],
                conf=args.confidence,
                imgsz=args.imgsz,
                device=args.device,
                verbose=False,
            )[0]
            detections = detections_from_result(result)
            expression = None
            if expression_analyzer is not None:
                expression = expression_analyzer.analyze(
                    frame,
                    classify=frame_count % args.expression_interval == 0,
                )

            if (
                sad_trigger is not None
                and speaker is not None
                and sad_trigger.update(expression, bool(detections), time.monotonic())
            ):
                if speaker.speak(args.sad_voice_text):
                    print(
                        f"Sad expression persisted; speaking: {args.sad_voice_text!r}",
                        flush=True,
                    )

            now = time.perf_counter()
            instantaneous_fps = 1.0 / max(now - previous_time, 1e-9)
            smoothed_fps = (
                instantaneous_fps
                if frame_count == 0
                else 0.1 * instantaneous_fps + 0.9 * smoothed_fps
            )
            previous_time = now
            draw_overlay(frame, detections, smoothed_fps, expression)
            frame_count += 1

            if args.output is not None:
                if writer is None:
                    args.output.parent.mkdir(parents=True, exist_ok=True)
                    output_fps = capture.get(cv2.CAP_PROP_FPS)
                    if not 1 <= output_fps <= 240:
                        output_fps = 30.0
                    writer = cv2.VideoWriter(
                        str(args.output),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        output_fps,
                        (frame.shape[1], frame.shape[0]),
                    )
                    if not writer.isOpened():
                        raise RuntimeError(f"Could not create output video: {args.output}")
                writer.write(frame)

            if not args.no_display:
                cv2.imshow("Person + expression detector", frame)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break

            if args.max_frames and frame_count >= args.max_frames:
                break
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        if not args.no_display:
            cv2.destroyAllWindows()
        if speaker is not None:
            speaker.close()

    print(f"Processed {frame_count} frame(s).", flush=True)
    if args.output is not None:
        print(f"Saved annotated video to {args.output}", flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Detect people and estimate the primary face's visible expression "
            "in a laptop webcam or video stream."
        )
    )
    parser.add_argument(
        "--source",
        default="0",
        help="camera index, video path, or stream URL (default: laptop camera 0)",
    )
    parser.add_argument("--model", type=Path, default=Path("yolo11n-pose.pt"))
    parser.add_argument("--confidence", type=float, default=0.4)
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO inference image size")
    parser.add_argument("--device", default=None, help="e.g. cpu, mps, 0; default: Ultralytics choice")
    parser.add_argument(
        "--face-model",
        type=Path,
        default=Path("assets/models/face_detection_yunet_2026may.onnx"),
    )
    parser.add_argument(
        "--expression-model",
        type=Path,
        default=Path("assets/models/enet_b0_8_best_afew.onnx"),
    )
    parser.add_argument("--face-confidence", type=float, default=0.75)
    parser.add_argument(
        "--expression-interval",
        type=int,
        default=3,
        help="classify the face every N frames (face detection still runs every frame)",
    )
    parser.add_argument(
        "--expression-smoothing",
        type=float,
        default=0.25,
        help="new-frame weight for expression smoothing",
    )
    parser.add_argument(
        "--no-expression",
        action="store_true",
        help="run only YOLO person detection",
    )
    parser.add_argument(
        "--no-sad-voice",
        action="store_true",
        help="disable the spoken response to a sustained sad expression",
    )
    parser.add_argument(
        "--sad-voice-text",
        default="Hey, you look a little sad. Are you okay?",
        help="text spoken after a sustained sad-expression cue",
    )
    parser.add_argument(
        "--sad-confidence",
        type=float,
        default=0.6,
        help="minimum smoothed sad-expression confidence before speaking",
    )
    parser.add_argument(
        "--sad-hold-seconds",
        type=float,
        default=1.5,
        help="how long a sad expression must persist before speaking",
    )
    parser.add_argument(
        "--sad-cooldown",
        type=float,
        default=30.0,
        help="minimum seconds between sad-expression voice responses",
    )
    parser.add_argument(
        "--sad-reset-seconds",
        type=float,
        default=2.0,
        help="how long the cue must clear before it can trigger again",
    )
    parser.add_argument("--width", type=int, default=1280, help="requested camera width")
    parser.add_argument("--height", type=int, default=720, help="requested camera height")
    parser.add_argument("--output", type=Path, help="optional annotated .mp4 recording")
    parser.add_argument("--no-display", action="store_true", help="do not open a preview window")
    parser.add_argument("--max-frames", type=int, default=0, help="stop after N frames; 0 runs until quit")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not 0 <= args.confidence <= 1:
        raise SystemExit("--confidence must be between 0 and 1")
    if not 0 <= args.face_confidence <= 1:
        raise SystemExit("--face-confidence must be between 0 and 1")
    if not 0 <= args.sad_confidence <= 1:
        raise SystemExit("--sad-confidence must be between 0 and 1")
    if not 0 < args.expression_smoothing <= 1:
        raise SystemExit("--expression-smoothing must be greater than 0 and at most 1")
    if args.expression_interval < 1:
        raise SystemExit("--expression-interval must be at least 1")
    if args.sad_hold_seconds < 0:
        raise SystemExit("--sad-hold-seconds cannot be negative")
    if args.sad_cooldown < 0:
        raise SystemExit("--sad-cooldown cannot be negative")
    if args.sad_reset_seconds < 0:
        raise SystemExit("--sad-reset-seconds cannot be negative")
    if not args.sad_voice_text.strip():
        raise SystemExit("--sad-voice-text cannot be empty")
    if args.no_display and not args.max_frames and isinstance(parse_source(args.source), int):
        print("Headless camera mode runs until Ctrl-C.", flush=True)
    try:
        return run(args)
    except (FileNotFoundError, RuntimeError) as exc:
        raise SystemExit(f"error: {exc}") from exc


if __name__ == "__main__":
    raise SystemExit(main())
