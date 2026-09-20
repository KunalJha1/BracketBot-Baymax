# /// script
# requires-python = "==3.10.*"
# dependencies = [
#   "bbos",
#   "numpy",
#   "scipy",
#   "opencv-python",
# ]
# [tool.uv.sources]
# bbos = { path = "/home/bracketbot/bbos", editable = true }
# ///
"""Contactless heart-rate scan from the BracketBot head camera. Runs ON THE ROBOT, read-only.

    scp scripts/robot_rppg.py rppg.py assets/models/face_detection_yunet_2026may.onnx bot:/tmp/
    ssh bot 'cd /tmp && ~/.local/bin/uv run robot_rppg.py --check'   # camera + face gate, no scan
    ssh bot 'cd /tmp && ~/.local/bin/uv run robot_rppg.py'           # 15 s scan, JSON on stdout

Opens one Reader on the head camera and never a Writer, so it cannot move the robot or take a topic
from its owner. It stores no frames: only per-frame mean skin RGB is kept, in memory. Run it from
/tmp (in ~ the bbos project folder shadows the bbos package). The robot should be stationary and
the person should have agreed to the scan; ask them to face the camera and hold still.

Not a medical device. The result is a demo-grade estimate: it must not be used to diagnose, treat,
triage, or trigger any robot action.
"""

import argparse
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))  # rppg.py at the repo root when run from a checkout

from rppg import DEFAULT_MODEL, FaceROI, measure_heart_rate  # noqa: E402

RAW_TOPIC = "camera.head.rgb"    # raw RGB: no JPEG blocking on a ~0.1-1 % pulse signal
JPEG_TOPIC = "camera.head.jpeg"
MIN_CAMERA_FPS = 15.0
MIN_FACE_FRACTION = 0.8
MIN_FACE_WIDTH_PX = 50.0
MAX_LUM_DRIFT_PCT = 3.0


class HeadCamera:
    """The head camera's left eye as BGR frames. The topic is 2560x960: two 1280x960 eyes side by side,
    so the frame is split at half its width and never processed as one normal camera image."""

    def __init__(self, jpeg=False):
        from bbos import Reader

        self.jpeg = jpeg
        self.reader = Reader(JPEG_TOPIC if jpeg else RAW_TOPIC, keeptime=False)
        self.frames = 0
        self.first = self.last = None

    def __enter__(self):
        self.reader.__enter__()
        return self

    def __exit__(self, *exc):
        return self.reader.__exit__(*exc)

    def grab(self):
        """Newest left-eye BGR frame, or None when the camera has not published a new one yet."""
        if not self.reader.ready():
            time.sleep(0.002)
            return None
        data = self.reader.data
        if self.jpeg:
            stereo = cv2.imdecode(np.frombuffer(bytes(data["jpeg"][:int(data["jpeg_len"])]), np.uint8),
                                  cv2.IMREAD_COLOR)
            left = stereo[:, :stereo.shape[1] // 2]
            frame = np.ascontiguousarray(left)
        else:
            stereo = data["rgb"]
            frame = cv2.cvtColor(stereo[:, :stereo.shape[1] // 2], cv2.COLOR_RGB2BGR)  # copies out of shm
        self.last = time.monotonic()
        self.first = self.first or self.last
        self.frames += 1
        return frame

    @property
    def fps(self):
        return self.frames / (self.last - self.first) if self.frames > 1 and self.last > self.first else 0.0


def camera_check_failures(seen, fps, faces, face_width, lum_drift):
    """Human-readable reasons a camera sample is not suitable for rPPG."""
    failures = []
    if not seen:
        failures.append("no camera frames")
    elif fps < MIN_CAMERA_FPS:
        failures.append(f"camera processing below {MIN_CAMERA_FPS:g} FPS")
    if seen and faces < MIN_FACE_FRACTION * seen:
        failures.append(f"face found in less than {MIN_FACE_FRACTION:.0%} of frames")
    if face_width is None or face_width < MIN_FACE_WIDTH_PX:
        failures.append(f"face is smaller than {MIN_FACE_WIDTH_PX:g} px; move closer")
    if lum_drift is None or lum_drift > MAX_LUM_DRIFT_PCT:
        failures.append(f"skin luminance drift exceeds {MAX_LUM_DRIFT_PCT:g}%")
    return failures


def check(cam, model, seconds):
    """Camera + face gate: is the eye split right, is the frame rate usable, is a face big enough?"""
    roi = FaceROI(model)
    seen = faces = 0
    widths, lums = [], []
    shape = None
    t0 = time.monotonic()
    while time.monotonic() - t0 < seconds:
        frame = cam.grab()
        if frame is None:
            continue
        shape = frame.shape
        seen += 1
        r = roi(frame, time.monotonic() * 1000)
        if r is not None:
            faces += 1
            widths.append(r[2])
            lums.append(float(np.dot(r[0], [0.299, 0.587, 0.114])))
    lum_drift = 100 * float(np.ptp(lums)) / (float(np.mean(lums)) + 1e-9) if lums else None
    face_width = float(np.median(widths)) if widths else None
    failures = camera_check_failures(seen, cam.fps, faces, face_width, lum_drift)
    return {"frames": seen, "fps": round(cam.fps, 1),
            "eye_shape": None if shape is None else list(shape),
            "face_frames": faces,
            "face_width_px": None if face_width is None else round(face_width, 0),
            "lum_drift_pct": None if lum_drift is None else round(lum_drift, 2),
            "failures": failures,
            "ok": not failures}


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--check", action="store_true", help="grab frames and report camera/face health; no scan")
    p.add_argument(
        "--self-test", action="store_true",
        help="load OpenCV and the face model without opening the robot camera",
    )
    p.add_argument("--jpeg", action="store_true", help=f"read {JPEG_TOPIC} instead of raw {RAW_TOPIC} (lossier)")
    p.add_argument("--duration", type=float, default=15.0, help="scan length in seconds")
    p.add_argument("--window", type=float, default=10.0, help="analysis window in seconds")
    p.add_argument("--fs", type=float, default=30.0, help="resample rate in Hz")
    p.add_argument(
        "--model", type=Path,
        help="YuNet face-detection ONNX model (default: next to this script)",
    )
    p.add_argument("--progress-json", action="store_true",
                   help="stream one compact JSON object per line on stdout (per-second estimates "
                        "tagged \"progress\", then the final \"result\" line) instead of a human report")
    a = p.parse_args()

    model = a.model or next(
        (m for m in (HERE / "face_detection_yunet_2026may.onnx", DEFAULT_MODEL) if m.exists()), None
    )
    if model is None or not model.exists():
        sys.exit(
            "YuNet face model not found: copy assets/models/"
            "face_detection_yunet_2026may.onnx next to this script"
        )
    if a.self_test:
        FaceROI(model)  # Constructor parses the ONNX graph; failure must stop deployment.
        print(json.dumps({
            "ok": True,
            "opencv": cv2.__version__,
            "model": model.name,
            "model_bytes": model.stat().st_size,
        }))
        return

    with HeadCamera(a.jpeg) as cam:
        t0 = time.monotonic()
        while (frame := cam.grab()) is None:
            if time.monotonic() - t0 > 3:
                sys.exit("no camera frame: is the camera daemon running?")
        if a.check:
            print(json.dumps(check(cam, model, seconds=4.0), indent=2))
            return

        def progress(bpm, snr, done):
            if a.progress_json:
                # One line per estimate so a caller can speak intermediate readings
                # while the scan is still running.
                print(json.dumps({"progress": round(float(done), 3),
                                  "bpm": None if bpm is None else round(float(bpm), 1),
                                  "snr_db": None if snr is None else round(float(snr), 1)}),
                      flush=True)
                return
            print(f"{done * 100:5.1f}%  bpm={'--' if bpm is None else round(bpm, 1)}  "
                  f"snr={'--' if snr is None else round(snr, 1)} dB", file=sys.stderr, flush=True)

        def guidance(kind):
            if a.progress_json:
                print(json.dumps({"guidance": kind}), flush=True)
                return
            print(f"guidance: {kind}", file=sys.stderr, flush=True)

        result = measure_heart_rate(duration_s=a.duration, window_s=a.window, fs=a.fs,
                                    model_path=model, on_update=progress,
                                    on_guidance=guidance, grab=cam.grab)
        report = {"result": result, "fps": round(cam.fps, 1), "frames": cam.frames,
                  "note": "camera-based demo estimate, not a medical measurement"}
        # The streaming form must stay one object per line; the human form stays indented.
        print(json.dumps(report) if a.progress_json else json.dumps(report, indent=2), flush=True)
        return 0 if result else 1


if __name__ == "__main__":
    raise SystemExit(main() or 0)
