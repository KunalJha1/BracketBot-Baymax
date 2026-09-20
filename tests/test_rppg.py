import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("scipy")
pytest.importorskip("cv2")

import rppg  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import robot_rppg  # noqa: E402


def synthetic_skin(bpm, seconds=12.0, fps=30.0, amplitude=0.004, jitter=0.004, seed=0):
    """Mean-RGB trace of skin with a small pulse on top, sampled at slightly irregular times."""
    rng = np.random.default_rng(seed)
    t = np.arange(0, seconds, 1 / fps) + rng.uniform(-jitter, jitter, int(seconds * fps))
    t = np.sort(t)
    pulse = np.sin(2 * np.pi * bpm / 60 * t)
    base = np.array([180.0, 130.0, 110.0])
    weights = np.array([0.3, 1.0, 0.5])                       # green carries most of the pulse
    rgb = base * (1 + amplitude * pulse[:, None] * weights) + rng.normal(0, 0.05, (len(t), 3))
    return t, rgb


@pytest.mark.parametrize("bpm", [55, 72, 96, 130])
def test_pos_recovers_synthetic_pulse(bpm):
    t, rgb = synthetic_skin(bpm)
    result = rppg.analyze(t, rgb, fs=30.0)
    assert result is not None
    assert abs(result["bpm"] - bpm) < 2.0
    assert result["snr_db"] > 0


def test_analyze_needs_six_seconds():
    t, rgb = synthetic_skin(72, seconds=5.0)
    assert rppg.analyze(t, rgb) is None
    assert rppg.estimate_hr(t, rgb) == (None, None)


def test_tracking_window_restricts_peak_search():
    t, rgb = synthetic_skin(72)
    result = rppg.analyze(t, rgb, prev_bpm=70.0, track_bpm=10.0)
    assert 60.0 <= result["bpm"] <= 80.0


def test_pure_noise_has_lower_snr_than_pulse():
    t, rgb = synthetic_skin(72)
    noise = np.random.default_rng(1).normal(150, 1.0, rgb.shape)
    assert rppg.analyze(t, noise)["snr_db"] < rppg.analyze(t, rgb)["snr_db"]


def test_resample_uniform_handles_jitter():
    t, rgb = synthetic_skin(72)
    tu, out = rppg.resample_uniform(t, rgb, 30.0)
    assert np.allclose(np.diff(tu), 1 / 30.0)
    assert out.shape == (len(tu), 3)


def test_default_model_ships_with_repo():
    assert rppg.DEFAULT_MODEL.exists()


def test_face_skin_mask_covers_three_skin_regions_without_background():
    # YuNet: bbox, right eye, left eye, nose, mouth corners, confidence.
    face = np.array(
        [20, 10, 100, 120, 50, 50, 90, 50, 70, 70, 52, 100, 88, 100, 0.99],
        dtype=np.float32,
    )
    mask = rppg.face_skin_mask(face, (160, 160, 3))

    assert mask.shape == (160, 160)
    assert mask[25, 70] == 255       # forehead
    assert mask[80, 45] == 255       # left cheek
    assert mask[80, 95] == 255       # right cheek
    assert mask[70, 70] == 0         # nose gap
    assert mask[150, 150] == 0       # background


def test_yunet_face_roi_loads_real_shipped_model():
    roi = rppg.FaceROI()
    frame = np.zeros((240, 320, 3), np.uint8)

    assert roi(frame, 0) is None


class FakeReader:
    """Stands in for bbos.Reader on a 2560x960 head-camera frame: left eye red, right eye blue."""

    def __init__(self, ready=True):
        stereo = np.zeros((960, 2560, 3), np.uint8)
        stereo[:, :1280] = (255, 0, 0)                        # RGB order, as the raw topic publishes it
        stereo[:, 1280:] = (0, 0, 255)
        self.data = {"rgb": stereo}
        self._ready = ready

    def ready(self):
        return self._ready


def fake_camera(reader):
    cam = robot_rppg.HeadCamera.__new__(robot_rppg.HeadCamera)
    cam.jpeg, cam.reader, cam.frames, cam.first, cam.last = False, reader, 0, None, None
    return cam


def test_head_camera_returns_left_eye_as_bgr():
    frame = fake_camera(FakeReader()).grab()
    assert frame.shape == (960, 1280, 3)
    assert tuple(frame[0, 0]) == (0, 0, 255)                  # red, expressed in BGR
    assert tuple(frame[-1, -1]) == (0, 0, 255)                # still the left eye at its far edge


def test_head_camera_returns_none_until_a_frame_is_ready():
    cam = fake_camera(FakeReader(ready=False))
    assert cam.grab() is None
    assert cam.frames == 0
    assert cam.fps == 0.0


def test_head_camera_jpeg_path_splits_the_eye():
    import cv2

    stereo = np.zeros((960, 2560, 3), np.uint8)
    stereo[:, :1280] = (0, 200, 0)
    stereo[:, 1280:] = (0, 0, 200)
    ok, encoded = cv2.imencode(".jpg", stereo)
    assert ok
    buffer = np.zeros(4_000_000, np.uint8)
    buffer[:len(encoded)] = encoded.ravel()
    reader = FakeReader()
    reader.data = {"jpeg": buffer, "jpeg_len": len(encoded)}
    cam = fake_camera(reader)
    cam.jpeg = True
    frame = cam.grab()
    assert frame.shape == (960, 1280, 3)
    assert frame[480, 640, 1] > 150 and frame[480, 640, 2] < 50


class SteppingClock:
    def __init__(self, step=1 / 30):
        self.now = -step
        self.step = step

    def __call__(self):
        self.now += self.step
        return self.now


class SyntheticFaceROI:
    """Stable synthetic skin signal with realistic one-frame detector misses."""

    def __init__(self, bpm=72, miss_every=10):
        self.bpm = bpm
        self.miss_every = miss_every
        self.calls = 0

    def __call__(self, _frame, t_ms):
        self.calls += 1
        if self.miss_every and self.calls % self.miss_every == 0:
            return None
        t = t_ms / 1000
        pulse = np.sin(2 * np.pi * self.bpm / 60 * t)
        base = np.array([180.0, 130.0, 110.0])
        weights = np.array([0.3, 1.0, 0.5])
        rgb = base * (1 + 0.004 * pulse * weights)
        return rgb, np.array([50.0, 50.0]), 100.0, None


def test_measurement_tolerates_brief_face_detector_misses(monkeypatch):
    roi = SyntheticFaceROI()
    monkeypatch.setattr(rppg, "FaceROI", lambda _model: roi)
    monkeypatch.setattr(rppg.time, "monotonic", SteppingClock())

    result = rppg.measure_heart_rate(
        duration_s=12.0,
        window_s=10.0,
        fs=30.0,
        grab=lambda: np.zeros((4, 4, 3), np.uint8),
    )

    assert result is not None
    assert abs(result["bpm"] - 72) < 2
    assert result["n_estimates"] >= 3


def test_measurement_does_not_publish_rejected_peak(monkeypatch):
    monkeypatch.setattr(rppg, "FaceROI", lambda _model: SyntheticFaceROI(miss_every=0))
    monkeypatch.setattr(rppg.time, "monotonic", SteppingClock())
    monkeypatch.setattr(rppg, "estimate_hr", lambda *_args, **_kwargs: (72.0, -10.0))
    updates = []

    result = rppg.measure_heart_rate(
        duration_s=7.2,
        fs=30.0,
        snr_min_db=-1.0,
        on_update=lambda bpm, snr, progress: updates.append((bpm, snr, progress)),
        grab=lambda: np.zeros((4, 4, 3), np.uint8),
    )

    assert result is None
    assert updates
    assert all(bpm is None for bpm, _snr, _progress in updates)
