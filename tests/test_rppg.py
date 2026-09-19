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
