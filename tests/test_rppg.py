import json
import subprocess
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


def synthetic_ppg(bpm, seconds=14.0, fps=30.0, amplitude=0.004, harmonic=1.6, seed=0):
    """Skin trace shaped like a real pulse waveform rather than a sine.

    A camera sees the dicrotic notch, and its second harmonic is often the taller
    line in the spectrum - which is exactly when a plain peak search reports 2x
    the real rate.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(0, seconds, 1 / fps) + rng.uniform(-0.004, 0.004, int(seconds * fps))
    t = np.sort(t)
    phase = 2 * np.pi * bpm / 60 * t
    pulse = np.sin(phase) + harmonic * np.sin(2 * phase + 0.7)
    pulse /= np.abs(pulse).max()
    base = np.array([180.0, 130.0, 110.0])
    weights = np.array([0.3, 1.0, 0.5])
    return t, base * (1 + amplitude * pulse[:, None] * weights) + rng.normal(0, 0.05, (len(t), 3))


def run_tracker(t, rgb, period=1.0, **kw):
    """Feed a trace through HeartRateTracker the way a scan does, one estimate a second."""
    tracker = rppg.HeartRateTracker(**kw)
    next_est = t[0] + rppg.MIN_SECONDS
    for now, sample in zip(t, rgb):
        tracker.add(now, sample)
        if now >= next_est:
            next_est = now + period
            tracker.update(now)
    return tracker


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


def test_yunet_detection_uses_bounded_preview_but_full_resolution_mask(monkeypatch):
    face = np.array(
        [[10, 20, 100, 120, 40, 60, 80, 60, 60, 80, 45, 110, 75, 110, 0.99]],
        dtype=np.float32,
    )

    class Detector:
        def __init__(self):
            self.input_size = None
            self.detected_shape = None

        def setInputSize(self, size):
            self.input_size = size

        def detect(self, frame):
            self.detected_shape = frame.shape
            return None, face

    detector = Detector()
    roi = rppg.FaceROI.__new__(rppg.FaceROI)
    import cv2
    roi.cv2 = cv2
    roi.detector = detector
    roi.detector_size = None
    frame = np.full((960, 1280, 3), 120, np.uint8)

    result = roi(frame, 0)

    assert detector.input_size == (560, 420)
    assert detector.detected_shape == (420, 560, 3)
    assert result is not None
    _rgb, nose, face_width, mask = result
    assert face_width == pytest.approx(100 * 1280 / 560)
    assert nose.tolist() == pytest.approx([60 * 1280 / 560, 80 * 960 / 420])
    assert mask.shape == (960, 1280)


def test_robot_cli_self_test_loads_model_without_camera():
    script = Path(__file__).resolve().parents[1] / "scripts" / "robot_rppg.py"
    completed = subprocess.run(
        [sys.executable, str(script), "--self-test"],
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads(completed.stdout)

    assert report["ok"] is True
    assert report["model"] == rppg.DEFAULT_MODEL.name
    assert report["model_bytes"] == rppg.DEFAULT_MODEL.stat().st_size


@pytest.mark.parametrize("bpm", [54, 68, 82])
def test_a_taller_second_harmonic_does_not_double_the_reported_rate(bpm):
    t, rgb = synthetic_ppg(bpm)
    result = rppg.analyze(t, rgb, fs=30.0)
    assert abs(result["bpm"] - bpm) < 3.0, "locked onto the harmonic"


def test_fundamental_keeps_the_peak_when_half_of_it_is_only_noise():
    freqs = np.linspace(0.0, 5.0, 2001)
    psd = np.full_like(freqs, 1.0)                            # flat noise floor
    peak = int(np.argmin(np.abs(freqs - 2.0)))                # 120 BPM
    psd[peak] = 500.0
    psd[int(np.argmin(np.abs(freqs - 1.0)))] = 3.0            # a nothing bump at half
    assert rppg.fundamental(freqs, psd, peak) == peak


def test_fundamental_steps_down_to_a_real_peak_at_half_the_frequency():
    freqs = np.linspace(0.0, 5.0, 2001)
    psd = np.full_like(freqs, 1.0)
    peak = int(np.argmin(np.abs(freqs - 2.4)))                # 144 BPM: the harmonic
    half = int(np.argmin(np.abs(freqs - 1.2)))                # 72 BPM: the real rate
    psd[peak] = 500.0
    psd[half] = 200.0
    assert rppg.fundamental(freqs, psd, peak) == half


def test_tracking_can_climb_out_of_a_locked_harmonic():
    t, rgb = synthetic_ppg(70)
    result = rppg.analyze(t, rgb, prev_bpm=140.0, track_bpm=15.0)
    assert abs(result["bpm"] - 70) < 3.0


def test_longest_clean_span_picks_the_side_of_a_gap_worth_analysing():
    t = np.r_[np.arange(0, 3, 1 / 30), np.arange(5, 13, 1 / 30)]
    span = rppg.longest_clean_span(t)
    assert t[span][0] == pytest.approx(5.0)
    assert t[span][-1] == pytest.approx(t[-1])


def test_longest_clean_span_keeps_everything_when_no_stretch_is_long_enough():
    t = np.r_[np.arange(0, 4, 1 / 30), np.arange(6, 10, 1 / 30)]
    assert rppg.longest_clean_span(t) == slice(0, len(t))


def test_analyze_reads_the_clean_stretch_instead_of_interpolating_over_a_gap():
    rng = np.random.default_rng(3)
    t_junk = np.arange(0, 4.0, 1 / 30)
    junk = rng.normal(150, 1.5, (len(t_junk), 3))
    t_good, good = synthetic_ppg(66, seconds=9.0, harmonic=0.0, seed=4)
    t = np.r_[t_junk, t_good + 5.0]                           # 1 s hole between them
    result = rppg.analyze(t, np.vstack([junk, good]), fs=30.0)
    assert abs(result["bpm"] - 66) < 3.0


def test_tracker_aggregates_a_clean_scan_and_calls_it_confident():
    t, rgb = synthetic_ppg(75, seconds=20.0, seed=5)
    result = run_tracker(t, rgb).result()
    assert abs(result["bpm"] - 75) < 3.0
    assert result["confident"] and result["n_estimates"] >= 3
    assert result["spread_bpm"] <= 6.0 and result["span_s"] >= 6.0


def test_tracker_returns_nothing_from_a_scan_with_no_pulse_in_it():
    t = np.arange(0, 20.0, 1 / 30)
    noise = np.random.default_rng(7).normal(150, 1.0, (len(t), 3))
    assert run_tracker(t, noise, snr_min_db=6.0).result() is None


def test_tracker_unlocks_its_search_after_a_run_of_rejected_estimates():
    t, rgb = synthetic_ppg(75, seconds=20.0, seed=6)
    tracker = run_tracker(t, rgb, snr_min_db=-1.0)
    assert tracker.prev is not None
    tracker.snr_min_db = 99.0                                 # nothing can pass from here on
    for _ in range(tracker.unlock_after):
        tracker.update(t[-1])
    assert tracker.prev is None


def test_tracker_distrusts_a_rate_that_only_showed_up_at_the_end_of_the_scan():
    late = rppg.HeartRateTracker()
    late.estimates = [(18.0, 58.0, 4.0), (19.0, 58.4, 4.2), (20.0, 58.2, 4.1)]
    assert late.result()["confident"] is False
    steady = rppg.HeartRateTracker()
    steady.estimates = [(8.0, 58.0, 4.0), (14.0, 58.4, 4.2), (20.0, 58.2, 4.1)]
    assert steady.result()["confident"] is True


def test_tracker_clear_keeps_the_estimates_and_only_unlocks_on_request():
    tracker = rppg.HeartRateTracker()
    tracker.add(0.0, [180.0, 130.0, 110.0])
    tracker.estimates = [(1.0, 70.0, 3.0)]
    tracker.prev = 70.0
    tracker.clear()
    assert not tracker.t and tracker.estimates and tracker.prev == 70.0
    tracker.clear(unlock=True)
    assert tracker.prev is None


def test_weighted_median_follows_the_estimates_the_chain_was_sure_about():
    bpms = [70.0, 71.0, 120.0]
    sure = rppg.weighted_median(bpms, [10.0, 10.0, 0.01])
    assert sure in (70.0, 71.0)
    assert rppg.weighted_median(bpms, [0.0, 0.0, 0.0]) == 71.0


def test_measure_heart_rate_scans_from_frames_and_reports_an_aggregate(monkeypatch):
    """The camera loop itself: frames in, aggregate out, with the motion gate in the
    path. No camera and no landmarker, and a fake clock so a 20 s scan takes no time."""
    t, rgb = synthetic_ppg(69, seconds=22.0, seed=8)
    clock = iter(np.r_[t, t[-1] + np.arange(1, 60) / 30.0])
    nose = np.array([10.0, 20.0])                             # never moves: nothing dropped

    class FakeROI:
        def __init__(self, model_path):
            self.i = 0

        def __call__(self, frame, t_ms):
            self.i += 1
            return (rgb[self.i - 1], nose, 100.0, None) if self.i <= len(rgb) else None

    monkeypatch.setattr(rppg, "FaceROI", FakeROI)
    monkeypatch.setattr(rppg.time, "monotonic", lambda: next(clock))
    seen = []
    result = rppg.measure_heart_rate(duration_s=20.0, model_path=None,
                                     grab=lambda: "frame",
                                     on_update=lambda b, s, p: seen.append(p))
    assert abs(result["bpm"] - 69) < 3.0
    assert result["confident"] and result["n_estimates"] >= 3
    assert seen and 0.0 < seen[-1] <= 1.0                     # progress reached the caller


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
    monkeypatch.setattr(rppg, "analyze", lambda *_args, **_kwargs: {"bpm": 72.0, "snr_db": -10.0})
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
