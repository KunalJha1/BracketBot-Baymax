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
    roi = rppg.FaceROI(detector=detector)
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


def test_fundamental_steps_down_on_power_share_when_the_noise_floor_is_high():
    """A weak scan lifts the floor; the half-peak must not need to clear it to be believed."""
    freqs = np.linspace(0.0, 5.0, 2001)
    psd = np.full_like(freqs, 1.0)
    peak = int(np.argmin(np.abs(freqs - 2.2)))                # 132 BPM: the harmonic
    half = int(np.argmin(np.abs(freqs - 1.1)))                # 66 BPM: the real rate
    psd[peak] = 7.0
    psd[half] = 4.8                                           # 69% of the peak, only 6.8 dB up

    assert psd[half] < psd[peak] * 10 ** (rppg.SUB_HARMONIC_FLOOR_DB / 10)
    assert rppg.fundamental(freqs, psd, peak) == half


def test_fundamental_keeps_a_high_rate_when_the_loud_line_is_not_quite_half():
    """An artefact near half the rate is not a subharmonic, however loud it is."""
    freqs = np.linspace(0.0, 5.0, 2001)
    psd = np.full_like(freqs, 1.0)
    peak = int(np.argmin(np.abs(freqs - 2.05)))               # 123 BPM: the real rate
    near_half = int(np.argmin(np.abs(freqs - 0.96)))          # 57.6 BPM, not 61.5
    psd[peak] = 7.0
    psd[near_half] = 4.6

    assert rppg.fundamental(freqs, psd, peak) == peak


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
        self.last = None                      # FaceROI records what it used; so must a stand-in

    def __call__(self, _frame, t_ms):
        self.calls += 1
        if self.miss_every and self.calls % self.miss_every == 0:
            self.last = None
            return None
        t = t_ms / 1000
        pulse = np.sin(2 * np.pi * self.bpm / 60 * t)
        base = np.array([180.0, 130.0, 110.0])
        weights = np.array([0.3, 1.0, 0.5])
        rgb = base * (1 + 0.004 * pulse * weights)
        self.last = {"pixels": 5000, "skin_frac": 0.9, "regions": {}}
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


# ---------------------------------------------------------------------------
# Stacking the whole scan's spectrum
# ---------------------------------------------------------------------------

def test_stacked_spectrum_finds_a_pulse_no_single_window_could():
    """The point of stacking: a rate too weak to win any one window still wins the average."""
    t, rgb = synthetic_ppg(66, seconds=26.0, amplitude=0.0012, seed=11)
    rgb = rgb + np.random.default_rng(3).normal(0, 0.15, rgb.shape)
    tracker = run_tracker(t, rgb)

    window_peaks = [b for _, b, _ in tracker.estimates]
    result = tracker.result()

    assert result is not None, "the stack should rescue a scan the windows could not"
    assert abs(result["bpm"] - 66) < 4.0
    assert result["n_windows"] > len(window_peaks), "windows below the gate must still stack"


def test_stack_normalises_windows_so_one_loud_window_cannot_dominate():
    grid = rppg.scan_grid()
    quiet = np.zeros_like(grid)
    quiet[np.argmin(np.abs(grid - 1.2))] = 1.0            # 72 BPM, faint
    loud = np.zeros_like(grid)
    loud[np.argmin(np.abs(grid - 2.4))] = 1e6             # 144 BPM, enormous

    stack = rppg.SpectrumStack()
    for _ in range(4):
        stack.add(grid, quiet, 3.0)
    stack.add(grid, loud, 3.0)

    assert abs(stack.peak()["bpm"] - 72) < 2.0


def test_stack_weights_windows_the_chain_was_sure_about():
    grid = rppg.scan_grid()
    sure = np.full_like(grid, 0.01)
    sure[np.argmin(np.abs(grid - 1.0))] = 1.0             # 60 BPM at high SNR
    unsure = np.full_like(grid, 0.01)
    unsure[np.argmin(np.abs(grid - 2.0))] = 1.0           # 120 BPM at low SNR

    stack = rppg.SpectrumStack()
    stack.add(grid, sure, 12.0)
    for _ in range(3):
        stack.add(grid, unsure, -6.0)

    assert abs(stack.peak()["bpm"] - 60) < 2.0


def test_reported_rate_must_beat_the_next_best_line_in_the_band():
    """A spectrum with two comparable lines picks its winner by chance; refuse it."""
    grid = rppg.scan_grid()
    psd = np.full_like(grid, 0.05)
    psd[np.argmin(np.abs(grid - 1.1))] = 1.00             # 66 BPM
    psd[np.argmin(np.abs(grid - 1.9))] = 0.98             # 114 BPM, a hair behind

    stack = rppg.SpectrumStack()
    stack.add(grid, psd, 4.0)

    assert stack.peak()["margin_db"] < rppg.AGG_MARGIN_MIN_DB

    tracker = rppg.HeartRateTracker()
    tracker.stack = stack
    assert tracker.result() is None


def test_result_falls_back_to_the_windows_when_nothing_was_stacked():
    """Estimates set directly, with no spectra behind them, still aggregate."""
    tracker = rppg.HeartRateTracker()
    tracker.estimates = [(8.0, 61.0, 4.0), (14.0, 61.4, 4.2), (20.0, 61.2, 4.1)]

    result = tracker.result()

    assert result["n_windows"] == 0
    assert abs(result["bpm"] - 61.2) < 0.5
    assert result["confident"] is True


def test_clear_keeps_the_stack_because_it_describes_the_scan_not_the_window():
    t, rgb = synthetic_ppg(70, seconds=16.0, seed=4)
    tracker = run_tracker(t, rgb)
    stacked = tracker.stack.n_windows

    tracker.clear(unlock=True)

    assert tracker.stack.n_windows == stacked
    assert not tracker.t


# ---------------------------------------------------------------------------
# ROI stage: which pixels get averaged
# ---------------------------------------------------------------------------

def test_skin_gate_drops_hair_and_highlights_but_keeps_the_face():
    import cv2

    frame = np.zeros((120, 120, 3), np.uint8)
    frame[:, :] = (112, 128, 175)                         # BGR skin
    frame[20:40, 20:24] = (28, 30, 38)                    # a dark strand
    cv2.circle(frame, (80, 80), 6, (250, 250, 252), -1)   # a blown highlight
    mask = np.zeros((120, 120), np.uint8)
    mask[10:110, 10:110] = 255

    gated = rppg.skin_pixels(frame, mask)

    assert gated[30, 22] == 0, "hair should not be averaged"
    assert gated[80, 80] == 0, "a specular highlight should not be averaged"
    assert gated[60, 30] == 255, "ordinary skin must survive"
    assert cv2.countNonZero(gated) > 0.8 * cv2.countNonZero(mask)


def test_skin_gate_keeps_the_mask_rather_than_return_too_few_pixels():
    """Better a diluted average than none: the caller needs a number every frame."""
    frame = np.random.default_rng(0).integers(0, 255, (80, 80, 3), dtype=np.uint8)
    mask = np.zeros((80, 80), np.uint8)
    mask[30:50, 30:50] = 255                              # 400 px of pure noise

    assert rppg.skin_pixels(frame, mask) is mask


def test_skin_gate_is_relative_so_it_does_not_assume_one_skin_tone():
    """The gate must behave the same on dark skin as on light: it compares a face to itself."""
    import cv2

    kept = []
    for skin in ((112, 128, 175), (38, 46, 66)):          # light and dark, same hue family
        frame = np.zeros((120, 120, 3), np.uint8)
        frame[:, :] = skin
        frame[20:40, 20:24] = (250, 250, 252)             # the same contaminant in both
        mask = np.zeros((120, 120), np.uint8)
        mask[10:110, 10:110] = 255
        kept.append(cv2.countNonZero(rppg.skin_pixels(frame, mask)))

    assert kept[0] == kept[1]


def test_steady_box_smooths_jitter_but_follows_a_real_move():
    previous = np.array([100.0, 100.0, 80.0, 100.0] + [0.0] * 10 + [0.9])
    jittered = previous.copy()
    jittered[0] += 2.0                                    # detector noise

    smoothed = rppg.steady_box(previous, jittered)
    assert 100.0 < smoothed[0] < 102.0, "jitter should be damped, not ignored"

    moved = previous.copy()
    moved[0] += 60.0                                      # the person stepped sideways
    assert rppg.steady_box(previous, moved)[0] == 160.0, "a real move must be taken whole"


def test_face_roi_records_what_it_averaged_for_the_debug_view():
    face = np.array(
        [[10, 20, 100, 120, 40, 60, 80, 60, 60, 80, 45, 110, 75, 110, 0.99]],
        dtype=np.float32,
    )

    class Detector:
        def setInputSize(self, size):
            pass

        def detect(self, _frame):
            return None, face

    roi = rppg.FaceROI(detector=Detector())
    frame = np.full((240, 320, 3), 120, np.uint8)

    assert roi(frame, 0) is not None
    detail = roi.last
    assert detail["pixels"] > 0 and 0 < detail["skin_frac"] <= 1.0
    assert detail["mask"].shape == frame.shape[:2]
    assert set(detail["regions"]) == set(rppg.REGION_NAMES)
    assert sum(part["pixels"] for part in detail["regions"].values()) == detail["pixels"]
    assert len(detail["landmarks"]) == 5


def test_measure_heart_rate_reports_every_frame_to_a_debug_view(monkeypatch):
    monkeypatch.setattr(rppg, "FaceROI", lambda _model: SyntheticFaceROI(miss_every=7))
    monkeypatch.setattr(rppg.time, "monotonic", SteppingClock())
    seen = []

    rppg.measure_heart_rate(
        duration_s=6.0,
        fs=30.0,
        on_frame=lambda _frame, _roi, _tracker, status: seen.append(status),
        grab=lambda: np.zeros((4, 4, 3), np.uint8),
    )

    assert seen, "the view must be fed while the scan runs"
    assert "measuring" in seen and "no-face" in seen


# ---------------------------------------------------------------------------
# The debug view the robot serves
# ---------------------------------------------------------------------------

def _tracker_with_a_reading():
    t, rgb = synthetic_ppg(72, seconds=16.0, seed=2)
    return run_tracker(t, rgb)


def test_view_draws_kept_and_rejected_pixels_differently():
    import rppg_view

    frame = np.full((120, 160, 3), 90, np.uint8)
    offered = np.zeros((120, 160), np.uint8)
    offered[30:80, 40:120] = 255
    kept = np.zeros((120, 160), np.uint8)
    kept[30:80, 40:100] = 255                             # the last 20 columns were rejected
    detail = {"box": [40, 30, 80, 50], "steady_box": [41, 31, 80, 50], "score": 0.9,
              "landmarks": [[60, 50]], "mask": kept, "region_mask": offered,
              "pixels": int(kept.sum() // 255), "skin_frac": 0.75, "regions": {}}

    out = rppg_view.draw_overlay(frame, detail)

    assert not np.array_equal(out[50, 60], out[50, 110]), "kept and rejected must differ"
    assert not np.array_equal(out[50, 60], frame[50, 60]), "kept skin must be tinted"
    assert np.array_equal(out[10, 10], frame[10, 10]), "untouched background stays put"


def test_view_panel_reports_the_scan_not_only_the_window():
    import rppg_view

    tracker = _tracker_with_a_reading()
    state = rppg_view.state_from(tracker, frame_fps=27.0, status="measuring")

    assert state["scan_snr"] == tracker.result()["snr_db"]
    assert state["snr"] == tracker.last_snr
    assert state["n_windows"] == tracker.stack.n_windows
    assert state["stack_psd"] is not None

    picture = rppg_view.compose(np.full((240, 320, 3), 40, np.uint8), state)
    assert picture.shape[0] >= 620 and picture.shape[2] == 3


def test_view_panel_survives_a_scan_that_has_produced_nothing_yet():
    import rppg_view

    tracker = rppg.HeartRateTracker()
    state = rppg_view.state_from(tracker, status="buffering")

    assert state["bpm_disp"] is None and state["stack_psd"] is None
    rppg_view.compose(None, state)                        # must not raise


def test_debug_view_serves_the_page_the_stream_and_the_numbers():
    import urllib.error
    import urllib.request

    tracker = _tracker_with_a_reading()
    frame = np.full((480, 640, 3), 70, np.uint8)
    view = robot_rppg.DebugView(port=0, fps=0.0)
    with view:
        port = view.server.server_address[1]
        view.publish(frame, None, tracker, "measuring", fps=26.0)
        base = f"http://127.0.0.1:{port}"

        page = urllib.request.urlopen(f"{base}/", timeout=5).read().decode()
        state = json.loads(urllib.request.urlopen(f"{base}/state.json", timeout=5).read())
        jpeg = urllib.request.urlopen(f"{base}/frame.jpg", timeout=5).read()
        try:
            urllib.request.urlopen(f"{base}/nope", timeout=5)
            missing_status = 200
        except urllib.error.HTTPError as exc:
            missing_status = exc.code

    assert "stream.mjpg" in page and "not a medical measurement" in page
    assert state["bpm"] == round(tracker.result()["bpm"], 1)
    assert state["windows"] == tracker.stack.n_windows and state["fps"] == 26.0
    assert jpeg.startswith(b"\xff\xd8"), "an actual JPEG"
    assert missing_status == 404


def test_debug_view_scales_the_overlay_onto_the_frame_it_draws():
    mask = np.zeros((960, 1280), np.uint8)
    mask[400:500, 600:700] = 255
    detail = {"box": [600.0, 400.0, 100.0, 130.0], "steady_box": [602.0, 402.0, 100.0, 130.0],
              "landmarks": [[640.0, 440.0]], "mask": mask, "region_mask": mask, "labels": None,
              "pixels": 10000, "skin_frac": 0.9, "regions": {}}

    scaled = robot_rppg.scale_detail(detail, 0.25)

    assert scaled["box"] == [150.0, 100.0, 25.0, 32.5]
    assert scaled["landmarks"] == [[160.0, 110.0]]
    assert scaled["mask"].shape == (240, 320)
    assert scaled["pixels"] == 10000, "the pixel count is the camera's, not the view's"
    assert detail["mask"].shape == (960, 1280), "the original must not be mutated"


def test_confidence_needs_the_stack_and_the_windows_to_agree_with_each_other():
    """Two readings of one scan from different evidence; disagreement is not confidence."""
    grid = rppg.scan_grid()
    # A clean line at 120 BPM: narrow, well clear of a low floor, so it passes
    # both aggregate gates and the confidence question is the only one left.
    psd = np.full_like(grid, 0.0005)
    psd += np.exp(-0.5 * ((grid - 2.0) / 0.01) ** 2)

    tracker = rppg.HeartRateTracker()
    for _ in range(4):
        tracker.stack.add(grid, psd, 8.0)
    # ...while every accepted window said 60, tightly and over a long span.
    tracker.estimates = [(6.0, 60.0, 6.0), (12.0, 60.2, 6.1), (18.0, 60.1, 6.0)]

    result = tracker.result()

    assert abs(result["bpm"] - 120) < 2 and result["window_bpm"] == 60.1
    assert result["confident"] is False


# ----------------------------------------------------------------------------
# Face-metered exposure
# ----------------------------------------------------------------------------

class FakeLight:
    """A camera whose skin brightness is whatever the exposure control has asked for."""

    def __init__(self, luma=0.15, clip_hi=0.0, clip_lo=0.0):
        self.luma, self.clip_hi, self.clip_lo = luma, clip_hi, clip_lo
        self.factors = []

    def apply(self, factor):
        self.factors.append(factor)
        self.luma = min(1.0, self.luma * factor)
        return {"exposure": self.luma, "gain": None, "note": "fake"}

    def stats(self):
        return {"luma": self.luma, "clip_hi": self.clip_hi, "clip_lo": self.clip_lo,
                "pixels": 5000}


def meter_for(seconds, light, meter=None, fps=30.0, t0=0.0):
    """Run the loop over a stretch of frames the way a live view does."""
    meter = meter or rppg.FaceExposure()
    for i in range(int(seconds * fps)):
        factor = meter.observe(t0 + i / fps, light.stats())
        if factor is not None:
            light.apply(factor)
    return meter


def test_skin_luma_stats_meters_the_skin_and_not_the_frame_around_it():
    """The background is what a camera's own AE would meter; the point is not to."""
    frame = np.full((60, 60, 3), 250, np.uint8)               # a bright wall behind the face
    frame[10:50, 10:50] = 60                                  # ...and a face left in the dark
    mask = np.zeros((60, 60), np.uint8)
    mask[10:50, 10:50] = 255

    stats = rppg.skin_luma_stats(frame, mask)
    assert abs(stats["luma"] - 60 / 255) < 0.01               # the face, not the 250 around it
    assert stats["pixels"] == 1600
    assert stats["clip_hi"] == 0.0 and stats["clip_lo"] == 0.0
    assert rppg.skin_luma_stats(frame, None) is None          # no face: nothing to meter


def test_skin_luma_stats_counts_the_skin_the_sensor_threw_away():
    frame = np.full((60, 60, 3), 120, np.uint8)
    frame[10:20, 10:50] = 255                                 # a blown highlight on the forehead
    frame[40:50, 10:50] = 2                                   # ...and a cheek in deep shadow
    mask = np.zeros((60, 60), np.uint8)
    mask[10:50, 10:50] = 255

    stats = rppg.skin_luma_stats(frame, mask)
    assert abs(stats["clip_hi"] - 0.25) < 0.01
    assert abs(stats["clip_lo"] - 0.25) < 0.01


def test_face_exposure_walks_a_dark_face_into_range_and_then_leaves_it_alone():
    """A face at 15% of the sensor's range carries its pulse in a couple of counts."""
    light = FakeLight(luma=0.15)
    meter = meter_for(6.0, light)

    assert abs(light.luma - rppg.EXPOSURE_TARGET) <= rppg.EXPOSURE_DEAD_BAND
    assert meter.settled and meter.changes <= 5                # it converges, it does not hunt
    before = meter.changes
    meter_for(10.0, light, meter, t0=6.0)
    assert meter.changes == before                             # ...and then stops touching it


def test_face_exposure_pulls_a_blown_face_down_even_when_the_mean_looks_right():
    """Clipped skin has had its modulation removed by the sensor: no filter gets it back."""
    light = FakeLight(luma=rppg.EXPOSURE_TARGET, clip_hi=0.10)
    meter = rppg.FaceExposure()
    move = next(f for f in (meter.observe(i / 30.0, light.stats()) for i in range(10))
                if f is not None)

    assert move < 1.0                              # the mean says stay, the highlights say down
    assert meter.changes == 1 and "clipping" in meter.reason


def test_face_exposure_waits_for_a_change_to_reach_the_pixels():
    """Asking again before the camera has applied the last request is how AE hunts."""
    meter = rppg.FaceExposure()
    light = FakeLight(luma=0.15)
    factor = None
    for i in range(10):                                        # first move, once the face is steady
        factor = meter.observe(i / 30.0, light.stats())
        if factor is not None:
            break
    assert factor is not None
    # The brightness has not moved yet - the camera applies the change a few frames later.
    assert all(meter.observe((10 + i) / 30.0, light.stats()) is None for i in range(10))


def test_face_exposure_ignores_a_blink_of_a_face_and_forgets_it_when_it_leaves():
    meter = rppg.FaceExposure()
    light = FakeLight(luma=0.15)
    assert meter.observe(0.0, light.stats()) is None           # min_frames not reached yet
    assert meter.observe(0.03, None) is None                   # face gone: the count restarts
    assert meter.state == "no face"
    assert meter.observe(0.06, light.stats()) is None
    assert meter.changes == 0


def test_face_exposure_reopens_on_a_lamp_but_not_on_the_wobble_of_a_lit_face():
    light = FakeLight(luma=0.15)
    meter = meter_for(6.0, light)
    assert meter.settled

    light.luma = rppg.EXPOSURE_TARGET + 0.08                   # inside the re-arm band
    settled = meter.changes
    meter_for(3.0, light, meter, t0=6.0)
    assert meter.changes == settled and meter.settled

    light.luma = rppg.EXPOSURE_TARGET + 0.25                   # someone switched a lamp on
    meter_for(4.0, light, meter, t0=9.0)
    assert meter.changes > settled
    assert abs(light.luma - rppg.EXPOSURE_TARGET) <= rppg.EXPOSURE_DEAD_BAND


class FakeCap:
    """The OpenCV property API, with whatever units the backend under test pretends to use."""

    def __init__(self, exposure=-7.0, gain=8.0, settable=True):
        import cv2

        self.props = {cv2.CAP_PROP_EXPOSURE: exposure, cv2.CAP_PROP_GAIN: gain}
        self.settable = settable

    def get(self, prop):
        return self.props.get(prop, 0.0)

    def set(self, prop, value):
        if not self.settable:
            return False
        self.props[prop] = value
        return True


def test_opencv_exposure_moves_in_whole_stops_where_the_units_are_stops():
    """DirectShow reports log2 seconds: -7 is 1/128 s, and one count is a full stop."""
    cap = FakeCap(exposure=-7.0)
    control = rppg.OpenCVExposure(cap, platform="win32")

    assert control.apply(2.0)["exposure"] == -6.0
    assert control.apply(0.5)["exposure"] == -7.0
    # A 10% request is smaller than the smallest step these units have, so it goes
    # to gain instead of becoming a whole stop of overshoot.
    result = control.apply(1.1)
    assert result["exposure"] == -7.0 and result["gain"] == 9.0


def test_opencv_exposure_scales_the_number_where_the_units_are_time():
    """V4L2 reports exposure_time_absolute in 100 us units, which scales with the light."""
    cap = FakeCap(exposure=100.0)
    control = rppg.OpenCVExposure(cap, platform="linux")

    assert control.apply(1.5)["exposure"] == 150.0
    assert control.apply(0.5)["exposure"] == 75.0


def test_opencv_exposure_buys_the_rest_with_gain_once_exposure_is_railed():
    """A noisy face still has a pulse in it; a face pinned at the bottom of the range does not."""
    cap = FakeCap(exposure=rppg.OpenCVExposure.ABS_RANGE[1], gain=8.0)
    control = rppg.OpenCVExposure(cap, platform="linux")

    result = control.apply(2.0)
    assert result["exposure"] == rppg.OpenCVExposure.ABS_RANGE[1]
    assert result["gain"] == 16.0
    assert control.applied == 1


def test_opencv_exposure_reports_a_camera_that_took_neither():
    control = rppg.OpenCVExposure(FakeCap(exposure=100.0, settable=False), platform="linux")
    assert control.apply(2.0) is None
    assert "ignored" in control.note
    assert rppg.OpenCVExposure(FakeCap(), platform="darwin").apply(2.0) is None


class FakePicam:
    def __init__(self):
        self.controls = []

    def set_controls(self, controls):
        self.controls.append(controls)


def test_picamera2_exposure_only_ever_spans_whole_flicker_cycles():
    """A part-cycle exposure samples the mains ripple instead of averaging it, and the
    beat lands in the pulse band - which is the one place it must not be."""
    cam = FakePicam()
    control = rppg.Picamera2Exposure(cam, exposure_us=8333, gain=4.0, flicker_us=8333, fps=30)

    result = control.apply(2.0)
    assert result["exposure"] % 8333 == 0
    assert abs(result["exposure"] * result["gain"] / (8333 * 4.0) - 2.0) < 0.05
    assert cam.controls[-1]["ExposureTime"] == result["exposure"]

    control.apply(0.25)                                        # down again, still on the grid
    assert control.exposure_us % 8333 == 0
    assert control.gain >= 1.0


def test_picamera2_exposure_prefers_a_longer_exposure_over_more_gain():
    """Gain is noise; exposure is not. Both reach the same brightness."""
    cam = FakePicam()
    control = rppg.Picamera2Exposure(cam, exposure_us=8333, gain=8.0, flicker_us=8333, fps=30)

    control.apply(1.0)                                         # same light, better spent
    assert control.exposure_us > 8333 and control.gain < 8.0


def test_picamera2_exposure_says_so_when_it_has_nothing_left_to_give():
    cam = FakePicam()
    control = rppg.Picamera2Exposure(cam, exposure_us=8333, gain=8.0, flicker_us=8333, fps=120)
    assert control.max_periods == 1                            # a frame holds one cycle at 120 fps
    assert control.apply(4.0) is None                          # already at the top of both
    assert "limit" in control.note


def test_measure_heart_rate_meters_the_face_and_drops_the_window_it_stepped_in(monkeypatch):
    """The scan's own frames drive the loop, and a scan still comes out the other end."""
    t, rgb = synthetic_ppg(72, seconds=26.0, seed=3)
    clock = iter(np.r_[t, t[-1] + np.arange(1, 120) / 30.0])
    nose = np.array([10.0, 20.0])
    mask = np.full((40, 40), 255, np.uint8)

    class FakeControls:
        def __init__(self):
            self.light, self.factors = 0.30, []               # a face at 30% of the range

        def apply(self, factor):
            self.factors.append(factor)
            self.light = min(1.0, self.light * factor)
            return {"exposure": self.light, "gain": None, "note": "fake"}

    class FakeCamera:
        def __init__(self):
            self.controls = FakeControls()

        def __call__(self):
            level = int(round(255 * self.controls.light))
            return np.full((40, 40, 3), level, np.uint8)

    class FakeROI:
        def __init__(self, model_path):
            self.i = 0

        def __call__(self, frame, t_ms):
            self.i += 1
            return (rgb[self.i - 1], nose, 100.0, mask) if self.i <= len(rgb) else None

    monkeypatch.setattr(rppg, "FaceROI", FakeROI)
    monkeypatch.setattr(rppg.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(rppg, "MIN_SKIN_PIXELS", 400)
    camera = FakeCamera()
    moves = []
    result = rppg.measure_heart_rate(duration_s=24.0, model_path=None, grab=camera,
                                     on_exposure=lambda f, m, r: moves.append(f))

    assert moves and all(f > 1.0 for f in moves)               # a dark face was brightened
    assert abs(camera.controls.light - rppg.EXPOSURE_TARGET) <= rppg.EXPOSURE_DEAD_BAND
    assert len(moves) <= 4                                     # and each move costs a window
    assert result is not None and abs(result["bpm"] - 72) < 3.0


def test_measure_heart_rate_leaves_a_camera_it_cannot_drive_alone(monkeypatch):
    """The robot reads a shared camera topic: there is nothing there to retune, and
    asking for a scan must not start trying."""
    t, rgb = synthetic_ppg(69, seconds=12.0, seed=1)
    clock = iter(np.r_[t, t[-1] + np.arange(1, 60) / 30.0])

    class FakeROI:
        def __init__(self, model_path):
            self.i = 0

        def __call__(self, frame, t_ms):
            self.i += 1
            return (rgb[self.i - 1], np.array([10.0, 20.0]), 100.0, None) if self.i <= len(rgb) \
                else None

    monkeypatch.setattr(rppg, "FaceROI", FakeROI)
    monkeypatch.setattr(rppg.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(rppg, "skin_luma_stats",
                        lambda *a, **k: pytest.fail("metered a camera it cannot drive"))
    rppg.measure_heart_rate(duration_s=10.0, model_path=None, grab=lambda: "frame")


def test_view_shows_how_much_of_the_skin_the_sensor_clipped():
    import rppg_view

    tracker = rppg.HeartRateTracker()
    panel = rppg_view.build_panel(620, rppg_view.state_from(tracker, clip_hi=0.07, clip_lo=0.0,
                                                            note="skin 52%  settled, 2 moves"))
    assert panel.shape == (620, rppg_view.PANEL_W, 3)
    assert panel.any()
