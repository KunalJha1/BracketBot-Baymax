"""
rppg.py - contactless heart rate for BracketBot / "Baymax" scan.

Pipeline:
  camera (locked exposure/WB) -> OpenCV YuNet face detector -> forehead + cheek ROI mask
  -> per-frame mean RGB + timestamp -> longest gap-free stretch -> resample to uniform fs
  -> POS projection -> detrend + Butterworth bandpass -> Hann-windowed zero-padded FFT
  -> peak, second-harmonic check (parabolic interp) + SNR gate + tracking
  -> SNR-weighted aggregation -> BPM, confidence

Deps: numpy, scipy, opencv-python
Model: assets/models/face_detection_yunet_2026may.onnx (source/checksum in assets/models/README.md)

NOT a medical device. Demo accuracy only.
"""
import time
from collections import deque
from pathlib import Path

import numpy as np
from scipy.signal import butter, filtfilt, detrend

# ----------------------------------------------------------------------------
# Signal processing (pure numpy/scipy - testable without a camera)
# ----------------------------------------------------------------------------

HR_LO_HZ, HR_HI_HZ = 0.7, 3.0          # 42-180 BPM
MIN_SECONDS = 6.0                      # shortest window that resolves a rate at all
MAX_GAP_S = 0.5                        # longer hole in the samples: analyse around it
SUB_HARMONIC_RATIO = 0.2               # half-peak power needed to call the peak a harmonic
SUB_HARMONIC_FLOOR_DB = 9.0            # ...and how far it must clear the in-band noise floor
DEFAULT_MODEL = (
    Path(__file__).resolve().parent / "assets" / "models" / "face_detection_yunet_2026may.onnx"
)


def resample_uniform(t, rgb, fs):
    """Pi frame timing jitters; FFT assumes uniform sampling. Linear-interp each channel."""
    t = np.asarray(t, dtype=float)
    rgb = np.asarray(rgb, dtype=float)
    tu = np.arange(t[0], t[-1], 1.0 / fs)
    out = np.column_stack([np.interp(tu, t, rgb[:, c]) for c in range(3)])
    return tu, out


def pos(rgb, fs, win_s=1.6):
    """
    Plane-Orthogonal-to-Skin (Wang et al., IEEE TBME 2017).
    rgb: (N, 3) mean skin RGB. Returns pulse signal H (N,).
    Temporal normalization per short window cancels illumination intensity
    changes; projection onto plane orthogonal to skin tone suppresses specular/motion.
    Each window is standardized before the overlap-add (eq. 7 in the paper), so a
    window that caught a motion jerk or an exposure step cannot shout down the
    clean ones; the running count divides out the ramp at the two ends.
    """
    n_total = len(rgb)
    l = int(round(win_s * fs))
    if l < 2 or n_total < l:
        return np.zeros(n_total)
    H = np.zeros(n_total)
    overlaps = np.zeros(n_total)
    P = np.array([[0.0, 1.0, -1.0],
                  [-2.0, 1.0, 1.0]])
    for n in range(l, n_total + 1):
        C = rgb[n - l:n].T                              # 3 x l
        mu = C.mean(axis=1, keepdims=True)
        if np.any(mu <= 0):
            continue
        S = P @ (C / mu)                                # 2 x l
        alpha = S[0].std() / (S[1].std() + 1e-9)
        h = S[0] + alpha * S[1]
        sd = h.std()
        if sd < 1e-12:
            continue
        H[n - l:n] += (h - h.mean()) / sd               # overlap-add, unit variance
        overlaps[n - l:n] += 1
    return H / np.maximum(overlaps, 1)


def green_only(rgb):
    """Baseline for comparison. Worse under lighting changes / motion."""
    g = rgb[:, 1]
    return g / g.mean() - 1.0


def bandpass(x, fs, lo=HR_LO_HZ, hi=HR_HI_HZ, order=3):
    b, a = butter(order, [lo / (fs / 2), hi / (fs / 2)], btype="band")
    return filtfilt(b, a, detrend(x))


def spectrum(x, fs, pad=8):
    n = len(x)
    nfft = 1 << int(np.ceil(np.log2(n * pad)))
    X = np.fft.rfft(x * np.hanning(n), nfft)
    return np.fft.rfftfreq(nfft, 1.0 / fs), np.abs(X) ** 2


def snr_db(freqs, psd, f0, hi_cap):
    """de Haan & Jeanne (2013) style: power at f0 and 2*f0 vs everything else in band."""
    band = (freqs >= HR_LO_HZ) & (freqs <= hi_cap)
    sig = band & ((np.abs(freqs - f0) <= 0.1) | (np.abs(freqs - 2 * f0) <= 0.2))
    noise = band & ~sig
    return 10 * np.log10(psd[sig].sum() / (psd[noise].sum() + 1e-12) + 1e-12)


def longest_clean_span(t, max_gap=MAX_GAP_S, min_seconds=MIN_SECONDS):
    """Slice of the longest stretch of t with no hole longer than max_gap.

    Frames dropped for motion or a lost face leave holes, and resample_uniform
    fills a hole with a straight line - a fake low-frequency swing sitting right
    where the pulse lives. Analysing only the longest clean stretch avoids that,
    but is worth doing only while that stretch is still long enough to estimate
    from; otherwise the whole span is kept and the SNR gate does the rejecting.
    """
    t = np.asarray(t, dtype=float)
    if len(t) < 2:
        return slice(0, len(t))
    breaks = np.flatnonzero(np.diff(t) > max_gap) + 1
    starts, stops = np.r_[0, breaks], np.r_[breaks, len(t)]
    spans = t[stops - 1] - t[starts]
    i = int(np.argmax(spans))
    if spans[i] < min_seconds or stops[i] - starts[i] < 2:
        return slice(0, len(t))
    return slice(int(starts[i]), int(stops[i]))


def _sub_bin_peak(freqs, psd, k):
    """Parabolic interpolation on log power, for a peak between two bins."""
    if 0 < k < len(psd) - 1:
        a, b, c = np.log(psd[k - 1:k + 2] + 1e-20)
        d = 0.5 * (a - c) / (a - 2 * b + c + 1e-20)
        return freqs[k] + d * (freqs[1] - freqs[0])
    return freqs[k]


def fundamental(freqs, psd, k, ratio=SUB_HARMONIC_RATIO, floor_db=SUB_HARMONIC_FLOOR_DB):
    """Bin of the pulse fundamental, given the strongest in-band bin k.

    A camera sees the PPG waveform, not a sine, and its second harmonic is often
    the taller line: the raw peak then reports exactly twice the real rate, and
    reports it consistently enough to look confident. If half the peak frequency
    is still a plausible heart rate and carries a peak of its own - both a fair
    share of the main peak and well clear of the in-band noise floor - then that
    half is the fundamental and the tall line is its harmonic.

    The trade is deliberate: a genuine rate near the top of the band whose half
    happens to collide with a strong artefact can be halved. That costs an
    occasional reading in a range this demo barely serves; harmonic lock-in was
    costing a systematic 2x on ordinary resting rates.
    """
    f_half = freqs[k] / 2.0
    if f_half < HR_LO_HZ:
        return k
    band = (freqs >= HR_LO_HZ) & (freqs <= HR_HI_HZ)
    floor = float(np.median(psd[band])) if band.any() else 0.0
    near = np.flatnonzero(np.abs(freqs - f_half) <= 0.1)
    if not len(near):
        return k
    j = int(near[np.argmax(psd[near])])
    own_peak = psd[j] >= floor * 10 ** (floor_db / 10)
    return j if own_peak and psd[j] >= ratio * psd[k] else k


def analyze(t, rgb, fs=30.0, method="pos", prev_bpm=None, track_bpm=15.0):
    """
    Full chain, returning intermediates for debugging/plotting.
    t: timestamps (s), rgb: (N,3) per-frame ROI means.
    prev_bpm: if given, peak search is restricted to prev +/- track_bpm
              (HR can't jump 40 BPM in a second; kills motion jumps). The
              second-harmonic check still runs over the whole band, so a tracker
              that locked onto 2x can climb back down to the real rate.
    Returns dict(bpm, snr_db, h, freqs, psd, rgb_u) or None if < 6 s of data.
    """
    t = np.asarray(t, dtype=float)
    rgb = np.asarray(rgb, dtype=float)
    if len(t) < 2 or (t[-1] - t[0]) < MIN_SECONDS:
        return None
    clean = longest_clean_span(t)
    t, rgb = t[clean], rgb[clean]
    if len(t) < 2 or (t[-1] - t[0]) < MIN_SECONDS:
        return None
    _, x = resample_uniform(t, rgb, fs)
    h = pos(x, fs) if method == "pos" else green_only(x)
    h = bandpass(h, fs)
    freqs, psd = spectrum(h, fs)

    lo, hi = HR_LO_HZ, HR_HI_HZ
    if prev_bpm is not None:
        lo = max(lo, (prev_bpm - track_bpm) / 60)
        hi = min(hi, (prev_bpm + track_bpm) / 60)
    idx = np.where((freqs >= lo) & (freqs <= hi))[0]
    if not len(idx):
        idx = np.where((freqs >= HR_LO_HZ) & (freqs <= HR_HI_HZ))[0]
    k = fundamental(freqs, psd, int(idx[np.argmax(psd[idx])]))
    f0 = _sub_bin_peak(freqs, psd, k)

    return {"bpm": 60.0 * f0,
            "snr_db": snr_db(freqs, psd, f0, min(2 * HR_HI_HZ + 0.3, fs / 2)),
            "h": h, "freqs": freqs, "psd": psd, "rgb_u": x}


def estimate_hr(t, rgb, fs=30.0, method="pos", prev_bpm=None, track_bpm=15.0):
    """Returns (bpm, snr_db) or (None, None) if not enough data."""
    r = analyze(t, rgb, fs, method, prev_bpm, track_bpm)
    return (None, None) if r is None else (r["bpm"], r["snr_db"])


# ----------------------------------------------------------------------------
# ROI extraction (OpenCV YuNet face box + five landmarks)
# ----------------------------------------------------------------------------

FACE_SCORE = 0.75
FACE_DETECT_MAX_WIDTH = 560


def face_skin_mask(face, image_shape):
    """Return a conservative forehead/cheek mask from one YuNet detection.

    YuNet rows are ``x, y, w, h``, five landmark pairs (eyes, nose, mouth
    corners), then confidence. Fixed face-relative polygons avoid the eyes,
    nose, mouth, hairline, and background. This is less anatomically precise
    than a 478-point mesh, but rPPG only needs stable skin averages and YuNet
    has supported ARM64 wheels through OpenCV, unlike the old MediaPipe stack.
    """
    import cv2

    h_img, w_img = image_shape[:2]
    values = np.asarray(face, dtype=float).reshape(-1)
    if len(values) < 15 or not np.isfinite(values[:15]).all():
        return None
    x, y, w, h = values[:4]
    if w <= 0 or h <= 0:
        return None

    def point(rx, ry):
        px = int(np.clip(round(x + rx * w), 0, max(0, w_img - 1)))
        py = int(np.clip(round(y + ry * h), 0, max(0, h_img - 1)))
        return px, py

    regions = (
        # Forehead: below the hairline and above the eyebrows.
        (point(0.24, 0.10), point(0.76, 0.10), point(0.69, 0.34), point(0.31, 0.34)),
        # Camera-left and camera-right cheeks, clear of nose and mouth.
        (point(0.10, 0.46), point(0.38, 0.42), point(0.40, 0.72), point(0.16, 0.76)),
        (point(0.62, 0.42), point(0.90, 0.46), point(0.84, 0.76), point(0.60, 0.72)),
    )
    mask = np.zeros((h_img, w_img), np.uint8)
    for region in regions:
        cv2.fillConvexPoly(mask, np.asarray(region, dtype=np.int32), 255)
    return mask


class FaceROI:
    def __init__(self, model_path=DEFAULT_MODEL):
        import cv2

        self.cv2 = cv2
        self.detector = cv2.FaceDetectorYN.create(
            str(model_path), "", (320, 320), FACE_SCORE, 0.3, 50
        )
        self.detector_size = None

    def __call__(self, frame_bgr, _t_ms):
        """Returns (mean_rgb (3,), nose_xy, face_width_px, mask) or None if no face."""
        h, w = frame_bgr.shape[:2]
        scale = min(1.0, FACE_DETECT_MAX_WIDTH / w)
        detect_w = max(1, int(round(w * scale)))
        detect_h = max(1, int(round(h * scale)))
        detect_size = (detect_w, detect_h)
        if self.detector_size != detect_size:
            self.detector.setInputSize(detect_size)
            self.detector_size = detect_size
        detection_frame = (
            frame_bgr if detect_size == (w, h)
            else self.cv2.resize(frame_bgr, detect_size, interpolation=self.cv2.INTER_AREA)
        )
        _, faces = self.detector.detect(detection_frame)
        if faces is None or not len(faces):
            return None
        face = max(faces, key=lambda row: float(row[2] * row[3] * row[-1]))
        if detect_size != (w, h):
            face = np.asarray(face, dtype=np.float32).copy()
            face[[0, 2, 4, 6, 8, 10, 12]] *= w / detect_w
            face[[1, 3, 5, 7, 9, 11, 13]] *= h / detect_h
        mask = face_skin_mask(face, frame_bgr.shape)
        if mask is None or self.cv2.countNonZero(mask) < 400:  # face too small/far
            return None
        rgb = self.cv2.cvtColor(frame_bgr, self.cv2.COLOR_BGR2RGB)
        mean_rgb = self.cv2.mean(rgb, mask=mask)[:3]
        nose = np.asarray(face[8:10], dtype=np.float32)
        return np.asarray(mean_rgb), nose, float(face[2]), mask


# ----------------------------------------------------------------------------
# Camera with locked exposure / white balance
# ----------------------------------------------------------------------------

def open_camera(width=640, height=480, fps=30, exposure_us=8333):
    """
    Auto-exposure / auto-WB / auto-gain MUST be off: their adjustments are
    bigger than the pulse signal (~0.1-1% intensity) and show up as fake peaks.
    exposure_us=8333 = 1/120 s -> integrates exactly one flicker cycle of
    60 Hz mains lighting (use 10000 in 50 Hz countries).
    """
    try:
        from picamera2 import Picamera2
        cam = Picamera2()
        cfg = cam.create_video_configuration(
            main={"size": (width, height), "format": "RGB888"},
            controls={"FrameDurationLimits": (int(1e6 / fps), int(1e6 / fps))})
        cam.configure(cfg)
        cam.start()
        time.sleep(1.5)                                   # let AE/AWB settle once...
        md = cam.capture_metadata()
        cam.set_controls({                                # ...then freeze them
            "AeEnable": False, "AwbEnable": False,
            "ExposureTime": exposure_us,
            "AnalogueGain": md.get("AnalogueGain", 4.0),
            "ColourGains": md.get("ColourGains", (1.8, 1.5)),
        })
        return lambda: cam.capture_array()                # RGB888 is BGR-ordered in memory
    except ImportError:
        import cv2
        cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, fps)
        # UVC: lock via v4l2-ctl beforehand (names vary by camera; `v4l2-ctl -l`):
        #   v4l2-ctl -c auto_exposure=1 -c exposure_time_absolute=83 \
        #            -c white_balance_automatic=0 -c gain=<fixed>
        return lambda: cap.read()[1]


# ----------------------------------------------------------------------------
# Sliding-window tracker (camera-free: the aggregation is testable without hardware)
# ----------------------------------------------------------------------------

def weighted_median(values, weights):
    """Median with weights - one sure window outweighs several marginal ones."""
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    order = np.argsort(v)
    v, w = v[order], w[order]
    total = w.sum()
    if not np.isfinite(total) or total <= 0:
        return float(np.median(v))
    return float(v[np.searchsorted(np.cumsum(w), 0.5 * total)])


class HeartRateTracker:
    """Rolling ROI-mean buffer, one rate estimate per call, and their aggregate.

    Keeps the last window_s seconds of per-frame mean skin RGB, runs the signal
    chain over it on demand, gates each estimate on SNR, and combines the ones
    that pass. Holding the tracking state here is what lets a scan climb out of
    a bad lock: after unlock_after rejected estimates in a row, the peak search
    is freed to look across the whole band again.
    """

    def __init__(self, fs=30.0, window_s=10.0, snr_min_db=-1.0, method="pos",
                 track_bpm=15.0, unlock_after=6, keep_n=8, spread_max=6.0,
                 conf_span_s=6.0):
        self.fs, self.window_s, self.snr_min_db = fs, window_s, snr_min_db
        self.method, self.track_bpm = method, track_bpm
        self.unlock_after, self.keep_n, self.spread_max = unlock_after, keep_n, spread_max
        self.conf_span_s = conf_span_s
        self.t, self.rgb = deque(), deque()
        self.estimates = []                 # (t, bpm, snr_db) that passed the gate
        self.prev = None                    # tracking centre; None = acquire freely
        self.low_run = 0
        self.last = None                    # last analyze() dict, for the debug view

    def add(self, t, rgb):
        self.t.append(float(t))
        self.rgb.append(np.asarray(rgb, dtype=float))
        while self.t and self.t[-1] - self.t[0] > self.window_s:
            self.t.popleft()
            self.rgb.popleft()

    def clear(self, unlock=False):
        """Drop the buffer (face lost, or motion). Accepted estimates survive."""
        self.t.clear()
        self.rgb.clear()
        self.last = None
        if unlock:
            self.prev, self.low_run = None, 0

    def update(self, now=None):
        """Estimate once over the current buffer. Returns the analyze() dict or None."""
        if len(self.t) < 2:
            self.last = None
            return None
        res = analyze(np.array(self.t), np.array(self.rgb), self.fs, self.method,
                      prev_bpm=self.prev, track_bpm=self.track_bpm)
        self.last = res
        if res is None:
            return None
        if res["snr_db"] >= self.snr_min_db:
            self.estimates.append((self.t[-1] if now is None else float(now),
                                   res["bpm"], res["snr_db"]))
            self.prev, self.low_run = res["bpm"], 0
        else:
            self.low_run += 1
            if self.low_run >= self.unlock_after:
                self.prev, self.low_run = None, 0
        return res

    @property
    def last_bpm(self):
        return None if self.last is None else self.last["bpm"]

    @property
    def last_snr(self):
        return None if self.last is None else self.last["snr_db"]

    def result(self):
        """Aggregate of the accepted estimates, or None if none passed the gate.

        SNR-weighted, so a window the chain was sure about counts for more, and
        median-based, so one stray estimate cannot drag the answer. Consecutive
        windows overlap heavily, so agreement between them is weak evidence on
        its own: confidence also asks that the accepted estimates span at least
        conf_span_s, which is what tells a rate that held through the scan from
        a noise peak that happened to win the last few windows.
        """
        if not self.estimates:
            return None
        tail = self.estimates[-self.keep_n:]
        bpms = np.array([b for _, b, _ in tail])
        snrs = np.array([s for _, _, s in tail])
        spread = float(np.ptp(bpms))
        snr_med = float(np.median(snrs))
        span = self.estimates[-1][0] - self.estimates[0][0]
        return {"bpm": weighted_median(bpms, 10 ** (np.clip(snrs, -10.0, 20.0) / 10.0)),
                "snr_db": round(snr_med, 2),
                "spread_bpm": round(spread, 1),
                "span_s": round(span, 1),
                "confident": bool(len(self.estimates) >= 3
                                  and spread <= self.spread_max
                                  and snr_med >= self.snr_min_db + 2.0
                                  and span >= self.conf_span_s),
                "n_estimates": len(self.estimates)}


# ----------------------------------------------------------------------------
# Measurement routine (call this from the robot's tool-calling agent)
# ----------------------------------------------------------------------------

def measure_heart_rate(duration_s=15.0, window_s=10.0, fs=30.0,
                       snr_min_db=-1.0, motion_max=0.03, face_gap_max_s=0.75,
                       model_path=DEFAULT_MODEL, on_update=None, grab=None):
    """
    Blocking scan. Robot should be stationary (motors holding balance only)
    and say "please hold still" first.

    motion_max: max nose displacement per frame as fraction of face width.
                Frames above this are dropped; if too many drop, buffer resets.
    face_gap_max_s: tolerate brief landmark misses without throwing away the
                    several seconds of good signal already collected. Longer
                    losses reset the window and peak tracker.
    on_update(bpm, snr, progress): optional hook for the face display.
    grab: zero-arg callable returning one BGR frame (or None if no new frame yet).
          Defaults to open_camera(); the robot passes its head-camera reader here.
    Returns dict(bpm, snr_db, spread_bpm, span_s, confident, n_estimates) or None.
    """
    grab = grab or open_camera(fps=int(fs))
    roi = FaceROI(model_path)
    tracker = HeartRateTracker(fs=fs, window_s=window_s, snr_min_db=snr_min_db)
    last_nose, last_face_at, bad_run = None, None, 0
    t0 = time.monotonic()
    next_est = t0 + MIN_SECONDS

    while (now := time.monotonic()) - t0 < duration_s:
        frame = grab()
        if frame is None:
            continue
        r = roi(frame, now * 1000)
        if r is None:
            # A face detector will occasionally miss one frame even with a
            # well-positioned subject. Clearing a 6-10 second pulse window on
            # every such miss made the scan require a literally perfect run,
            # despite the camera health gate accepting an 80% face rate.
            # Short gaps are bridged (or analysed around, see longest_clean_span);
            # a sustained loss drops the window and frees the peak tracker.
            if last_face_at is not None and now - last_face_at > face_gap_max_s:
                tracker.clear(unlock=True)
                last_nose = last_face_at = None
            continue
        rgb, nose, face_w, _ = r
        last_face_at = now

        if last_nose is not None and np.linalg.norm(nose - last_nose) / face_w > motion_max:
            bad_run += 1
            if bad_run > int(0.5 * fs):                   # >0.5 s of motion: restart window
                tracker.clear(unlock=True)
            last_nose = nose
            continue
        bad_run = 0
        last_nose = nose
        tracker.add(now, rgb)

        if now >= next_est:
            next_est = now + 1.0
            res = tracker.update(now)
            if on_update:
                # Do not present a rejected spectral peak as a live reading.
                accepted = res is not None and res["snr_db"] >= snr_min_db
                on_update(tracker.last_bpm if accepted else None, tracker.last_snr,
                          (now - t0) / duration_s)

    return tracker.result()


if __name__ == "__main__":
    print(measure_heart_rate(on_update=lambda b, s, p: print(
        f"{p*100:5.1f}%  bpm={b if b is None else round(b,1)}  snr={s if s is None else round(s,1)} dB")))
