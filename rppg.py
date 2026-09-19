"""
rppg.py - contactless heart rate for BracketBot / "Baymax" scan.

Pipeline:
  camera (locked exposure/WB) -> MediaPipe FaceLandmarker -> forehead + cheek ROI mask
  -> per-frame mean RGB + timestamp -> resample to uniform fs -> POS projection
  -> detrend + Butterworth bandpass -> Hann-windowed zero-padded FFT
  -> peak (parabolic interp) + SNR gate + tracking -> BPM, confidence

Deps: numpy, scipy, opencv-python, mediapipe>=0.10 (Tasks API)
Model: assets/models/face_landmarker.task (source and checksum in assets/models/README.md)

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
DEFAULT_MODEL = Path(__file__).resolve().parent / "assets" / "models" / "face_landmarker.task"


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
    """
    n_total = len(rgb)
    l = int(round(win_s * fs))
    H = np.zeros(n_total)
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
        H[n - l:n] += h - h.mean()                      # overlap-add
    return H


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


def analyze(t, rgb, fs=30.0, method="pos", prev_bpm=None, track_bpm=15.0):
    """
    Full chain, returning intermediates for debugging/plotting.
    t: timestamps (s), rgb: (N,3) per-frame ROI means.
    prev_bpm: if given, peak search is restricted to prev +/- track_bpm
              (HR can't jump 40 BPM in a second; kills harmonic/motion jumps).
    Returns dict(bpm, snr_db, h, freqs, psd, rgb_u) or None if < 6 s of data.
    """
    if len(t) < 2 or (t[-1] - t[0]) < 6.0:
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
    k = idx[np.argmax(psd[idx])]

    # Parabolic interpolation on log power for sub-bin peak
    if 0 < k < len(psd) - 1:
        a, b, c = np.log(psd[k - 1:k + 2] + 1e-20)
        d = 0.5 * (a - c) / (a - 2 * b + c + 1e-20)
        f0 = freqs[k] + d * (freqs[1] - freqs[0])
    else:
        f0 = freqs[k]

    return {"bpm": 60.0 * f0,
            "snr_db": snr_db(freqs, psd, f0, min(2 * HR_HI_HZ + 0.3, fs / 2)),
            "h": h, "freqs": freqs, "psd": psd, "rgb_u": x}


def estimate_hr(t, rgb, fs=30.0, method="pos", prev_bpm=None, track_bpm=15.0):
    """Returns (bpm, snr_db) or (None, None) if not enough data."""
    r = analyze(t, rgb, fs, method, prev_bpm, track_bpm)
    return (None, None) if r is None else (r["bpm"], r["snr_db"])


# ----------------------------------------------------------------------------
# ROI extraction (MediaPipe Face Mesh 478-landmark topology)
# ----------------------------------------------------------------------------

FOREHEAD = [10, 67, 69, 104, 108, 109, 151, 297, 299, 333, 337, 338]
L_CHEEK = [36, 50, 101, 117, 118, 123, 142, 187, 205]
R_CHEEK = [266, 280, 330, 346, 347, 352, 371, 411, 425]
NOSE_TIP = 1


class FaceROI:
    def __init__(self, model_path=DEFAULT_MODEL):
        import mediapipe as mp
        from mediapipe.tasks import python as mpp
        from mediapipe.tasks.python import vision
        self.mp = mp
        opts = vision.FaceLandmarkerOptions(
            base_options=mpp.BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
        )
        self.lm = vision.FaceLandmarker.create_from_options(opts)

    def __call__(self, frame_bgr, t_ms):
        """Returns (mean_rgb (3,), nose_xy, face_width_px, mask) or None if no face."""
        import cv2
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        res = self.lm.detect_for_video(
            self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb), int(t_ms))
        if not res.face_landmarks:
            return None
        pts = np.array([[p.x * w, p.y * h] for p in res.face_landmarks[0]], dtype=np.float32)

        mask = np.zeros((h, w), np.uint8)
        for region in (FOREHEAD, L_CHEEK, R_CHEEK):
            hull = cv2.convexHull(pts[region].astype(np.int32))
            cv2.fillConvexPoly(mask, hull, 255)
        mask = cv2.erode(mask, np.ones((5, 5), np.uint8))   # stay off edges/hairline

        if cv2.countNonZero(mask) < 400:                    # face too small/far
            return None
        m = cv2.mean(rgb, mask=mask)[:3]
        face_w = pts[:, 0].max() - pts[:, 0].min()
        return np.array(m), pts[NOSE_TIP], face_w, mask


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
# Measurement routine (call this from the robot's tool-calling agent)
# ----------------------------------------------------------------------------

def measure_heart_rate(duration_s=15.0, window_s=10.0, fs=30.0,
                       snr_min_db=-1.0, motion_max=0.03,
                       model_path=DEFAULT_MODEL, on_update=None, grab=None):
    """
    Blocking scan. Robot should be stationary (motors holding balance only)
    and say "please hold still" first.

    motion_max: max nose displacement per frame as fraction of face width.
                Frames above this are dropped; if too many drop, buffer resets.
    on_update(bpm, snr, progress): optional hook for the face display.
    grab: zero-arg callable returning one BGR frame (or None if no new frame yet).
          Defaults to open_camera(); the robot passes its head-camera reader here.
    Returns dict(bpm, snr_db, confident, n_estimates) or None.
    """
    grab = grab or open_camera(fps=int(fs))
    roi = FaceROI(model_path)
    buf_t, buf_rgb = deque(), deque()
    last_nose, estimates, prev = None, [], None
    t0 = time.monotonic()
    next_est = t0 + 6.0
    bad_run = 0

    while (now := time.monotonic()) - t0 < duration_s:
        frame = grab()
        if frame is None:
            continue
        r = roi(frame, now * 1000)
        if r is None:
            buf_t.clear(); buf_rgb.clear(); last_nose = None
            continue
        rgb, nose, face_w, _ = r

        if last_nose is not None and np.linalg.norm(nose - last_nose) / face_w > motion_max:
            bad_run += 1
            if bad_run > int(0.5 * fs):                   # >0.5 s of motion: restart window
                buf_t.clear(); buf_rgb.clear(); prev = None
            last_nose = nose
            continue
        bad_run = 0
        last_nose = nose

        buf_t.append(now); buf_rgb.append(rgb)
        while buf_t and now - buf_t[0] > window_s:
            buf_t.popleft(); buf_rgb.popleft()

        if now >= next_est:
            next_est = now + 1.0
            bpm, snr = estimate_hr(np.array(buf_t), np.array(buf_rgb), fs, prev_bpm=prev)
            if bpm is not None and snr >= snr_min_db:
                estimates.append(bpm)
                prev = bpm
            if on_update:
                on_update(bpm, snr, (now - t0) / duration_s)

    if not estimates:
        return None
    tail = estimates[-5:]
    return {"bpm": float(np.median(tail)),
            "snr_db": float(snr) if snr is not None else None,
            "confident": len(estimates) >= 3 and np.ptp(tail) < 8,
            "n_estimates": len(estimates)}


if __name__ == "__main__":
    print(measure_heart_rate(on_update=lambda b, s, p: print(
        f"{p*100:5.1f}%  bpm={b if b is None else round(b,1)}  snr={s if s is None else round(s,1)} dB")))
