"""
rppg.py - contactless heart rate for BracketBot / "Baymax" scan.

Pipeline:
  camera (no auto-exposure; a face-metered loop sets it) -> OpenCV YuNet face detector
  -> forehead + cheek ROI mask
  -> per-frame mean RGB + timestamp -> longest gap-free stretch -> resample to uniform fs
  -> POS projection -> detrend + Butterworth bandpass -> Hann-windowed zero-padded FFT
  -> peak, second-harmonic check (parabolic interp) + SNR gate + tracking
  -> SNR-weighted aggregation -> BPM, confidence

Deps: numpy, scipy, opencv-python
Model: assets/models/face_detection_yunet_2026may.onnx (source/checksum in assets/models/README.md)

NOT a medical device. Demo accuracy only.
"""
import sys
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
SUB_HARMONIC_STRONG = 0.4              # a half-peak this big is the fundamental, floor or no floor
SUB_HARMONIC_STRONG_TOL_HZ = 0.02      # ...if it sits this close to exactly half the peak
SCAN_GRID_HZ = 0.004                   # ~0.24 BPM: the grid the per-window spectra are stacked on
AGG_SNR_MIN_DB = -3.0                  # stacked spectrum must clear this to report a rate at all
AGG_SNR_CONF_DB = 2.0                  # ...and this before the rate may be called confident
AGG_MARGIN_MIN_DB = 2.5                # ...and must beat the band's best rival line by this much
DEFAULT_MODEL = (
    Path(__file__).resolve().parent / "assets" / "models" / "face_detection_yunet_2026may.onnx"
)


def scan_grid():
    """Frequency grid the whole-scan spectrum is stacked on, over the heart-rate band."""
    return np.arange(HR_LO_HZ, HR_HI_HZ + 1e-9, SCAN_GRID_HZ)


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


def fundamental(freqs, psd, k, ratio=SUB_HARMONIC_RATIO, floor_db=SUB_HARMONIC_FLOOR_DB,
                strong=SUB_HARMONIC_STRONG, strong_tol=SUB_HARMONIC_STRONG_TOL_HZ):
    """Bin of the pulse fundamental, given the strongest in-band bin k.

    A camera sees the PPG waveform, not a sine, and its second harmonic is often
    the taller line: the raw peak then reports exactly twice the real rate, and
    reports it consistently enough to look confident. If half the peak frequency
    is still a plausible heart rate and carries a peak of its own, that half is
    the fundamental and the tall line is its harmonic.

    Two kinds of evidence say the half is real, and they fail in different
    places. Its share of the peak's power is scale-free: noise near half the
    rate is a few percent of the peak, never half of it. Its height over the
    in-band noise floor is not scale-free, and that is the catch - on a weak
    scan the floor rises until a perfectly good fundamental sits under the bar,
    and the estimate silently doubles. That happened to real 66 BPM traces
    whose half-line held 68% of the peak's power but cleared the floor by only
    8.5 dB. So a half carrying `strong` of the peak is taken on that alone.

    That shortcut needs one guard, because a loud artefact near half the rate
    can carry the same share as a real fundamental. The two are told apart by
    where they sit: a genuine second harmonic is at exactly twice its
    fundamental, while an unrelated line merely lands nearby. A dim 124 BPM
    trace with an artefact at 57.6 - four BPM off the 61.5 that would be half -
    was halved until the shortcut demanded `strong_tol`. The floor test still
    judges the weaker, further-off halves it was written for.

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
    share = psd[j] / (psd[k] + 1e-20)
    if share >= strong and abs(freqs[j] - f_half) <= strong_tol:
        return j
    own_peak = psd[j] >= floor * 10 ** (floor_db / 10)
    return j if own_peak and share >= ratio else k


class SpectrumStack:
    """Incoherent average of every analysed window's spectrum across one scan.

    A single 10 s window only resolves the pulse when the pulse is strong. Its
    noise, though, is different in every window, while the rate is in the same
    place in all of them: averaging the windows' spectra pushes the noise floor
    down and leaves the pulse standing, so a rate too weak to win any one window
    can still be the tallest line in the average. That is the case the head
    camera actually delivers - a face a metre away, lit by a ceiling, with a
    pulse near the noise floor - and before this it produced no reading at all.

    Each window is normalised by its own in-band power before it is added, so a
    window that caught a motion jerk cannot shout down the clean ones, and then
    weighted by its SNR so a window the chain was sure about counts for more.
    Windows overlap heavily, so their noise is correlated and the floor falls
    more slowly than it would for independent samples; the gain is real but it
    is not the square root of the window count.
    """

    def __init__(self, grid=None):
        self.grid = scan_grid() if grid is None else np.asarray(grid, dtype=float)
        self.total = np.zeros_like(self.grid)
        self.weight = 0.0
        self.n_windows = 0

    def add(self, freqs, psd, snr_db_value):
        """Fold one window's spectrum in. Windows below the gate are added too: rescuing
        a weak scan is the whole point, and the SNR weight already discounts them."""
        freqs = np.asarray(freqs, dtype=float)
        psd = np.asarray(psd, dtype=float)
        band = (freqs >= HR_LO_HZ) & (freqs <= HR_HI_HZ)
        in_band = float(psd[band].sum()) if band.any() else 0.0
        if not np.isfinite(in_band) or in_band <= 0:
            return
        w = 10 ** (np.clip(float(snr_db_value), -10.0, 20.0) / 10.0)
        self.total += w * np.interp(self.grid, freqs, psd / in_band)
        self.weight += w
        self.n_windows += 1

    @property
    def spectrum(self):
        """The averaged in-band spectrum, or None before any window was added."""
        return None if self.weight <= 0 else self.total / self.weight

    def peak(self):
        """dict(bpm, snr_db, margin_db) off the averaged spectrum, or None if it is empty.

        Two numbers describe the line, because a scan can fail in two ways.
        `snr_db` asks how far it stands above the broad in-band noise floor, and
        catches a scan whose spectrum is flat. `margin_db` asks how far it beats
        the best rival line elsewhere in the band, and catches the other case:
        a spectrum with several comparable lines, where the tallest wins by
        chance and moves to a different rate on the next scan. On synthetic
        scans a real pulse beats its rival by 5 dB or more, while traces with no
        pulse in them top out near 3 dB whatever their SNR says.
        """
        avg = self.spectrum
        if avg is None or not np.isfinite(avg).all():
            return None
        k = fundamental(self.grid, avg, int(np.argmax(avg)))
        f0 = _sub_bin_peak(self.grid, avg, k)
        sel = (np.abs(self.grid - f0) <= 0.1) | (np.abs(self.grid - 2 * f0) <= 0.2)
        if sel.all() or not sel.any():
            return None
        snr = 10 * np.log10(avg[sel].sum() / (avg[~sel].sum() + 1e-12) + 1e-12)
        # The rate's own line, its harmonic and its half all belong to the same
        # candidate: a rival has to be a different rate, not another view of this one.
        own = ((np.abs(self.grid - f0) <= 0.15) | (np.abs(self.grid - 2 * f0) <= 0.25)
               | (np.abs(self.grid - f0 / 2) <= 0.15))
        if own.all() or not own.any():
            margin = float("inf")
        else:
            margin = 10 * np.log10(avg[own].max() / (avg[~own].max() + 1e-20) + 1e-20)
        return {"bpm": 60.0 * f0, "snr_db": float(snr), "margin_db": float(margin)}


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
BOX_SMOOTH = 0.65                      # weight on the previous box; the rest is the new one
BOX_JUMP = 0.35                        # centre move, as a fraction of width, that ends smoothing
SKIN_CHROMA_RADIUS = 14.0              # Cr/Cb distance from the face's own median that stays skin
SKIN_LUM_PCT = (4.0, 96.0)             # luminance percentiles kept: drops shadow and highlights
MIN_SKIN_PIXELS = 400                  # below this the face is too small or too occluded to use

REGION_NAMES = ("forehead", "cheek-left", "cheek-right")

# Face-relative polygons, in the order of REGION_NAMES. Fixed fractions of the
# YuNet box avoid the eyes, nose, mouth, hairline, and background. Less
# anatomically precise than a 478-point mesh, but rPPG only needs stable skin
# averages, and YuNet has supported ARM64 wheels through OpenCV.
REGION_POLYGONS = (
    # Forehead: below the hairline and above the eyebrows.
    ((0.24, 0.10), (0.76, 0.10), (0.69, 0.34), (0.31, 0.34)),
    # Camera-left and camera-right cheeks, clear of nose and mouth.
    ((0.10, 0.46), (0.38, 0.42), (0.40, 0.72), (0.16, 0.76)),
    ((0.62, 0.42), (0.90, 0.46), (0.84, 0.76), (0.60, 0.72)),
)


def steady_box(previous, box, smooth=BOX_SMOOTH, jump=BOX_JUMP):
    """Blend a face box with the one before it, unless the face actually moved.

    A detector re-running on every frame puts the box a pixel or two somewhere
    else each time, even on a person holding still. The ROI then averages
    slightly different skin every frame, and that wobble lands in the same band
    as the pulse - which is a good part of why a small, distant face reads worse
    than a near one. Smoothing the box removes the wobble without hiding real
    movement: a centre that moves more than `jump` of the box width is someone
    moving, and the new box is taken whole so tracking stays immediate.
    """
    box = np.asarray(box, dtype=float)
    if previous is None:
        return box
    previous = np.asarray(previous, dtype=float)
    moved = np.hypot(*((previous[:2] + previous[2:4] / 2) - (box[:2] + box[2:4] / 2)))
    if not np.isfinite(moved) or moved > jump * max(box[2], 1e-6):
        return box
    out = box.copy()
    out[:14] = smooth * previous[:14] + (1.0 - smooth) * box[:14]
    return out


def skin_pixels(frame_bgr, mask, chroma_radius=SKIN_CHROMA_RADIUS, lum_pct=SKIN_LUM_PCT):
    """Narrow a face mask to the pixels that look like the same lit skin.

    The box polygons are mostly skin by construction, but the minority that is
    not - a strand of hair, a glasses frame, a specular highlight on the nose
    bridge, a cheek edge that catches background - does real damage: those
    pixels carry no pulse, so they dilute a signal that is only 0.1-1% of the
    brightness to begin with, and they move in and out of the mask as the box
    shifts.

    The gate is relative to the face's own median colour rather than a fixed
    skin range, so it adapts to the person and to whatever white balance the
    camera daemon settled on. That matters here: a fixed Cr/Cb box is the usual
    way this is done and it is exactly where rPPG pipelines bake in an
    assumption about skin tone. This one only asks that a pixel match the rest
    of the face. Returns a uint8 mask, or the mask unchanged if the gate would
    leave too little to average over.
    """
    import cv2

    if mask is None:
        return None
    selected = mask > 0
    if not selected.any():
        return mask
    ycrcb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = (ycrcb[..., i].astype(np.float32) for i in range(3))
    med_cr, med_cb = float(np.median(cr[selected])), float(np.median(cb[selected]))
    lo, hi = np.percentile(y[selected], lum_pct)
    keep = (selected
            & ((cr - med_cr) ** 2 + (cb - med_cb) ** 2 <= chroma_radius ** 2)
            & (y >= lo) & (y <= hi))
    if int(keep.sum()) < MIN_SKIN_PIXELS:
        return mask
    return (keep * 255).astype(np.uint8)


def face_polygons(face, image_shape):
    """Pixel polygons for the three skin regions of one YuNet detection.

    YuNet rows are ``x, y, w, h``, five landmark pairs (eyes, nose, mouth
    corners), then confidence. Returns one polygon per REGION_NAMES entry, or
    None when the row is unusable.
    """
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

    return [np.asarray([point(*corner) for corner in polygon], dtype=np.int32)
            for polygon in REGION_POLYGONS]


def face_skin_mask(face, image_shape):
    """Return a conservative forehead/cheek mask from one YuNet detection."""
    import cv2

    polygons = face_polygons(face, image_shape)
    if polygons is None:
        return None
    mask = np.zeros(image_shape[:2], np.uint8)
    for polygon in polygons:
        cv2.fillConvexPoly(mask, polygon, 255)
    return mask


def region_label_mask(face, image_shape):
    """Mask labelled 1..3 by region, for the debug view and the per-region means."""
    import cv2

    polygons = face_polygons(face, image_shape)
    if polygons is None:
        return None
    labels = np.zeros(image_shape[:2], np.uint8)
    for index, polygon in enumerate(polygons, start=1):
        cv2.fillConvexPoly(labels, polygon, index)
    return labels


class FaceROI:
    """Per-frame mean skin colour from the best face in the frame.

    Beyond finding the face, this stage decides which pixels are averaged, and
    that decision is most of the difference between a usable trace and noise on
    a camera whose exposure cannot be locked: the box is steadied so the ROI
    stops wobbling frame to frame, and the pixels inside it are narrowed to the
    ones that match the rest of the face. `last` holds what it did, so the
    debug view can show the same thing the estimate was computed from.
    """

    def __init__(self, model_path=DEFAULT_MODEL, steady=True, skin_gate=True, detector=None):
        import cv2

        self.cv2 = cv2
        self.detector = detector or cv2.FaceDetectorYN.create(
            str(model_path), "", (320, 320), FACE_SCORE, 0.3, 50
        )
        self.detector_size = None
        self.steady, self.skin_gate = steady, skin_gate
        self.box = None                     # smoothed detection, carried between frames
        self.last = None                    # what the last call saw, for the debug view

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
            self.box, self.last = None, None
            return None
        face = max(faces, key=lambda row: float(row[2] * row[3] * row[-1]))
        face = np.asarray(face, dtype=np.float32).copy()
        if detect_size != (w, h):
            face[[0, 2, 4, 6, 8, 10, 12]] *= w / detect_w
            face[[1, 3, 5, 7, 9, 11, 13]] *= h / detect_h
        raw = face
        if self.steady:
            face = steady_box(self.box, face)
            self.box = face

        mask = face_skin_mask(face, frame_bgr.shape)
        if mask is None or self.cv2.countNonZero(mask) < MIN_SKIN_PIXELS:
            self.last = None
            return None                                   # face too small/far
        region_mask = mask
        if self.skin_gate:
            mask = skin_pixels(frame_bgr, mask)

        rgb = self.cv2.cvtColor(frame_bgr, self.cv2.COLOR_BGR2RGB)
        mean_rgb = np.asarray(self.cv2.mean(rgb, mask=mask)[:3])
        nose = np.asarray(face[8:10], dtype=np.float32)

        labels = region_label_mask(face, frame_bgr.shape)
        regions = {}
        if labels is not None:
            for index, name in enumerate(REGION_NAMES, start=1):
                part = ((labels == index) & (mask > 0)).astype(np.uint8)
                count = int(part.sum())
                regions[name] = {
                    "pixels": count,
                    "rgb": None if not count else [round(float(v), 2)
                                                   for v in self.cv2.mean(rgb, mask=part)[:3]],
                }
        offered = int(self.cv2.countNonZero(region_mask))
        kept = int(self.cv2.countNonZero(mask))
        self.last = {
            "box": [float(v) for v in raw[:4]],
            "steady_box": [float(v) for v in face[:4]],
            "score": float(face[-1]),
            "landmarks": [[float(face[i]), float(face[i + 1])] for i in range(4, 14, 2)],
            "mask": mask,
            "region_mask": region_mask,
            "labels": labels,
            "pixels": kept,
            "skin_frac": round(kept / offered, 3) if offered else 0.0,
            "regions": regions,
            "rgb": [round(float(v), 2) for v in mean_rgb],
        }
        return mean_rgb, nose, float(face[2]), mask


# ----------------------------------------------------------------------------
# Face-metered exposure
# ----------------------------------------------------------------------------
#
# A camera's own auto-exposure meters the whole frame, so it spends the sensor's
# range on whatever fills it - a bright window, a white wall - and leaves the face
# wherever it lands. Both halves of that hurt here. A face down at the dark end of
# the range carries its pulse in a couple of counts, and every AE correction is a
# brightness step orders of magnitude larger than the ~0.1-1% modulation we are
# looking for, which POS only partly rejects and green-only does not reject at all.
#
# So the loop below meters the skin pixels the estimate is actually taken over, and
# then gets out of the way: it moves exposure only while the face sits outside a
# dead band, holds still once it is inside, and re-opens only on a drift wide enough
# to matter (a lamp switched on, someone stepping into the shade). Every move it
# does make is reported, so the caller can throw away the window the step falls in
# rather than read a fake peak off it.

EXPOSURE_TARGET = 0.52                 # skin luma we aim for, as a fraction of full scale
EXPOSURE_DEAD_BAND = 0.06              # how far off target the face may sit before we move
EXPOSURE_REARM_BAND = 0.11             # ...and how far it must drift to reopen a settled loop
EXPOSURE_CLIP_HI_MAX = 0.02            # blown skin pixels tolerated before forcing a darker frame
EXPOSURE_CLIP_LO_MAX = 0.05            # crushed skin pixels tolerated before forcing a brighter one
EXPOSURE_STEP_MAX = 1.8                # largest single move, and 1/1.8 the other way
EXPOSURE_SETTLE_S = 0.7                # a change takes a few frames to reach the pixels
EXPOSURE_HOLD_S = 1.5                  # in band this long -> settled, stop touching the camera
EXPOSURE_MIN_FRAMES = 3                # frames of face before the first move: no metering a blink
EXPOSURE_MIN_MOVE = 0.04               # smaller corrections are not worth a step in the signal
CLIP_HI_LEVEL, CLIP_LO_LEVEL = 250, 16


def skin_luma_stats(frame_bgr, mask, clip_hi=CLIP_HI_LEVEL, clip_lo=CLIP_LO_LEVEL,
                    min_pixels=MIN_SKIN_PIXELS):
    """How bright the skin is, measured over exactly the pixels the mean RGB came from.

    Metering the same mask the signal chain uses is the whole point: a frame
    average is mostly background, and background brightness is not what the
    estimate lives on. Returns fractions of full scale - the mean luma, plus how
    much of the skin is pinned at either rail. A clipped pixel has had its
    modulation thrown away by the sensor, so it carries no pulse at all, which
    makes clipping worth acting on even when the mean looks healthy.
    """
    if mask is None:
        return None
    px = frame_bgr[mask > 0]
    if len(px) < min_pixels:
        return None
    px = px.astype(np.float32)
    return {"luma": float(np.dot(px.mean(axis=0), (0.114, 0.587, 0.299)) / 255.0),   # BGR order
            "clip_hi": float(np.mean(px.max(axis=1) >= clip_hi)),
            "clip_lo": float(np.mean(px.min(axis=1) <= clip_lo)),
            "pixels": int(len(px))}


class FaceExposure:
    """Decides when the camera should change exposure, and by how much.

    Camera-free and unit-free: it takes skin_luma_stats in and returns a factor to
    multiply the light by, which the control classes below translate into whatever
    the hardware means by "exposure". Returning None - the normal answer once a
    face is exposed well - is what keeps this from becoming the auto-exposure it
    replaces.
    """

    def __init__(self, target=EXPOSURE_TARGET, dead_band=EXPOSURE_DEAD_BAND,
                 rearm_band=EXPOSURE_REARM_BAND, step_max=EXPOSURE_STEP_MAX,
                 settle_s=EXPOSURE_SETTLE_S, hold_s=EXPOSURE_HOLD_S,
                 min_frames=EXPOSURE_MIN_FRAMES, clip_hi_max=EXPOSURE_CLIP_HI_MAX,
                 clip_lo_max=EXPOSURE_CLIP_LO_MAX, min_move=EXPOSURE_MIN_MOVE):
        self.target, self.dead_band, self.rearm_band = target, dead_band, rearm_band
        self.step_max, self.settle_s, self.hold_s = step_max, settle_s, hold_s
        self.min_frames, self.min_move = min_frames, min_move
        self.clip_hi_max, self.clip_lo_max = clip_hi_max, clip_lo_max
        self.settled = False                # in band and holding: the camera is left alone
        self.changes = 0
        self.last = None                    # last stats seen, for the debug readout
        self.reason = "no face"
        self._face_frames = 0
        self._in_band_since = None
        self._last_change_at = None
        self._last_dir = 0
        self._damp = 1.0                    # shrinks when the loop overshoots and turns around

    @property
    def state(self):
        if self.last is None:
            return "no face"
        return "settled" if self.settled else "metering"

    def describe(self):
        """One line for the debug panel."""
        if self.last is None:
            return f"face metering: {self.reason}"
        moves = f"{self.changes} move" + ("" if self.changes == 1 else "s")
        line = f"skin {100 * self.last['luma']:.0f}%  {self.state}, {moves}"
        return line if self.settled else f"{line}: {self.reason}"

    def observe(self, now, stats):
        """One frame of skin stats in; a factor to multiply the light by, or None."""
        self.last = stats
        if stats is None:
            self._face_frames, self._in_band_since = 0, None
            self.reason = "no face"
            return None
        self._face_frames += 1
        if self._face_frames < self.min_frames:
            self.reason = "waiting for a steady face"
            return None

        luma = stats["luma"]
        too_bright = stats["clip_hi"] > self.clip_hi_max
        too_dark = stats["clip_lo"] > self.clip_lo_max and luma < self.target
        # A settled loop is harder to reopen than a metering one is to keep moving:
        # without the wider band it would chase the ordinary wobble of a lit face.
        band = self.rearm_band if self.settled else self.dead_band
        if not (too_bright or too_dark or abs(luma - self.target) > band):
            return self._hold(now, "in band")

        self.settled, self._in_band_since = False, None
        if self._last_change_at is not None and now - self._last_change_at < self.settle_s:
            self.reason = "waiting for the last change to land"
            return None

        factor = self.target / max(luma, 0.02)
        if too_bright:
            factor = min(factor, 1.0 - min(0.4, 8.0 * stats["clip_hi"]))
            self.reason = f"skin clipping ({100 * stats['clip_hi']:.0f}%)"
        elif too_dark:
            factor = max(factor, 1.0 + min(0.4, 8.0 * stats["clip_lo"]))
            self.reason = f"skin crushed ({100 * stats['clip_lo']:.0f}%)"
        else:
            self.reason = f"skin {100 * luma:.0f}%, want {100 * self.target:.0f}%"

        direction = 1 if factor > 1.0 else -1
        if self._last_dir and direction != self._last_dir:
            self._damp = max(0.25, 0.5 * self._damp)      # it overshot: take smaller bites
        factor = float(np.exp(self._damp * np.log(factor)))
        factor = float(np.clip(factor, 1.0 / self.step_max, self.step_max))
        if abs(factor - 1.0) < self.min_move:
            # Nothing left worth a visible step in the trace - count that as arrived,
            # or a face the camera cannot quite reach would meter forever.
            return self._hold(now, "close enough")
        self._last_change_at, self._last_dir = now, direction
        self.changes += 1
        return factor

    def _hold(self, now, reason):
        if self._in_band_since is None:
            self._in_band_since = now
        if not self.settled and now - self._in_band_since >= self.hold_s:
            self.settled = True
        self.reason = "settled" if self.settled else reason
        return None


class OpenCVExposure:
    """Relative exposure changes on a UVC camera through OpenCV's property API.

    The units are neither portable nor documented per backend. DirectShow and
    Media Foundation report exposure as log2 seconds, so -6 means 1/64 s and one
    count is a whole stop; V4L2 reports exposure_time_absolute in 100 us units,
    where the number scales with the light. Callers only ever say "multiply the
    light by f"; this is what knows which camera means what by the number it holds.
    """

    LOG_RANGE = (-13.0, -2.0)              # 1/8192 s .. 1/4 s
    ABS_RANGE = (1.0, 2500.0)              # 100 us units: 0.1 ms .. 250 ms

    def __init__(self, cap, platform=None, use_gain=True, gain_range=(1.0, 255.0), fps=None):
        import cv2

        self.cv2, self.cap = cv2, cap
        self.platform = sys.platform if platform is None else platform
        self.log_units = self.platform == "win32"
        self.use_gain, self.gain_range = use_gain, gain_range
        # An exposure longer than a frame does not brighten anything, it drops the
        # frame rate (a DSHOW webcam at -2 delivers 4 fps), so the range stops there.
        self.log_range, self.abs_range = self.LOG_RANGE, self.ABS_RANGE
        if fps:
            self.log_range = (self.LOG_RANGE[0], min(self.LOG_RANGE[1], float(np.floor(np.log2(1.0 / fps)))))
            self.abs_range = (self.ABS_RANGE[0], min(self.ABS_RANGE[1], float(np.floor(1e4 / fps))))
        self.applied = 0
        self.locked = False
        self.note = ""

    @property
    def available(self):
        return self.platform != "darwin"

    def auto(self, settle_frames=30):
        """Hand exposure and WB back to the camera, and give its AE time to land.

        DirectShow drivers remember manual mode after the process exits, so without
        this a webcam opens at whatever exposure the last run left it on - and so
        does every other app that uses it.
        """
        cv2 = self.cv2
        if not self.available:
            return
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75 if self.log_units else 3)  # DSHOW 0.75 / V4L2 3 = auto
        self.cap.set(cv2.CAP_PROP_AUTO_WB, 1)
        self.locked = False
        for _ in range(settle_frames):
            self.cap.read()

    def _frame_mean(self, n):
        """Mean pixel level over n frames, after flushing the few already in flight."""
        read = getattr(self.cap, "read", None)
        if read is None or n <= 0:
            return None
        vals = []
        for i in range(n + 3):
            ok, frame = read()
            if ok and frame is not None and i >= 3:
                vals.append(float(frame.mean()))
        return float(np.mean(vals)) if vals else None

    def manual(self, check_frames=10, min_kept=0.6):
        """Take the camera off auto-exposure and auto-WB so a setting sticks.

        Best effort: laptop webcams differ in which of these they honour, and some
        accept the call and ignore it. Many also reach their auto brightness with
        gain the property API cannot touch, or report a stale exposure value, so a
        lock that leaves the frame much darker than AE had it is undone: a dark face
        has no pulse to measure, and AE drift is the lesser problem.
        """
        cv2 = self.cv2
        if not self.available:
            self.note = "macOS: OpenCV cannot set exposure (camera AE stays on)"
            return self.note
        before = self._frame_mean(check_frames)
        exposure = float(np.clip(self.cap.get(cv2.CAP_PROP_EXPOSURE),     # may be stale under AE
                                 *(self.log_range if self.log_units else self.abs_range)))
        if self.log_units:
            ok = self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)    # DSHOW: 0.25 = manual
        else:
            ok = self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)       # V4L2 menu: 1 = manual
        self.cap.set(cv2.CAP_PROP_EXPOSURE, exposure)              # hold the settled value
        self.cap.set(cv2.CAP_PROP_AUTO_WB, 0)
        after = self._frame_mean(check_frames)
        if before and after is not None and after < min_kept * before:
            self.auto(settle_frames=check_frames)
            self.note = f"lock darkened frame ({before:.0f} -> {after:.0f}), camera AE kept"
            return self.note
        self.locked = bool(ok)
        self.note = f"AE {'manual' if ok else 'lock unsupported'} (exp={exposure:g})"
        return self.note

    def apply(self, factor):
        """Multiply the light reaching the sensor by factor. Returns what it did, or None."""
        cv2 = self.cv2
        if not self.available:
            self.note = "no exposure control on this platform"
            return None
        value = float(self.cap.get(cv2.CAP_PROP_EXPOSURE))
        new = value
        if self.log_units:
            stops = np.log2(factor)
            step = int(np.sign(stops) * np.floor(abs(stops) + 0.5))
            if step:                       # a whole stop is the smallest move these units allow
                new = float(np.clip(value + step, *self.log_range))
        else:
            new = float(np.clip(round(value * factor), *self.abs_range))
        if new != value and self.cap.set(cv2.CAP_PROP_EXPOSURE, new):
            self.applied += 1
            self.note = f"exposure {value:g} -> {new:g}"
            return {"exposure": new, "gain": None, "note": self.note}
        return self._gain_fallback(factor, value)

    def _gain_fallback(self, factor, exposure):
        """Exposure is railed or too coarse to ask: buy the rest with sensor gain.

        Gain costs noise, which the bandpass and the SNR gate can argue with; a
        face stuck at the bottom of the range has no pulse left to argue about.
        """
        if not self.use_gain:
            self.note = "exposure at its limit"
            return None
        gain = float(self.cap.get(self.cv2.CAP_PROP_GAIN))
        if not np.isfinite(gain) or gain <= 0:
            self.note = "camera ignored the exposure change"
            return None
        new = float(np.clip(round(gain * factor), *self.gain_range))
        if new == gain or not self.cap.set(self.cv2.CAP_PROP_GAIN, new):
            self.note = "camera ignored the exposure change"
            return None
        self.applied += 1
        self.note = f"gain {gain:g} -> {new:g}"
        return {"exposure": exposure, "gain": new, "note": self.note}


class Picamera2Exposure:
    """Relative exposure changes on a Picamera2 sensor, in whole mains-flicker periods.

    Exposure is only ever set to a multiple of flicker_us, because an exposure that
    does not span a whole cycle of the room's lighting samples the lamp's 100/120 Hz
    ripple instead of averaging it away, and the beat lands straight in the pulse
    band. The rest of each change goes to analogue gain, which costs noise but no
    flicker, and the longest flicker-safe exposure is preferred so that as little
    gain as possible is used.
    """

    def __init__(self, cam, exposure_us=8333, gain=4.0, flicker_us=8333, fps=30,
                 gain_range=(1.0, 8.0)):
        self.cam, self.flicker_us = cam, int(flicker_us)
        self.periods = max(1, int(round(exposure_us / self.flicker_us)))
        self.max_periods = max(1, int((0.9e6 / fps) // self.flicker_us))   # must fit in a frame
        self.gain, self.gain_range = float(gain), gain_range
        self.applied = 0
        self.note = ""
        self.available = True

    @property
    def exposure_us(self):
        return self.periods * self.flicker_us

    def apply(self, factor):
        want = self.exposure_us * self.gain * float(factor)
        lo, hi = self.gain_range
        best = None
        for periods in range(1, self.max_periods + 1):
            gain = want / (periods * self.flicker_us)
            if lo <= gain <= hi:
                best = (periods, gain)     # keep going: the last fit is the longest exposure
        if best is None:                   # out of range in one direction; go as far as we can
            periods = self.max_periods if want > self.exposure_us * self.gain else 1
            best = (periods, float(np.clip(want / (periods * self.flicker_us), lo, hi)))
        periods, gain = best
        if periods == self.periods and abs(gain - self.gain) < 0.02:
            self.note = "exposure and gain are already at their limit"
            return None
        self.periods, self.gain = periods, round(float(gain), 3)
        self.cam.set_controls({"ExposureTime": self.exposure_us, "AnalogueGain": self.gain})
        self.applied += 1
        self.note = f"exposure {self.exposure_us} us ({periods} flicker cycles), gain {self.gain:g}"
        return {"exposure": self.exposure_us, "gain": self.gain, "note": self.note}


class Camera:
    """A frame source plus whatever exposure control it turned out to have.

    Callable, so anything that only ever wanted grab() keeps working; controls is
    None when the camera cannot be driven, and the metering loop then stays off
    rather than pretending it steered something.
    """

    def __init__(self, grab, controls=None, close=None, note=""):
        self._grab, self.controls, self._close, self.note = grab, controls, close, note

    def __call__(self):
        return self._grab()

    def close(self):
        if self._close:
            self._close()


# ----------------------------------------------------------------------------
# Camera: the sensor's own auto-exposure off, the face-metered loop on
# ----------------------------------------------------------------------------

def open_camera(width=640, height=480, fps=30, exposure_us=8333):
    """
    Auto-exposure / auto-WB / auto-gain MUST be off: their adjustments are
    bigger than the pulse signal (~0.1-1% intensity) and show up as fake peaks.
    What takes their place is FaceExposure, which meters the face rather than the
    frame and then holds still - see measure_heart_rate(auto_exposure=True).
    exposure_us=8333 = 1/120 s -> integrates exactly one flicker cycle of
    60 Hz mains lighting (use 10000 in 50 Hz countries), and stays the unit the
    metering loop moves exposure in.
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
        gain = md.get("AnalogueGain", 4.0)
        cam.set_controls({                                # ...then freeze them
            "AeEnable": False, "AwbEnable": False,
            "ExposureTime": exposure_us,
            "AnalogueGain": gain,
            "ColourGains": md.get("ColourGains", (1.8, 1.5)),
        })
        # The sensor's settled guess is a starting point, not an answer: it metered
        # the frame, and the face is what has to be exposed well.
        controls = Picamera2Exposure(cam, exposure_us=exposure_us, gain=gain,
                                     flicker_us=exposure_us, fps=fps)
        return Camera(cam.capture_array,                  # RGB888 is BGR-ordered in memory
                      controls, close=cam.stop, note="picamera2")
    except ImportError:
        import cv2
        cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, fps)
        for _ in range(30):                               # let the camera's own AE settle first
            cap.read()
        controls = OpenCVExposure(cap, platform="linux")
        note = controls.manual()
        # Some UVC cameras ignore the property API; then lock them from outside
        # beforehand (names vary by camera; `v4l2-ctl -l`):
        #   v4l2-ctl -c auto_exposure=1 -c exposure_time_absolute=83 \
        #            -c white_balance_automatic=0 -c gain=<fixed>
        return Camera(lambda: cap.read()[1], controls, close=cap.release, note=note)


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
                 conf_span_s=6.0, agg_snr_min_db=AGG_SNR_MIN_DB,
                 agg_snr_conf_db=AGG_SNR_CONF_DB, agg_margin_min_db=AGG_MARGIN_MIN_DB):
        self.fs, self.window_s, self.snr_min_db = fs, window_s, snr_min_db
        self.method, self.track_bpm = method, track_bpm
        self.unlock_after, self.keep_n, self.spread_max = unlock_after, keep_n, spread_max
        self.conf_span_s = conf_span_s
        self.agg_snr_min_db, self.agg_snr_conf_db = agg_snr_min_db, agg_snr_conf_db
        self.agg_margin_min_db = agg_margin_min_db
        self.t, self.rgb = deque(), deque()
        self.estimates = []                 # (t, bpm, snr_db) that passed the gate
        self.stack = SpectrumStack()        # every window's spectrum, gate or no gate
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
        """Drop the buffer (face lost, or motion). Accepted estimates and the stacked
        spectrum survive: both describe the scan, not the window that was interrupted."""
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
        if res.get("freqs") is not None and res.get("psd") is not None:
            self.stack.add(res["freqs"], res["psd"], res["snr_db"])
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
        """The scan's answer, or None when nothing in it was worth reporting.

        The rate is read off the stacked spectrum rather than from the winning
        windows, because stacking is what lets a weak scan produce any answer at
        all; `window_bpm` keeps the SNR-weighted median of the windows that
        passed the gate, as an independent view of the same scan. A rate is
        reported only once the stacked spectrum clears `agg_snr_min_db`, which
        is what separates a pulse near the noise floor from a scan that only
        contains noise - both used to look alike to the per-window gate, one as
        a wrong number and the other as no number.

        Confidence is deliberately harder to earn than a reading: the stacked
        spectrum must clear `agg_snr_conf_db`, at least three windows must have
        passed the gate, they must agree within `spread_max`, and they must span
        `conf_span_s` of the scan. Consecutive windows overlap by 90%, so their
        agreeing with each other is weak evidence on its own.
        """
        peak = self.stack.peak()
        tail = self.estimates[-self.keep_n:]
        window_bpm = spread = snr_med = None
        span = 0.0
        if tail:
            bpms = np.array([b for _, b, _ in tail])
            snrs = np.array([s for _, _, s in tail])
            window_bpm = weighted_median(bpms, 10 ** (np.clip(snrs, -10.0, 20.0) / 10.0))
            spread = float(np.ptp(bpms))
            snr_med = float(np.median(snrs))
            span = self.estimates[-1][0] - self.estimates[0][0]

        if peak is not None and (peak["snr_db"] < self.agg_snr_min_db
                                 or peak["margin_db"] < self.agg_margin_min_db):
            return None                     # the scan's own spectrum says there is no pulse in it
        if peak is not None:
            bpm, agg_snr = peak["bpm"], peak["snr_db"]
        elif window_bpm is not None:
            bpm, agg_snr = window_bpm, snr_med
        else:
            return None                     # nothing was ever analysed

        windows_agree = bool(len(self.estimates) >= 3
                             and spread is not None and spread <= self.spread_max
                             and snr_med is not None and snr_med >= self.snr_min_db + 2.0
                             and span >= self.conf_span_s)
        # The windows agreeing among themselves says nothing about whether they
        # agree with the rate actually being reported. They are two readings of
        # the same scan from different parts of the evidence, and a scan where
        # they disagree is not one to be confident about, however tidy either
        # looks alone. On the synthetic set one scan splits them by 53 BPM.
        if windows_agree and peak is not None and window_bpm is not None:
            windows_agree = abs(bpm - window_bpm) <= self.spread_max
        return {"bpm": bpm,
                "snr_db": round(float(agg_snr), 2),
                "margin_db": None if peak is None else round(peak["margin_db"], 2),
                "window_bpm": None if window_bpm is None else round(window_bpm, 1),
                "spread_bpm": None if spread is None else round(spread, 1),
                "span_s": round(span, 1),
                "confident": bool(windows_agree
                                  and (peak is None or agg_snr >= self.agg_snr_conf_db)),
                "n_estimates": len(self.estimates),
                "n_windows": self.stack.n_windows}


# ----------------------------------------------------------------------------
# Measurement routine (call this from the robot's tool-calling agent)
# ----------------------------------------------------------------------------

def measure_heart_rate(duration_s=15.0, window_s=10.0, fs=30.0,
                       snr_min_db=-1.0, motion_max=0.03, face_gap_max_s=0.75,
                       model_path=DEFAULT_MODEL, on_update=None, grab=None,
                       on_frame=None, auto_exposure=True, on_exposure=None):
    """
    Blocking scan. Robot should be stationary (motors holding balance only)
    and say "please hold still" first.

    motion_max: max nose displacement per frame as fraction of face width.
                Frames above this are dropped; if too many drop, buffer resets.
    face_gap_max_s: tolerate brief landmark misses without throwing away the
                    several seconds of good signal already collected. Longer
                    losses reset the window and peak tracker.
    on_update(bpm, snr, progress): optional hook for the face display.
    on_frame(frame, roi, tracker, status): optional hook, every frame, for a debug
          view. `roi` is FaceROI.last (None when no face) and `status` is the
          one-word reason the frame was used or dropped. It is called inside the
          scan loop, so a slow hook costs frame rate: encode elsewhere.
    grab: zero-arg callable returning one BGR frame (or None if no new frame yet).
          Defaults to open_camera(); the robot passes its head-camera reader here.
    auto_exposure: meter exposure on the face while the scan runs, using whatever
          control the frame source carries (`grab.controls`). A source that has
          none - the robot's shared camera topic, a recording - leaves it off,
          because nothing would come of the request.
    on_exposure(factor, meter, result): optional hook, called when the camera was
          actually moved, for a readout.
    Returns dict(bpm, snr_db, spread_bpm, span_s, confident, n_estimates) or None.
    """
    grab = grab or open_camera(fps=int(fs))
    controls = getattr(grab, "controls", None)
    meter = FaceExposure() if auto_exposure and controls is not None else None
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
            if meter:
                meter.observe(now, None)
            if on_frame:
                on_frame(frame, None, tracker, "no-face")
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
        rgb, nose, face_w, mask = r
        last_face_at = now

        if meter:
            factor = meter.observe(now, skin_luma_stats(frame, mask))
            applied = controls.apply(factor) if factor is not None else None
            if applied:
                # The step is orders of magnitude larger than a pulse, so the window
                # it falls in would read as a peak. Drop the buffer and refill it;
                # the tracking lock stays, because the rate did not change.
                tracker.clear()
                next_est = max(next_est, now + MIN_SECONDS)
                if on_exposure:
                    on_exposure(factor, meter, applied)

        if last_nose is not None and np.linalg.norm(nose - last_nose) / face_w > motion_max:
            bad_run += 1
            if bad_run > int(0.5 * fs):                   # >0.5 s of motion: restart window
                tracker.clear(unlock=True)
            last_nose = nose
            if on_frame:
                on_frame(frame, roi.last, tracker, "motion")
            continue
        bad_run = 0
        last_nose = nose
        tracker.add(now, rgb)
        if on_frame:
            on_frame(frame, roi.last, tracker, "measuring")

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
