"""
laptop_rppg.py - concept check for rPPG on a laptop webcam, with a live debug view.

Uses the same signal chain and the same HeartRateTracker as rppg.py that scripts/robot_rppg.py
runs on the robot's head camera, so what you tune here is what the robot measures.

NOT a medical device: a demo-grade estimate from a camera. See docs/rppg-robot-port.md.

Uses the model at assets/models/face_landmarker.task, downloading it if absent.

Setup:
  uv run --extra rppg python laptop_rppg.py                  # live, camera 0
  uv run --extra rppg python laptop_rppg.py --cam 1          # other camera
  uv run --extra rppg python laptop_rppg.py --replay rec.csv # offline re-analysis

Keys (live window):
  q / Esc   quit
  m         toggle method POS <-> GREEN (watch GREEN fail under lighting changes)
  r         start/stop recording CSV (t, R, G, B) for offline tuning
  space     reset buffer
  l         try to (re)lock exposure / white balance

Face ~50-70 cm from camera, even front lighting, hold still ~10 s for first reading.
Compare against a smartwatch / phone pulse app / MAX30102.

The 'r' recording holds someone's pulse trace: treat the CSV as personal data (gitignored).
"""
import argparse
import csv
import os
import sys
import time
import urllib.request
from collections import deque

import cv2
import numpy as np

from rppg import FaceROI, HeartRateTracker, MIN_SECONDS, DEFAULT_MODEL, HR_LO_HZ, HR_HI_HZ

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
             "face_landmarker/float16/1/face_landmarker.task")

PANEL_W = 440
COL = {"g": (80, 220, 80), "c": (230, 200, 60), "y": (60, 220, 240), "r": (70, 70, 240),
       "w": (235, 235, 235), "d": (110, 110, 110), "bg": (22, 22, 22)}


# ----------------------------------------------------------------------------
# Setup helpers
# ----------------------------------------------------------------------------

def ensure_model(path):
    if not os.path.exists(path):
        print(f"Downloading FaceLandmarker model -> {path}")
        urllib.request.urlretrieve(MODEL_URL, path)


def open_webcam(index, w, h, fps):
    backend = {"win32": cv2.CAP_DSHOW, "darwin": cv2.CAP_AVFOUNDATION}.get(sys.platform, cv2.CAP_V4L2)
    cap = cv2.VideoCapture(index, backend)
    if not cap.isOpened():
        cap = cv2.VideoCapture(index)                     # fallback: default backend
    if not cap.isOpened():
        sys.exit(f"Could not open camera {index}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    cap.set(cv2.CAP_PROP_FPS, fps)
    for _ in range(30):                                   # let AE/AWB settle first
        cap.read()
    return cap


def lock_exposure(cap):
    """
    Best effort. Laptop webcams vary a lot; the live 'lum drift' readout tells you
    whether it actually held. POS tolerates some AE drift, green-only does not.
    """
    if sys.platform == "darwin":
        return "macOS: OpenCV can't lock AE (POS still OK)"
    exp = cap.get(cv2.CAP_PROP_EXPOSURE)
    if sys.platform == "win32":
        ok = cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)     # DSHOW: 0.25 = manual
    else:
        ok = cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)        # V4L2 menu: 1 = manual
    cap.set(cv2.CAP_PROP_EXPOSURE, exp)                    # hold settled value
    cap.set(cv2.CAP_PROP_AUTO_WB, 0)
    return f"AE lock {'requested' if ok else 'unsupported'} (exp={exp:g})"


# ----------------------------------------------------------------------------
# Drawing
# ----------------------------------------------------------------------------

def txt(img, s, xy, scale=0.5, color=COL["w"], thick=1):
    cv2.putText(img, s, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def draw_trace(img, rect, y, color, label):
    x0, y0, w, h = rect
    cv2.rectangle(img, (x0, y0), (x0 + w, y0 + h), COL["d"], 1)
    txt(img, label, (x0 + 5, y0 + 15), 0.42, COL["d"])
    y = np.asarray(y, float)
    if len(y) < 2 or not np.all(np.isfinite(y)):
        return
    lo, hi = np.percentile(y, 1), np.percentile(y, 99)
    if hi - lo < 1e-12:
        return
    ys = np.clip((y - lo) / (hi - lo), 0, 1)
    xs = np.linspace(x0 + 2, x0 + w - 2, len(y))
    pts = np.column_stack([xs, y0 + h - 4 - ys * (h - 22)]).astype(np.int32)
    cv2.polylines(img, [pts], False, color, 1, cv2.LINE_AA)


def draw_spectrum(img, rect, freqs, psd, bpm, snr_ok):
    x0, y0, w, h = rect
    cv2.rectangle(img, (x0, y0), (x0 + w, y0 + h), COL["d"], 1)
    f_lo, f_hi = 0.5, 4.0
    fx = lambda f: int(x0 + (f - f_lo) / (f_hi - f_lo) * w)
    # passband shading + ticks
    ov = img.copy()
    cv2.rectangle(ov, (fx(HR_LO_HZ), y0 + 1), (fx(HR_HI_HZ), y0 + h - 1), (45, 45, 45), -1)
    cv2.addWeighted(ov, 0.6, img, 0.4, 0, img)
    txt(img, "spectrum (BPM)  shaded = search band", (x0 + 5, y0 + 15), 0.42, COL["d"])
    for b in (60, 90, 120, 150, 180, 210):
        x = fx(b / 60)
        cv2.line(img, (x, y0 + h - 4), (x, y0 + h), COL["d"], 1)
        txt(img, str(b), (x - 10, y0 + h + 14), 0.38, COL["d"])
    if freqs is None:
        return
    m = (freqs >= f_lo) & (freqs <= f_hi)
    p = psd[m] / (psd[m].max() + 1e-20)
    pts = np.column_stack([[fx(f) for f in freqs[m]], y0 + h - 3 - p * (h - 22)]).astype(np.int32)
    cv2.polylines(img, [pts], False, COL["y"], 1, cv2.LINE_AA)
    if bpm is not None:
        x = fx(bpm / 60)
        cv2.line(img, (x, y0 + 18), (x, y0 + h), COL["g"] if snr_ok else COL["r"], 1)
        x2 = fx(2 * bpm / 60)                              # harmonic marker
        if x2 < x0 + w:
            cv2.line(img, (x2, y0 + h - 12), (x2, y0 + h), COL["d"], 1)


def build_panel(h, st):
    p = np.full((h, PANEL_W, 3), COL["bg"], np.uint8)
    bpm_txt = f"{st['bpm_disp']:.0f}" if st["bpm_disp"] is not None else "--"
    txt(p, bpm_txt, (20, 70), 2.2, COL["g"] if st["confident"] else COL["w"], 3)
    txt(p, "BPM", (20 + 45 * len(bpm_txt) + 10, 70), 0.7, COL["d"], 2)
    snr = st["snr"]
    txt(p, f"SNR {snr:+.1f} dB" if snr is not None else "SNR --", (260, 35), 0.55,
        COL["g"] if st["snr_ok"] else COL["r"])
    txt(p, f"method {st['method'].upper()}", (260, 58), 0.5, COL["c"])
    txt(p, f"fps {st['fps']:.1f}", (260, 80), 0.5, COL["w"])

    res = st["res"]
    rgb = res["rgb_u"][:, 1] if res is not None else st["raw_g"]
    draw_trace(p, (10, 100, PANEL_W - 20, 80), rgb, COL["g"], "raw ROI green (steps = auto-exposure)")
    draw_trace(p, (10, 190, PANEL_W - 20, 90), res["h"] if res is not None else [], COL["c"],
               "pulse signal (after POS/green + bandpass)")
    draw_spectrum(p, (10, 290, PANEL_W - 20, 110),
                  res["freqs"] if res is not None else None,
                  res["psd"] if res is not None else None, st["bpm_raw"], st["snr_ok"])

    y = 440
    spread = st.get("spread")
    txt(p, f"agree +-{spread:.1f} BPM" if spread is not None else "agree --", (260, 102), 0.5,
        COL["g"] if st["confident"] else COL["d"])
    txt(p, st["status"], (10, y), 0.55, st["status_col"]); y += 20
    txt(p, f"lum drift (2s): {st['lum_drift']:.2f}%   face {st['face_w']:.0f}px", (10, y), 0.45); y += 18
    txt(p, st["lock_msg"], (10, y), 0.42, COL["d"]); y += 18
    if st["rec"]:
        txt(p, "REC", (PANEL_W - 55, y - 36), 0.6, COL["r"], 2)
    return p


# ----------------------------------------------------------------------------
# Live loop
# ----------------------------------------------------------------------------

def run_live(a):
    ensure_model(a.model)
    cap = open_webcam(a.cam, a.width, a.height, a.fps)
    lock_msg = lock_exposure(cap) if not a.no_lock else "AE lock disabled"
    roi = FaceROI(a.model)

    lum_hist = deque()
    method = a.method
    last_nose, bad_run = None, 0
    next_an = 0.0
    fps, t_last = 0.0, time.monotonic()
    writer, rec_file = None, None
    status, status_col, face_w = "starting", COL["w"], 0.0

    def new_tracker():
        """The robot's tracker, same settings: buffer, SNR gate, tracking, aggregation."""
        return HeartRateTracker(fs=a.fs, window_s=a.window, snr_min_db=a.snr_min, method=method)

    tracker = new_tracker()

    def reset():
        nonlocal tracker, last_nose, bad_run
        tracker = new_tracker()
        last_nose, bad_run = None, 0

    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        frame = cv2.flip(frame, 1)                         # mirror view; irrelevant to signal
        now = time.monotonic()
        dt = now - t_last; t_last = now
        fps = 0.9 * fps + 0.1 * (1.0 / dt if dt > 0 else 0)

        r = roi(frame, now * 1000)
        if r is None:
            if tracker.t:
                reset()
            status, status_col = "NO FACE / too far", COL["r"]
        else:
            rgb, nose, face_w, mask = r
            moved = last_nose is not None and np.linalg.norm(nose - last_nose) / face_w > a.motion_max
            last_nose = nose
            if moved:
                bad_run += 1
                status, status_col = "MOTION - hold still", COL["r"]
                if bad_run > int(0.5 * a.fs):
                    tracker.clear(unlock=True)
            else:
                bad_run = 0
                tracker.add(now, rgb)
                span = now - tracker.t[0]
                status, status_col = ("measuring", COL["g"]) if span >= MIN_SECONDS else \
                                     (f"buffering {span:.1f}/{MIN_SECONDS:.0f} s", COL["y"])
                if writer:
                    writer.writerow([f"{now:.4f}", *(f"{v:.4f}" for v in rgb)])

            # luminance stability readout: exposes AE hunting
            lum_hist.append((now, float(np.dot(rgb, [0.299, 0.587, 0.114]))))
            # overlay ROI
            tint = frame.copy(); tint[mask > 0] = (0, 255, 0)
            cv2.addWeighted(tint, 0.25, frame, 0.75, 0, frame)
            cv2.circle(frame, tuple(nose.astype(int)), 3, COL["c"], -1)

        while lum_hist and now - lum_hist[0][0] > 2.0:
            lum_hist.popleft()
        lv = np.array([v for _, v in lum_hist]) if lum_hist else np.array([1.0])
        lum_drift = 100 * np.ptp(lv) / (lv.mean() + 1e-9)

        if now >= next_an and len(tracker.t) > 2:
            next_an = now + 0.5
            tracker.update(now)                            # gate, tracking and unlock live here

        agg = tracker.result()
        snr = tracker.last_snr
        st = dict(bpm_disp=None if agg is None else agg["bpm"], bpm_raw=tracker.last_bpm, snr=snr,
                  snr_ok=snr is not None and snr >= a.snr_min,
                  confident=bool(agg and agg["confident"]),
                  spread=None if agg is None else agg["spread_bpm"],
                  method=method, fps=fps, res=tracker.last,
                  raw_g=[c[1] for c in tracker.rgb], status=status, status_col=status_col,
                  lum_drift=lum_drift, face_w=face_w, lock_msg=lock_msg, rec=writer is not None)
        canvas = np.hstack([frame, build_panel(frame.shape[0], st)]) \
            if frame.shape[0] >= 500 else \
            np.hstack([cv2.resize(frame, (int(frame.shape[1] * 500 / frame.shape[0]), 500)),
                       build_panel(500, st)])
        cv2.imshow("rPPG concept check", canvas)

        k = cv2.waitKey(1) & 0xFF
        if k in (ord("q"), 27):
            break
        elif k == ord("m"):
            method = "green" if method == "pos" else "pos"; reset()
        elif k == ord(" "):
            reset()
        elif k == ord("l"):
            lock_msg = lock_exposure(cap)
        elif k == ord("r"):
            if writer:
                rec_file.close(); writer = None
                print(f"saved {rec_path}")
            else:
                rec_path = time.strftime("rppg_%Y%m%d_%H%M%S.csv")
                rec_file = open(rec_path, "w", newline="")
                writer = csv.writer(rec_file); writer.writerow(["t", "R", "G", "B"])

    if writer:
        rec_file.close(); print(f"saved {rec_path}")
    cap.release(); cv2.destroyAllWindows()


# ----------------------------------------------------------------------------
# Offline replay: re-run analysis on a recording with different parameters
# ----------------------------------------------------------------------------

def run_replay(a):
    d = np.genfromtxt(a.replay, delimiter=",", names=True)
    t = d["t"] - d["t"][0]
    rgb = np.column_stack([d["R"], d["G"], d["B"]])
    print(f"{a.replay}: {len(t)} frames, {t[-1]:.1f} s, mean fps {len(t)/t[-1]:.1f}")
    print(f"{'t(s)':>6} {'POS':>7} {'snr':>6}   {'GREEN':>7} {'snr':>6}")
    trackers = {m: HeartRateTracker(fs=a.fs, window_s=a.window, snr_min_db=a.snr_min, method=m)
                for m in ("pos", "green")}
    i = 0
    for te in np.arange(MIN_SECONDS, t[-1] + 1e-9, 1.0):
        while i < len(t) and t[i] <= te:
            for tr in trackers.values():
                tr.add(t[i], rgb[i])
            i += 1
        row = [f"{te:6.1f}"]
        for meth in ("pos", "green"):
            r = trackers[meth].update(te)
            if r is None:
                row += [f"{'--':>7}", f"{'':>6}"]; continue
            ok = r["snr_db"] >= a.snr_min
            row += [f"{r['bpm']:7.1f}", f"{r['snr_db']:+6.1f}" + ("" if ok else "*")]
        print(" ".join(row[:3]), "  ", " ".join(row[3:]))
    for meth, tr in trackers.items():
        r = tr.result()
        if r:
            print(f"{meth.upper():>5}: {r['bpm']:.1f} BPM over {r['n_estimates']} accepted windows"
                  f"  spread {r['spread_bpm']} BPM  SNR {r['snr_db']:+.1f} dB"
                  f"  {'confident' if r['confident'] else 'NOT confident'}")
    print("* = rejected by SNR gate")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--fs", type=float, default=30.0, help="resample rate (Hz)")
    ap.add_argument("--window", type=float, default=10.0, help="analysis window (s)")
    ap.add_argument("--snr-min", type=float, default=-1.0)
    ap.add_argument("--motion-max", type=float, default=0.03)
    ap.add_argument("--method", choices=["pos", "green"], default="pos")
    ap.add_argument("--model", default=str(DEFAULT_MODEL))
    ap.add_argument("--no-lock", action="store_true", help="leave auto-exposure on")
    ap.add_argument("--replay", help="CSV from 'r' key to re-analyze offline")
    a = ap.parse_args()
    run_replay(a) if a.replay else run_live(a)
