"""
laptop_rppg.py - concept check for rPPG on a laptop webcam, with a live debug view.

Uses the same signal chain and the same HeartRateTracker as rppg.py that scripts/robot_rppg.py
runs on the robot's head camera, so what you tune here is what the robot measures.

NOT a medical device: a demo-grade estimate from a camera. See docs/rppg-robot-port.md.

Uses the YuNet model at assets/models/face_detection_yunet_2026may.onnx.

Setup:
  uv run --extra rppg python laptop_rppg.py                  # live, camera 0
  uv run --extra rppg python laptop_rppg.py --cam 1          # other camera
  uv run --extra rppg python laptop_rppg.py --replay rec.csv # offline re-analysis

Exposure is metered on the face, not the frame: the camera's own AE is switched off
and a slow loop steps exposure until the skin sits near the middle of the sensor's range
with nothing clipped, then holds still. Watch the "skin NN%" readout settle. Each move it
makes is a step in the raw trace, so the window it lands in is thrown away rather than
read - which is why the loop is deliberately reluctant to move at all.

Keys (live window):
  q / Esc   quit
  m         toggle method POS <-> GREEN (watch GREEN fail under lighting changes)
  g         toggle the skin gate (watch the red pixels rejoin the average)
  b         toggle box steadying
  e         toggle face-metered exposure (watch a dark face stay dark without it)
  r         start/stop recording CSV (t, R, G, B) for offline tuning
  space     reset buffer
  l         try to (re)lock exposure / white balance

Face ~50-70 cm from camera, even front lighting, hold still ~10 s for first reading.
Compare against a smartwatch / phone pulse app / MAX30102.

The 'r' recording holds someone's pulse trace: treat the CSV as personal data (gitignored).
"""
import argparse
import csv
from pathlib import Path
import sys
import time
from collections import deque

import cv2
import numpy as np

import rppg_view
from rppg import (DEFAULT_MODEL, EXPOSURE_TARGET, FaceExposure, FaceROI,
                  HeartRateTracker, MIN_SECONDS, OpenCVExposure, skin_luma_stats)
from rppg_view import COL


# ----------------------------------------------------------------------------
# Setup helpers
# ----------------------------------------------------------------------------

def ensure_model(path):
    if not Path(path).is_file():
        sys.exit(
            f"YuNet face model not found at {path}; restore "
            "assets/models/face_detection_yunet_2026may.onnx"
        )


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
    return cap


# ----------------------------------------------------------------------------
# Live loop
# ----------------------------------------------------------------------------

def run_live(a):
    ensure_model(a.model)
    cap = open_webcam(a.cam, a.width, a.height, a.fps)
    # Taking the camera off its own AE and then steering exposure from the face are
    # the same job split in two, and both halves live in rppg.py, so the robot runs
    # the loop this tool is tuned against. Best effort either way: webcams vary a
    # lot, and the "lum drift" and "skin NN%" readouts are how you see whether this
    # camera honoured any of it.
    exposure = OpenCVExposure(cap, fps=a.fps)
    exposure.auto()                                       # undo a lock a previous run left behind
    lock_msg = exposure.manual() if not a.no_lock else "AE lock disabled (camera AE still on)"
    # Metering the face is only meaningful once the camera has stopped metering the
    # frame; with --no-lock the loop would be arguing with the camera's own AE.
    metering_possible = exposure.locked and not a.no_meter
    meter = FaceExposure(target=a.exposure_target) if metering_possible else None
    roi = FaceROI(a.model)

    lum_hist = deque()
    method = a.method
    last_nose, bad_run = None, 0
    next_an = 0.0
    fps, t_last = 0.0, time.monotonic()
    writer, rec_file = None, None
    status, status_col, face_w = "starting", COL["w"], 0.0
    skin = None

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
            if meter:
                meter.observe(now, None)
            skin = None
            status, status_col = "NO FACE / too far", COL["r"]
        else:
            rgb, nose, face_w, mask = r
            skin = skin_luma_stats(frame, mask)
            if meter:
                factor = meter.observe(now, skin)
                if factor is not None and exposure.apply(factor):
                    # The step dwarfs the pulse: keep the lock, drop the window it
                    # would have landed in.
                    tracker.clear()
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
            frame = rppg_view.draw_overlay(frame, roi.last, nose)

        while lum_hist and now - lum_hist[0][0] > 2.0:
            lum_hist.popleft()
        lv = np.array([v for _, v in lum_hist]) if lum_hist else np.array([1.0])
        lum_drift = 100 * np.ptp(lv) / (lv.mean() + 1e-9)

        if now >= next_an and len(tracker.t) > 2:
            next_an = now + 0.5
            tracker.update(now)                            # gate, tracking and unlock live here

        st = rppg_view.state_from(tracker, frame_fps=fps, status=status, status_col=status_col,
                                  lum_drift=lum_drift, face_w=face_w,
                                  clip_hi=None if skin is None else skin["clip_hi"],
                                  clip_lo=None if skin is None else skin["clip_lo"],
                                  lock_msg=lock_msg,
                                  note=meter.describe() if meter else "face metering off",
                                  note_col=COL["g"] if meter and meter.settled else COL["d"],
                                  detail=roi.last, rec=writer is not None)
        cv2.imshow("rPPG concept check", rppg_view.compose(frame, st))

        k = cv2.waitKey(1) & 0xFF
        if k in (ord("q"), 27):
            break
        elif k == ord("m"):
            method = "green" if method == "pos" else "pos"; reset()
        elif k == ord("g"):
            roi.skin_gate = not roi.skin_gate; reset()
        elif k == ord("b"):
            roi.steady = not roi.steady; roi.box = None; reset()
        elif k == ord("e"):
            meter = None if meter or not metering_possible \
                else FaceExposure(target=a.exposure_target)
        elif k == ord(" "):
            reset()
        elif k == ord("l"):
            lock_msg = exposure.manual()
            metering_possible = exposure.locked and not a.no_meter
            meter = FaceExposure(target=a.exposure_target) if metering_possible else None
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
    exposure.auto(settle_frames=0)                        # leave the webcam usable for other apps
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
    ap.add_argument("--no-lock", action="store_true",
                    help="leave the camera's own auto-exposure on (also disables face metering)")
    ap.add_argument("--no-meter", action="store_true",
                    help="lock exposure where it settled, without metering it on the face")
    ap.add_argument("--exposure-target", type=float, default=EXPOSURE_TARGET,
                    help="skin brightness to aim for, as a fraction of full scale")
    ap.add_argument("--replay", help="CSV from 'r' key to re-analyze offline")
    a = ap.parse_args()
    run_replay(a) if a.replay else run_live(a)
