"""
rppg_view.py - the debug picture for the rPPG chain: what the camera sees, and
what the model does to it.

`laptop_rppg.py` draws it into an OpenCV window and `scripts/robot_rppg.py
--view` serves it as MJPEG from the robot, both through this module, so the
picture on the robot is the same picture as the one on the laptop and a
threshold tuned against one is tuned against the other.

The left half is the frame with the model's own working drawn on top: the
detector's raw box, the steadied box the ROI actually used, the three skin
regions it offered, and which pixels inside them survived the skin gate. The
right half is the signal: the raw ROI green, the pulse after POS and the
bandpass, this window's spectrum, and the stacked spectrum of the whole scan
that the reported rate is read off.

NOT a medical device. See docs/rppg-robot-port.md.
"""
import numpy as np

from rppg import HR_LO_HZ, HR_HI_HZ, REGION_NAMES

PANEL_W = 440
COL = {"g": (80, 220, 80), "c": (230, 200, 60), "y": (60, 220, 240), "r": (70, 70, 240),
       "w": (235, 235, 235), "d": (110, 110, 110), "bg": (22, 22, 22),
       "kept": (90, 230, 120), "drop": (70, 70, 240), "box": (230, 200, 60),
       "steady": (60, 220, 240)}


def txt(img, s, xy, scale=0.5, color=COL["w"], thick=1):
    import cv2
    cv2.putText(img, s, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


# ----------------------------------------------------------------------------
# The frame, with the model's working drawn on it
# ----------------------------------------------------------------------------

def draw_overlay(frame, detail, nose=None):
    """Draw one FaceROI.last onto a copy of the frame.

    Green is the skin the mean was actually taken over; red is what the gate
    threw away - hair over the forehead, a specular highlight, a cheek edge
    that caught background. Seeing the red is the point: it is the difference
    between a clean trace and one diluted by pixels that carry no pulse.
    """
    import cv2

    out = frame.copy()
    if not detail:
        return out

    mask = detail.get("mask")
    offered = detail.get("region_mask")
    if mask is not None and offered is not None:
        dropped = (offered > 0) & (mask == 0)
        tint = out.copy()
        tint[mask > 0] = COL["kept"]
        tint[dropped] = COL["drop"]
        touched = (offered > 0)
        out[touched] = cv2.addWeighted(tint, 0.35, out, 0.65, 0)[touched]

    box = detail.get("box")
    steady = detail.get("steady_box")
    if box:
        x, y, w, h = (int(round(v)) for v in box)
        cv2.rectangle(out, (x, y), (x + w, y + h), COL["box"], 1)
    if steady and steady != box:
        x, y, w, h = (int(round(v)) for v in steady)
        cv2.rectangle(out, (x, y), (x + w, y + h), COL["steady"], 2)
    for point in detail.get("landmarks") or ():
        cv2.circle(out, (int(point[0]), int(point[1])), 2, COL["c"], -1)
    if nose is not None:
        cv2.circle(out, tuple(np.asarray(nose).astype(int)), 3, COL["c"], -1)

    if box:
        x, y = int(round(box[0])), int(round(box[1]))
        label = f"score {detail.get('score', 0):.2f}  {detail.get('pixels', 0)} px"
        skin_frac = detail.get("skin_frac")
        if skin_frac is not None:
            label += f"  skin {skin_frac * 100:.0f}%"
        txt(out, label, (x, max(14, y - 6)), 0.45, COL["w"])
    return out


# ----------------------------------------------------------------------------
# The signal panel
# ----------------------------------------------------------------------------

def draw_trace(img, rect, y, color, label):
    import cv2

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


def draw_spectrum(img, rect, freqs, psd, bpm, snr_ok, label="spectrum (BPM)  shaded = search band",
                  color=None):
    import cv2

    x0, y0, w, h = rect
    cv2.rectangle(img, (x0, y0), (x0 + w, y0 + h), COL["d"], 1)
    f_lo, f_hi = 0.5, 4.0
    fx = lambda f: int(x0 + (f - f_lo) / (f_hi - f_lo) * w)
    ov = img.copy()
    cv2.rectangle(ov, (fx(HR_LO_HZ), y0 + 1), (fx(HR_HI_HZ), y0 + h - 1), (45, 45, 45), -1)
    cv2.addWeighted(ov, 0.6, img, 0.4, 0, img)
    txt(img, label, (x0 + 5, y0 + 15), 0.42, COL["d"])
    for b in (60, 90, 120, 150, 180, 210):
        x = fx(b / 60)
        cv2.line(img, (x, y0 + h - 4), (x, y0 + h), COL["d"], 1)
        txt(img, str(b), (x - 10, y0 + h + 14), 0.38, COL["d"])
    if freqs is None or psd is None:
        return
    freqs, psd = np.asarray(freqs, float), np.asarray(psd, float)
    m = (freqs >= f_lo) & (freqs <= f_hi)
    if not m.any() or psd[m].max() <= 0:
        return
    p = psd[m] / (psd[m].max() + 1e-20)
    pts = np.column_stack([[fx(f) for f in freqs[m]], y0 + h - 3 - p * (h - 22)]).astype(np.int32)
    cv2.polylines(img, [pts], False, color or COL["y"], 1, cv2.LINE_AA)
    if bpm is not None:
        x = fx(bpm / 60)
        cv2.line(img, (x, y0 + 18), (x, y0 + h), COL["g"] if snr_ok else COL["r"], 1)
        x2 = fx(2 * bpm / 60)                              # harmonic marker
        if x2 < x0 + w:
            cv2.line(img, (x2, y0 + h - 12), (x2, y0 + h), COL["d"], 1)


def build_panel(h, st):
    """The right-hand readout. `st` is the state dict the callers assemble.

    Two SNRs appear, and they are not the same number. `scan` is the stacked
    spectrum's, and it is the one the reported rate is gated on; `window` is
    the current window's, and it is the one that decides whether this window
    joins the accepted estimates. A scan reads well when the second is noisy
    and the first is not.
    """
    p = np.full((h, PANEL_W, 3), COL["bg"], np.uint8)
    bpm_txt = f"{st['bpm_disp']:.0f}" if st.get("bpm_disp") is not None else "--"
    txt(p, bpm_txt, (20, 62), 2.0, COL["g"] if st.get("confident") else COL["w"], 3)
    txt(p, "BPM", (20 + 40 * len(bpm_txt) + 10, 62), 0.62, COL["d"], 2)
    txt(p, "confident" if st.get("confident") else "not confident", (22, 86), 0.46,
        COL["g"] if st.get("confident") else COL["d"])

    scan_snr, snr = st.get("scan_snr"), st.get("snr")
    txt(p, f"scan SNR {scan_snr:+.1f} dB" if scan_snr is not None else "scan SNR --",
        (238, 30), 0.52, COL["g"] if st.get("confident") else COL["w"])
    txt(p, f"window SNR {snr:+.1f} dB" if snr is not None else "window SNR --", (238, 50), 0.46,
        COL["g"] if st.get("snr_ok") else COL["r"])
    margin = st.get("margin")
    txt(p, f"beats rival {margin:+.1f} dB" if margin is not None else "beats rival --",
        (238, 68), 0.46, COL["g"] if st.get("confident") else COL["d"])
    spread = st.get("spread")
    txt(p, f"windows agree +-{spread:.1f}" if spread is not None else "windows agree --",
        (238, 86), 0.46, COL["g"] if st.get("confident") else COL["d"])

    res = st.get("res")
    rgb = res["rgb_u"][:, 1] if res is not None else st.get("raw_g", [])
    draw_trace(p, (10, 100, PANEL_W - 20, 72), rgb, COL["g"],
               "raw ROI green (steps = an exposure move)")
    draw_trace(p, (10, 180, PANEL_W - 20, 78), res["h"] if res is not None else [], COL["c"],
               "pulse signal (after POS/green + bandpass)")
    draw_spectrum(p, (10, 268, PANEL_W - 20, 92),
                  res["freqs"] if res is not None else None,
                  res["psd"] if res is not None else None, st.get("bpm_raw"), st.get("snr_ok"),
                  label="this window's spectrum")
    # The stacked spectrum is what the reported rate is actually read off: one
    # window can be beaten by noise, the stack of them usually is not.
    draw_spectrum(p, (10, 378, PANEL_W - 20, 92), st.get("stack_freqs"), st.get("stack_psd"),
                  st.get("bpm_disp"), st.get("snr_ok"),
                  label=f"whole scan, {st.get('n_windows', 0)} windows stacked", color=COL["g"])

    y = 502
    txt(p, st.get("status", ""), (10, y), 0.55, st.get("status_col", COL["w"])); y += 22
    # Luminance drift needs a camera loop to track it; the robot's view has no
    # such readout, so the line says what it knows rather than printing a zero.
    bits = []
    if st.get("lum_drift") is not None:
        bits.append(f"lum drift (2s): {st['lum_drift']:.2f}%")
    # Clipped skin is worth its own place in the readout: those pixels had their
    # modulation thrown away by the sensor, so no amount of filtering gets it back.
    if st.get("clip_hi") or st.get("clip_lo"):
        bits.append(f"clip {100 * (st.get('clip_hi') or 0):.0f}/"
                    f"{100 * (st.get('clip_lo') or 0):.0f}%")
    if st.get("face_w"):
        bits.append(f"face {st['face_w']:.0f}px")
    bits.append(f"{st.get('fps', 0.0):.1f} fps")
    bits.append(st.get("method", "pos").upper())
    txt(p, "   ".join(bits), (10, y), 0.44); y += 19
    detail = st.get("detail") or {}
    if detail:
        txt(p, f"skin kept {detail.get('pixels', 0)} px "
               f"({100 * detail.get('skin_frac', 0.0):.0f}% of the ROI offered)",
            (10, y), 0.44, COL["kept"])
        y += 17
        parts = detail.get("regions") or {}
        counts = "  ".join(f"{name.split('-')[-1][:4]} {parts.get(name, {}).get('pixels', 0)}"
                           for name in REGION_NAMES)
        txt(p, counts, (10, y), 0.42, COL["d"]); y += 19
    if st.get("lock_msg"):
        txt(p, st["lock_msg"], (10, y), 0.42, COL["d"]); y += 17
    if st.get("note"):
        txt(p, st["note"], (10, y), 0.42, st.get("note_col") or COL["d"]); y += 17
    if st.get("rec"):
        txt(p, "REC", (PANEL_W - 55, 62), 0.6, COL["r"], 2)
    return p


def compose(frame, st, min_height=620):
    """Frame plus panel, side by side, as one BGR image."""
    import cv2

    if frame is None:
        frame = np.full((min_height, min_height * 4 // 3, 3), COL["bg"], np.uint8)
    h = max(frame.shape[0], min_height)
    if frame.shape[0] != h:
        frame = cv2.resize(frame, (int(frame.shape[1] * h / frame.shape[0]), h))
    return np.hstack([frame, build_panel(h, st)])


def state_from(tracker, *, frame_fps=0.0, status="", status_col=None, **extra):
    """Assemble the panel state from a HeartRateTracker, so callers agree on the readout."""
    agg = tracker.result()
    snr = tracker.last_snr
    stack = tracker.stack.spectrum
    st = {"bpm_disp": None if agg is None else agg["bpm"],
          "bpm_raw": tracker.last_bpm,
          "snr": snr,
          "scan_snr": None if agg is None else agg["snr_db"],
          "snr_ok": snr is not None and snr >= tracker.snr_min_db,
          "confident": bool(agg and agg["confident"]),
          "spread": None if agg is None else agg.get("spread_bpm"),
          "margin": None if agg is None else agg.get("margin_db"),
          "method": tracker.method,
          "fps": frame_fps,
          "res": tracker.last,
          "raw_g": [c[1] for c in tracker.rgb],
          "stack_freqs": None if stack is None else tracker.stack.grid,
          "stack_psd": stack,
          "n_windows": tracker.stack.n_windows,
          "status": status,
          "status_col": status_col or COL["w"]}
    st.update(extra)
    # The ROI already knows how wide the face was; a caller that tracks it itself
    # (the laptop tool does) has overwritten this through `extra` by now.
    detail = st.get("detail")
    if not st.get("face_w") and detail and detail.get("steady_box"):
        st["face_w"] = float(detail["steady_box"][2])
    return st
