# /// script
# requires-python = "==3.10.*"
# dependencies = [
#   "bbos",
#   "numpy",
#   "scipy",
#   "opencv-python",
# ]
# [tool.uv.sources]
# bbos = { path = "/home/bracketbot/bbos", editable = true }
# ///
"""Read-only browser preview of the pixels used by the robot rPPG scan.

The green overlay is the exact forehead-and-cheek mask whose mean RGB values
feed the heart-rate estimator. Frames are kept only as the latest in-memory
JPEG and are never written to disk.
"""

from __future__ import annotations

import argparse
from collections import deque
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
import time

import cv2
import numpy as np

from robot_rppg import (
    MAX_LUM_DRIFT_PCT, MIN_CAMERA_FPS, MIN_FACE_FRACTION, MIN_FACE_WIDTH_PX, HeadCamera,
)
from rppg import DEFAULT_MODEL, HR_HI_HZ, HR_LO_HZ, MIN_SECONDS, FaceROI, HeartRateTracker

STREAM_FPS = 8.0          # annotate/encode rate; every frame is still sampled for the pulse
RATE_WINDOW_S = 3.0       # rolling window behind the fps / face / motion percentages


PAGE = b"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Heartbeat camera preview</title>
<style>
body{margin:0;background:#07100b;color:#e8fff1;font:16px system-ui,sans-serif}
main{max-width:1400px;margin:auto;padding:20px}h1{margin:.2rem 0;font-size:1.45rem}
h2{font-size:1rem;margin:1rem 0 .4rem;color:#a9cbb5;font-weight:600}
p{color:#a9cbb5;margin:.4rem 0 1rem}.card{background:#0d1b13;border:1px solid #214c31;border-radius:14px;padding:12px}
img{display:block;width:100%;height:auto;border-radius:9px;background:#000}strong{color:#70dc78}
.grid{display:grid;grid-template-columns:minmax(0,3fr) minmax(320px,2fr);gap:16px}
@media(max-width:900px){.grid{grid-template-columns:1fr}}
#bpm{font-size:3rem;font-weight:700;line-height:1}#phase{color:#a9cbb5;margin:.3rem 0 .6rem}
.bar{height:10px;background:#16301f;border-radius:5px;overflow:hidden}.bar div{height:100%;background:#70dc78;width:0}
table{width:100%;border-collapse:collapse;font-size:.92rem}td{padding:3px 0;border-bottom:1px solid #173323}
td:nth-child(2),td:nth-child(3){text-align:right;font-variant-numeric:tabular-nums}td:nth-child(3){color:#7fa08b;padding-left:10px}
.ok{color:#70dc78}.bad{color:#ff7a5c}.warn{color:#ffc04d}
ul{margin:.2rem 0;padding-left:1.1rem}li{margin:.25rem 0}canvas{width:100%;height:90px;display:block;background:#08130d;border-radius:8px}
</style></head><body><main><h1>Heartbeat camera preview</h1>
<p><strong>Green pixels</strong> are the exact forehead and cheek samples used for pulse estimation. The dot is the tracked nose.
The panel runs the same gates as the real scan, continuously.</p>
<div class="grid"><div class="card"><img src="/stream.mjpg" alt="Live annotated left-eye camera view"></div>
<div class="card"><div id="bpm">--</div><div id="phase">connecting...</div><div class="bar"><div id="fill"></div></div>
<h2>Problems right now</h2><ul id="problems"></ul>
<h2>Gates</h2><table id="gates"></table>
<h2>Pulse signal (last window)</h2><canvas id="wave" width="600" height="90"></canvas>
<h2>Spectrum 42-180 BPM (line = picked peak)</h2><canvas id="spec" width="600" height="90"></canvas>
</div></div></main><script>
function plot(id,ys,mark){const c=document.getElementById(id),g=c.getContext('2d');g.clearRect(0,0,c.width,c.height);
if(!ys||ys.length<2)return;const lo=Math.min(...ys),hi=Math.max(...ys),r=(hi-lo)||1;g.strokeStyle='#70dc78';g.lineWidth=1.5;g.beginPath();
ys.forEach((y,i)=>{const px=i/(ys.length-1)*c.width,py=c.height-4-(y-lo)/r*(c.height-8);i?g.lineTo(px,py):g.moveTo(px,py)});g.stroke();
if(mark!=null){g.strokeStyle='#ffc04d';g.beginPath();g.moveTo(mark*c.width,0);g.lineTo(mark*c.width,c.height);g.stroke()}}
async function tick(){try{const s=await (await fetch('/state',{cache:'no-store'})).json();
document.getElementById('bpm').textContent=s.headline;document.getElementById('bpm').className=s.headline_class;
document.getElementById('phase').textContent=s.phase;document.getElementById('fill').style.width=(100*s.progress)+'%';
document.getElementById('problems').innerHTML=s.problems.length?s.problems.map(p=>'<li class="bad">'+p+'</li>').join(''):'<li class="ok">none - hold still</li>';
document.getElementById('gates').innerHTML=s.gates.map(g=>'<tr><td>'+g[0]+'</td><td class="'+(g[3]===null?'':g[3]?'ok':'bad')+'">'+g[1]+'</td><td>'+g[2]+'</td></tr>').join('');
plot('wave',s.wave,null);plot('spec',s.psd,s.peak);}catch(e){document.getElementById('phase').textContent='preview server not answering';}
setTimeout(tick,500)}tick();
</script></body></html>"""


class Preview:
    def __init__(self, model: Path, max_width: int = 960):
        self.model = model
        self.max_width = max_width
        self.condition = threading.Condition()
        self.sequence = 0
        self.jpeg: bytes | None = None
        self.error: str | None = None
        self.started = time.monotonic()
        self.frames = 0
        self.state: dict = {"headline": "--", "headline_class": "", "phase": "starting camera...",
                            "progress": 0.0, "problems": [], "gates": [], "wave": [], "psd": [],
                            "peak": None}

    def publish(self, jpeg: bytes) -> None:
        with self.condition:
            self.sequence += 1
            self.jpeg = jpeg
            self.condition.notify_all()

    def wait_after(self, sequence: int, timeout: float = 2.0) -> tuple[int, bytes | None]:
        with self.condition:
            self.condition.wait_for(
                lambda: self.sequence != sequence or self.error is not None,
                timeout=timeout,
            )
            return self.sequence, self.jpeg

    def annotate(self, frame: np.ndarray, result, forehead_covered: bool) -> np.ndarray:
        output = frame.copy()
        status = "NO FACE - move into the left-eye view"
        colour = (40, 70, 255)
        if result is not None:
            _rgb, nose, face_width, mask = result
            tint = output.copy()
            tint[mask > 0] = (45, 255, 70)
            output = cv2.addWeighted(tint, 0.42, output, 0.58, 0)
            cv2.circle(output, tuple(np.rint(nose).astype(int)), 7, (0, 255, 255), -1)
            sampled = int(cv2.countNonZero(mask))
            status = f"TRACKING  face {face_width:.0f}px  sampled {sampled:,} pixels"
            colour = (70, 230, 90)
            if forehead_covered:
                status = "FOREHEAD MAY BE COVERED - move hair away"
                colour = (0, 190, 255)

        elapsed = max(time.monotonic() - self.started, 1e-6)
        fps = self.frames / elapsed
        cv2.rectangle(output, (0, 0), (output.shape[1], 52), (5, 12, 8), -1)
        cv2.putText(
            output,
            f"{status}   preview {fps:.1f} FPS",
            (16, 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            colour,
            2,
            cv2.LINE_AA,
        )
        if output.shape[1] > self.max_width:
            scale = self.max_width / output.shape[1]
            output = cv2.resize(
                output,
                (self.max_width, round(output.shape[0] * scale)),
                interpolation=cv2.INTER_AREA,
            )
        return output

    def run(self) -> None:
        try:
            self.scan()
        except Exception as exc:
            with self.condition:
                self.error = str(exc)
                self.condition.notify_all()

    def scan(self) -> None:
        """measure_heart_rate's gating, run forever, with every reason a frame is refused counted.

        Kept in step with rppg.measure_heart_rate by hand: that routine is a blocking
        fixed-length scan with no hook for why a sample was dropped.
        """
        fs, snr_min_db, motion_max, face_gap_max_s = 30.0, -1.0, 0.03, 0.75
        roi = FaceROI(self.model)
        tracker = HeartRateTracker(fs=fs, snr_min_db=snr_min_db)
        events: deque = deque()      # (t, kind): kind in frame/face/motion/covered/used
        widths: deque = deque()
        resets: deque = deque()      # (t, reason)
        last_nose = last_face_at = None
        bad_run = covered_run = 0
        next_est = next_draw = 0.0
        with HeadCamera(jpeg=False) as camera:
            while True:
                frame = camera.grab()
                if frame is None:
                    continue
                now = time.monotonic()
                self.frames += 1
                events.append((now, "frame"))
                result = roi(frame, now * 1000)
                covered = bool(getattr(roi, "forehead_covered", False))

                def reset(reason):
                    nonlocal last_nose
                    if tracker.t:
                        resets.append((now, reason))
                    tracker.clear(unlock=True)
                    last_nose = None

                if result is None:
                    if last_face_at is not None and now - last_face_at > face_gap_max_s:
                        reset("face lost")
                        tracker.estimates.clear()      # a new person starts a new reading
                        last_face_at = None
                else:
                    rgb, nose, face_w, _mask = result
                    last_face_at = now
                    events.append((now, "face"))
                    widths.append((now, face_w))
                    if covered:
                        covered_run += 1
                        events.append((now, "covered"))
                        if covered_run >= max(3, int(0.35 * fs)):
                            reset("forehead covered")
                    else:
                        covered_run = 0
                        moved = (last_nose is not None
                                 and np.linalg.norm(nose - last_nose) / face_w > motion_max)
                        last_nose = nose
                        if moved:
                            bad_run += 1
                            events.append((now, "motion"))
                            if bad_run > int(0.5 * fs):
                                reset("head motion")
                        else:
                            bad_run = 0
                            tracker.add(now, rgb)
                            events.append((now, "used"))

                for queue, keep in ((events, RATE_WINDOW_S), (widths, RATE_WINDOW_S), (resets, 30.0)):
                    while queue and now - queue[0][0] > keep:
                        queue.popleft()
                if now >= next_est:
                    next_est = now + 1.0
                    tracker.update(now)
                    self.state = self.describe(tracker, events, widths, resets, snr_min_db, now)
                if now >= next_draw:
                    next_draw = now + 1.0 / STREAM_FPS
                    annotated = self.annotate(frame, result, covered)
                    ok, encoded = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    if ok:
                        self.publish(encoded.tobytes())

    @staticmethod
    def describe(tracker, events, widths, resets, snr_min_db, now) -> dict:
        count = lambda kind: sum(1 for _, k in events if k == kind)  # noqa: E731
        frames, faces = count("frame"), count("face")
        span = max(now - events[0][0], 1e-6) if events else 1e-6
        fps = frames / span if span > 1.0 else 0.0
        face_pct = 100.0 * faces / frames if frames else 0.0
        motion_pct = 100.0 * count("motion") / faces if faces else 0.0
        covered_pct = 100.0 * count("covered") / faces if faces else 0.0
        width = float(np.median([w for _, w in widths])) if widths else None
        buffer_s = tracker.t[-1] - tracker.t[0] if len(tracker.t) > 1 else 0.0
        lum_drift = None
        if len(tracker.rgb) > 1:
            lum = np.asarray(tracker.rgb) @ [0.299, 0.587, 0.114]
            lum_drift = 100.0 * float(np.ptp(lum)) / (float(np.mean(lum)) + 1e-9)
        snr = tracker.last_snr
        result = tracker.result()
        accepted = len(tracker.estimates)
        est_span = tracker.estimates[-1][0] - tracker.estimates[0][0] if accepted else 0.0

        problems = []
        if fps and fps < MIN_CAMERA_FPS:
            problems.append(f"Only {fps:.0f} camera frames/s reach the scan (needs {MIN_CAMERA_FPS:g}+): robot CPU is overloaded.")
        if not faces:
            problems.append("No face in the left-eye view.")
        elif face_pct < 100 * MIN_FACE_FRACTION:
            problems.append(f"Face found in only {face_pct:.0f}% of frames (needs {MIN_FACE_FRACTION:.0%}).")
        if width is not None and width < MIN_FACE_WIDTH_PX:
            problems.append(f"Face is {width:.0f}px wide (needs {MIN_FACE_WIDTH_PX:g}+): too few skin pixels, move closer.")
        if covered_pct > 30:
            problems.append("Forehead looks covered (hair/hat): those frames are thrown away.")
        if motion_pct > 20:
            problems.append(f"{motion_pct:.0f}% of frames dropped for head motion: hold still (or the robot is swaying).")
        if len(resets) >= 2:
            reasons = ", ".join(sorted({r for _, r in resets}))
            problems.append(f"Buffer restarted {len(resets)}x in the last 30 s ({reasons}); each restart costs {MIN_SECONDS:g} s.")
        if lum_drift is not None and lum_drift > MAX_LUM_DRIFT_PCT:
            problems.append(f"Skin brightness drifts {lum_drift:.0f}% across the window (limit {MAX_LUM_DRIFT_PCT:g}%): "
                            "auto-exposure/lighting swamps the ~1% pulse signal.")
        if snr is not None and snr < snr_min_db:
            problems.append(f"Pulse peak is buried in noise: SNR {snr:.1f} dB, gate {snr_min_db:g} dB. Estimate rejected.")

        if buffer_s < MIN_SECONDS:
            phase = f"Filling buffer: {buffer_s:.1f}/{MIN_SECONDS:g} s of clean samples - first estimate in ~{MIN_SECONDS - buffer_s:.0f} s if nothing resets it"
            progress = 0.5 * buffer_s / MIN_SECONDS
        elif result is None:
            phase = "Buffer full, estimating every second - none has passed the SNR gate yet. More time will not help until the problems below are fixed."
            progress = 0.5
        elif not result["confident"]:
            need = max(0.0, tracker.conf_span_s - est_span)
            phase = (f"{accepted} estimate(s) accepted over {est_span:.0f} s, spread {result['spread_bpm']} BPM - "
                     + (f"~{need:.0f} s more of agreeing estimates for a confident reading" if need
                        else "waiting for them to agree within 6 BPM at higher SNR"))
            progress = 0.5 + 0.5 * min(1.0, est_span / tracker.conf_span_s) * 0.9
        else:
            phase = f"Confident reading from {accepted} estimates over {est_span:.0f} s"
            progress = 1.0
        headline = "-- BPM" if result is None else f"{result['bpm']:.0f} BPM"
        klass = "" if result is None else ("ok" if result["confident"] else "warn")

        fmt = lambda v, spec, unit="": "--" if v is None else f"{v:{spec}}{unit}"  # noqa: E731
        gates = [
            ["Camera frames/s", fmt(fps, ".1f"), f"needs {MIN_CAMERA_FPS:g}+", fps >= MIN_CAMERA_FPS],
            ["Face found", fmt(face_pct, ".0f", "%"), f"needs {MIN_FACE_FRACTION:.0%}", face_pct >= 100 * MIN_FACE_FRACTION],
            ["Face width", fmt(width, ".0f", " px"), f"needs {MIN_FACE_WIDTH_PX:g}+", width is not None and width >= MIN_FACE_WIDTH_PX],
            ["Frames dropped: motion", fmt(motion_pct, ".0f", "%"), "under 20%", motion_pct <= 20],
            ["Frames dropped: forehead", fmt(covered_pct, ".0f", "%"), "under 30%", covered_pct <= 30],
            ["Brightness drift", fmt(lum_drift, ".1f", "%"), f"under {MAX_LUM_DRIFT_PCT:g}%", lum_drift is not None and lum_drift <= MAX_LUM_DRIFT_PCT],
            ["Clean buffer", fmt(buffer_s, ".1f", " s"), f"needs {MIN_SECONDS:g}+", buffer_s >= MIN_SECONDS],
            ["Latest SNR", fmt(snr, ".1f", " dB"), f"gate {snr_min_db:g}", snr is not None and snr >= snr_min_db],
            ["Latest raw peak", fmt(tracker.last_bpm, ".0f", " BPM"), "shown even if rejected", None],
            ["Accepted estimates", str(accepted), "needs 3+", accepted >= 3],
            ["Accepted span", fmt(est_span, ".1f", " s"), f"needs {tracker.conf_span_s:g}+", est_span >= tracker.conf_span_s],
            ["Buffer restarts (30 s)", str(len(resets)), "want 0", not resets],
        ]
        gates = [[a, b, c, None if ok is None else bool(ok)] for a, b, c, ok in gates]
        wave, psd, peak = [], [], None
        if tracker.last is not None:
            wave = [round(float(v), 5) for v in tracker.last["h"][::2]]
            freqs = tracker.last["freqs"]
            band = (freqs >= HR_LO_HZ) & (freqs <= HR_HI_HZ)
            psd = [float(v) for v in tracker.last["psd"][band][::2]]
            peak = float((tracker.last["bpm"] / 60.0 - HR_LO_HZ) / (HR_HI_HZ - HR_LO_HZ))
        return {"headline": headline, "headline_class": klass, "phase": phase, "progress": round(progress, 3),
                "problems": problems, "gates": gates, "wave": wave, "psd": psd, "peak": peak}


class Handler(BaseHTTPRequestHandler):
    preview: Preview

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        if self.path == "/":
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(PAGE)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(PAGE)
            return
        if self.path == "/healthz":
            payload = json.dumps({
                "ok": self.preview.error is None,
                "frames": self.preview.frames,
                "error": self.preview.error,
            }).encode()
            self.send_response(HTTPStatus.OK if self.preview.error is None else HTTPStatus.SERVICE_UNAVAILABLE)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        if self.path == "/state":
            payload = json.dumps(self.preview.state).encode()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(payload)
            return
        if self.path != "/stream.mjpg":
            self.send_error(HTTPStatus.NOT_FOUND)
            return

        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        sequence = -1
        try:
            while True:
                sequence, frame = self.preview.wait_after(sequence)
                if frame is None:
                    continue
                self.wfile.write(
                    b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                    + str(len(frame)).encode()
                    + b"\r\n\r\n"
                    + frame
                    + b"\r\n"
                )
        except (BrokenPipeError, ConnectionResetError):
            return

    def log_message(self, _format: str, *_args) -> None:
        return


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8005)
    parser.add_argument("--model", type=Path)
    args = parser.parse_args()
    model = args.model or next(
        (path for path in (Path(__file__).with_name("face_detection_yunet_2026may.onnx"), DEFAULT_MODEL) if path.exists()),
        None,
    )
    if model is None:
        parser.error("face model not found")

    preview = Preview(model)
    Handler.preview = preview
    threading.Thread(target=preview.run, name="rppg-preview-camera", daemon=True).start()
    print(f"Heartbeat preview: http://{args.host}:{args.port}/", flush=True)
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
