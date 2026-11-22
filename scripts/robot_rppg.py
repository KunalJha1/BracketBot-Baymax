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
"""Contactless heart-rate scan from the BracketBot head camera. Runs ON THE ROBOT, read-only.

    scp scripts/robot_rppg.py rppg.py rppg_view.py \
        assets/models/face_detection_yunet_2026may.onnx bot:/tmp/
    ssh bot 'cd /tmp && ~/.local/bin/uv run robot_rppg.py --check'   # camera + face gate, no scan
    ssh bot 'cd /tmp && ~/.local/bin/uv run robot_rppg.py'           # 15 s scan, JSON on stdout
    ssh bot 'cd /tmp && ~/.local/bin/uv run robot_rppg.py --view --monitor'   # live debug page

--view serves a debug page on the robot (default port 8021) showing the left-eye frame with the
model's own working drawn on it - the detector box, the steadied box, the three skin regions, and
which pixels inside them survived the skin gate - beside the traces and spectra the rate is read
off. --monitor repeats the scan until interrupted, which is the useful mode for tuning. The page
is unauthenticated and shows a live camera view of whoever is in front of the robot: serve it on
a trusted network, only while someone is watching, and stop it afterwards.

The head camera is a shared topic, not this tool's camera, so the face-metered exposure loop in
rppg.py stays off here: nothing may retune a camera every other service is reading. Exposure for
the scan is whatever the camera service publishes, which is why the debug view carries the skin
brightness readout - a face down in the bottom of the range is a scan worth repeating in better
light rather than one worth trusting.

Opens one Reader on the head camera and never a Writer, so it cannot move the robot or take a topic
from its owner. It stores no frames: only per-frame mean skin RGB is kept, in memory. Run it from
/tmp (in ~ the bbos project folder shadows the bbos package). The robot should be stationary and
the person should have agreed to the scan; ask them to face the camera and hold still.

Not a medical device. The result is a demo-grade estimate: it must not be used to diagnose, treat,
triage, or trigger any robot action.
"""

import argparse
import json
from pathlib import Path
import sys
import threading
import time

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))  # rppg.py at the repo root when run from a checkout

from rppg import DEFAULT_MODEL, FaceROI, measure_heart_rate  # noqa: E402

VIEW_PORT = 8021
VIEW_FPS = 6.0            # the scan loop pays for every frame drawn; it must stay cheap
VIEW_FRAME_WIDTH = 480    # the eye is 1280 wide, and nothing in the debug picture needs that

RAW_TOPIC = "camera.head.rgb"    # raw RGB: no JPEG blocking on a ~0.1-1 % pulse signal
JPEG_TOPIC = "camera.head.jpeg"


class HeadCamera:
    """The head camera's left eye as BGR frames. The topic is 2560x960: two 1280x960 eyes side by side,
    so the frame is split at half its width and never processed as one normal camera image."""

    def __init__(self, jpeg=False):
        from bbos import Reader

        self.jpeg = jpeg
        self.reader = Reader(JPEG_TOPIC if jpeg else RAW_TOPIC, keeptime=False)
        self.frames = 0
        self.first = self.last = None

    def __enter__(self):
        self.reader.__enter__()
        return self

    def __exit__(self, *exc):
        return self.reader.__exit__(*exc)

    def grab(self):
        """Newest left-eye BGR frame, or None when the camera has not published a new one yet."""
        if not self.reader.ready():
            time.sleep(0.002)
            return None
        data = self.reader.data
        if self.jpeg:
            stereo = cv2.imdecode(np.frombuffer(bytes(data["jpeg"][:int(data["jpeg_len"])]), np.uint8),
                                  cv2.IMREAD_COLOR)
            left = stereo[:, :stereo.shape[1] // 2]
            frame = np.ascontiguousarray(left)
        else:
            stereo = data["rgb"]
            frame = cv2.cvtColor(stereo[:, :stereo.shape[1] // 2], cv2.COLOR_RGB2BGR)  # copies out of shm
        self.last = time.monotonic()
        self.first = self.first or self.last
        self.frames += 1
        return frame

    @property
    def fps(self):
        return self.frames / (self.last - self.first) if self.frames > 1 and self.last > self.first else 0.0


# ----------------------------------------------------------------------------
# Live debug view (optional; serves what the model sees over HTTP)
# ----------------------------------------------------------------------------

PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>rPPG debug view</title>
<style>
  :root { color-scheme: dark; }
  body { margin: 0; background: #131316; color: #e8e8ea; font: 14px/1.5 ui-sans-serif, system-ui,
         -apple-system, Segoe UI, Roboto, sans-serif; }
  header { padding: 12px 16px; border-bottom: 1px solid #2a2a30; }
  h1 { margin: 0; font-size: 15px; font-weight: 600; }
  p { margin: 4px 0 0; color: #9a9aa4; font-size: 12px; max-width: 70ch; }
  main { padding: 16px; }
  img { width: 100%; max-width: 1180px; height: auto; display: block; border-radius: 8px;
        background: #000; }
  #state { margin-top: 12px; color: #9a9aa4; font-size: 12px; font-variant-numeric: tabular-nums; }
  b { color: #e8e8ea; font-weight: 600; }
</style></head>
<body>
  <header>
    <h1>rPPG debug view &mdash; BracketBot head camera</h1>
    <p>Green is the skin the mean was taken over; red is what the skin gate threw away.
       A camera-based demo estimate, not a medical measurement.</p>
  </header>
  <main>
    <img src="/stream.mjpg" alt="Live rPPG debug view">
    <div id="state">connecting&hellip;</div>
  </main>
  <script>
    async function poll() {
      let text = "view stopped";
      try {
        const s = await (await fetch("/state.json", { cache: "no-store" })).json();
        const bpm = s.bpm === null ? "--" : s.bpm.toFixed(0);
        const snr = s.snr_db === null ? "--" : s.snr_db.toFixed(1) + " dB";
        text = `<b>${bpm} BPM</b> &middot; SNR ${snr} &middot; ${s.windows} windows stacked`
             + ` &middot; ${s.status} &middot; ${s.fps.toFixed(1)} fps`
             + (s.confident ? " &middot; <b>confident</b>" : " &middot; not confident");
      } catch (e) { /* the scan ended or the robot went away */ }
      document.getElementById("state").innerHTML = text;
      setTimeout(poll, 1000);
    }
    poll();
  </script>
</body></html>
"""


class DebugView:
    """Serve the debug picture as MJPEG, plus the numbers as JSON.

    The scan owns the camera and the timing, so this only ever holds the newest
    already-drawn frame: a browser that cannot keep up falls behind on frames
    rather than slowing the measurement down. Frames are throttled and shrunk
    before they are drawn, because a 1280x960 eye composed and encoded thirty
    times a second would cost the robot more CPU than the measurement does.
    """

    def __init__(self, port=VIEW_PORT, host="0.0.0.0", fps=VIEW_FPS, quality=70):
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

        self.period = 1.0 / fps if fps > 0 else 0.0
        self.quality = int(quality)
        self.lock = threading.Lock()
        self.jpeg = None
        self.state = {"bpm": None, "snr_db": None, "confident": False, "windows": 0,
                      "status": "starting", "fps": 0.0}
        self.next_at = 0.0
        view = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *_args):
                pass                                      # stdout is a JSON stream for the caller

            def _send(self, code, body, content_type):
                self.send_response(code)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                path = self.path.split("?")[0]
                if path in ("/", "/index.html"):
                    self._send(200, PAGE.encode("utf-8"), "text/html; charset=utf-8")
                elif path == "/state.json":
                    with view.lock:
                        body = json.dumps(view.state).encode("utf-8")
                    self._send(200, body, "application/json")
                elif path == "/frame.jpg":
                    with view.lock:
                        jpeg = view.jpeg
                    if jpeg is None:
                        self._send(503, b"no frame yet", "text/plain")
                    else:
                        self._send(200, jpeg, "image/jpeg")
                elif path == "/stream.mjpg":
                    self.stream()
                else:
                    self._send(404, b"not found", "text/plain")

            def stream(self):
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                last = None
                try:
                    while True:
                        with view.lock:
                            jpeg = view.jpeg
                        if jpeg is not None and jpeg is not last:
                            last = jpeg
                            self.wfile.write(
                                b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                + str(len(jpeg)).encode() + b"\r\n\r\n" + jpeg + b"\r\n")
                        time.sleep(0.03)
                except (BrokenPipeError, ConnectionResetError, OSError):
                    pass                                  # the browser went away

        self.server = ThreadingHTTPServer((host, port), Handler)
        self.server.daemon_threads = True
        self.thread = threading.Thread(target=self.server.serve_forever,
                                       name="rppg-view", daemon=True)

    @property
    def url(self):
        host, port = self.server.server_address[:2]
        return f"http://{'127.0.0.1' if host in ('0.0.0.0', '::') else host}:{port}/"

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.server.shutdown()
        self.server.server_close()
        return False

    def publish(self, frame, detail, tracker, status, fps=0.0):
        """Draw and store one frame, at most every `period` seconds."""
        now = time.monotonic()
        if self.period and now < self.next_at:
            return
        self.next_at = now + self.period
        import rppg_view

        # The face width has to be read before the frame is shrunk: it is quoted in
        # camera pixels everywhere else, and the working-range table is in those too.
        face_w = float(detail["steady_box"][2]) if detail and detail.get("steady_box") else 0.0
        small = frame
        if frame is not None and frame.shape[1] > VIEW_FRAME_WIDTH:
            scale = VIEW_FRAME_WIDTH / frame.shape[1]
            small = cv2.resize(frame, (VIEW_FRAME_WIDTH, max(1, int(frame.shape[0] * scale))),
                               interpolation=cv2.INTER_AREA)
            detail = scale_detail(detail, scale)
        picture = rppg_view.compose(
            rppg_view.draw_overlay(small, detail),
            rppg_view.state_from(tracker, frame_fps=fps, status=status, detail=detail,
                                 face_w=face_w, note="not a medical measurement"))
        ok, buf = cv2.imencode(".jpg", picture, [int(cv2.IMWRITE_JPEG_QUALITY), self.quality])
        if not ok:
            return
        agg = tracker.result()
        with self.lock:
            self.jpeg = buf.tobytes()
            self.state = {
                "bpm": None if agg is None else round(float(agg["bpm"]), 1),
                "snr_db": None if agg is None else float(agg["snr_db"]),
                "confident": bool(agg and agg["confident"]),
                "windows": tracker.stack.n_windows,
                "status": status,
                "fps": round(float(fps), 1),
            }


def scale_detail(detail, scale):
    """Move one FaceROI.last onto the shrunk frame the view draws."""
    if not detail:
        return detail
    out = dict(detail)
    for key in ("box", "steady_box"):
        if out.get(key):
            out[key] = [v * scale for v in out[key]]
    if out.get("landmarks"):
        out["landmarks"] = [[x * scale, y * scale] for x, y in out["landmarks"]]
    for key in ("mask", "region_mask", "labels"):
        m = out.get(key)
        if m is not None:
            out[key] = cv2.resize(m, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    return out


def check(cam, model, seconds):
    """Camera + face gate: is the eye split right, is the frame rate usable, is a face big enough?"""
    roi = FaceROI(model)
    seen = faces = 0
    widths, lums = [], []
    shape = None
    t0 = time.monotonic()
    while time.monotonic() - t0 < seconds:
        frame = cam.grab()
        if frame is None:
            continue
        shape = frame.shape
        seen += 1
        r = roi(frame, time.monotonic() * 1000)
        if r is not None:
            faces += 1
            widths.append(r[2])
            lums.append(float(np.dot(r[0], [0.299, 0.587, 0.114])))
    lum_drift = 100 * float(np.ptp(lums)) / (float(np.mean(lums)) + 1e-9) if lums else None
    return {"frames": seen, "fps": round(cam.fps, 1),
            "eye_shape": None if shape is None else list(shape),
            "face_frames": faces,
            "face_width_px": round(float(np.median(widths)), 0) if widths else None,
            "lum_drift_pct": None if lum_drift is None else round(lum_drift, 2),
            "ok": bool(seen and cam.fps >= 15 and faces >= 0.8 * seen)}


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--check", action="store_true", help="grab frames and report camera/face health; no scan")
    p.add_argument(
        "--self-test", action="store_true",
        help="load OpenCV and the face model without opening the robot camera",
    )
    p.add_argument("--jpeg", action="store_true", help=f"read {JPEG_TOPIC} instead of raw {RAW_TOPIC} (lossier)")
    p.add_argument("--duration", type=float, default=15.0, help="scan length in seconds")
    p.add_argument("--window", type=float, default=10.0, help="analysis window in seconds")
    p.add_argument("--fs", type=float, default=30.0, help="resample rate in Hz")
    p.add_argument(
        "--model", type=Path,
        help="YuNet face-detection ONNX model (default: next to this script)",
    )
    p.add_argument("--view", action="store_true",
                   help=f"serve the live debug view on port {VIEW_PORT} for as long as the scan "
                        "runs (unauthenticated: trusted networks only)")
    p.add_argument("--view-port", type=int, default=VIEW_PORT, help="port for --view")
    p.add_argument("--view-fps", type=float, default=VIEW_FPS,
                   help="how often --view redraws; higher costs the scan CPU")
    p.add_argument("--monitor", action="store_true",
                   help="repeat the scan until interrupted, for tuning and demos; prints one "
                        "JSON result per scan and never exits on its own")
    p.add_argument("--progress-json", action="store_true",
                   help="stream one compact JSON object per line on stdout (per-second estimates "
                        "tagged \"progress\", then the final \"result\" line) instead of a human report")
    a = p.parse_args()

    model = a.model or next(
        (m for m in (HERE / "face_detection_yunet_2026may.onnx", DEFAULT_MODEL) if m.exists()), None
    )
    if model is None or not model.exists():
        sys.exit(
            "YuNet face model not found: copy assets/models/"
            "face_detection_yunet_2026may.onnx next to this script"
        )
    if a.self_test:
        FaceROI(model)  # Constructor parses the ONNX graph; failure must stop deployment.
        print(json.dumps({
            "ok": True,
            "opencv": cv2.__version__,
            "model": model.name,
            "model_bytes": model.stat().st_size,
        }))
        return

    with HeadCamera(a.jpeg) as cam:
        t0 = time.monotonic()
        while (frame := cam.grab()) is None:
            if time.monotonic() - t0 > 3:
                sys.exit("no camera frame: is the camera daemon running?")
        if a.check:
            print(json.dumps(check(cam, model, seconds=4.0), indent=2))
            return

        view = DebugView(port=a.view_port, fps=a.view_fps) if a.view else None

        def progress(bpm, snr, done):
            if a.progress_json:
                # One line per estimate so a caller can speak intermediate readings
                # while the scan is still running.
                print(json.dumps({"progress": round(float(done), 3),
                                  "bpm": None if bpm is None else round(float(bpm), 1),
                                  "snr_db": None if snr is None else round(float(snr), 1)}),
                      flush=True)
                return
            print(f"{done * 100:5.1f}%  bpm={'--' if bpm is None else round(bpm, 1)}  "
                  f"snr={'--' if snr is None else round(snr, 1)} dB", file=sys.stderr, flush=True)

        def scan_once():
            on_frame = None
            if view is not None:
                def on_frame(frame, detail, tracker, status):
                    view.publish(frame, detail, tracker, status, fps=cam.fps)
            result = measure_heart_rate(duration_s=a.duration, window_s=a.window, fs=a.fs,
                                        model_path=model, on_update=progress, grab=cam.grab,
                                        on_frame=on_frame)
            report = {"result": result, "fps": round(cam.fps, 1), "frames": cam.frames,
                      "note": "camera-based demo estimate, not a medical measurement"}
            # The streaming form stays one object per line; the human form stays indented.
            print(json.dumps(report) if a.progress_json else json.dumps(report, indent=2),
                  flush=True)
            return result

        def run():
            """One scan and an exit code, or scan after scan until interrupted."""
            if not a.monitor:
                sys.exit(0 if scan_once() else 1)
            try:
                while True:
                    scan_once()
            except KeyboardInterrupt:
                print("stopped", file=sys.stderr, flush=True)

        if view is None:
            run()
            return

        with view:
            # The URL is for a person to open, so it goes to stderr and leaves stdout
            # a clean JSON stream for whatever started the scan.
            print(f"debug view on {view.url} (Ctrl-C to stop)", file=sys.stderr, flush=True)
            run()


if __name__ == "__main__":
    main()
