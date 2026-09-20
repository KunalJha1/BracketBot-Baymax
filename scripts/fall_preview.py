"""Live fall-detector preview on the laptop, fed by the robot head camera.

    python3 scripts/fall_preview.py            # then open http://127.0.0.1:8031/

The robot only streams JPEGs (scripts/fall_preview_streamer.py over scripts/bot);
pose inference and the shipped ground_safety decision code run here.  Read-only:
nothing in this path can move the robot.
"""

from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import struct
import subprocess
import threading
import time
import webbrowser

import cv2
import numpy as np

import fall_check_frame as fcf

REPO_ROOT = Path(__file__).resolve().parents[1]
BOT = str(REPO_ROOT / "scripts" / "bot")
STATUS_COLOR = {"alert": (0, 0, 255), "checking": (0, 165, 255)}

latest = {"jpeg": None, "stamp": 0.0}
lock = threading.Condition()


def read_exact(stream, count):
    data = b""
    while len(data) < count:
        chunk = stream.read(count - len(data))
        if not chunk:
            raise EOFError("robot stream ended")
        data += chunk
    return data


def banner(frame, text, color):
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 44), (0, 0, 0), -1)
    cv2.putText(frame, text, (12, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2, cv2.LINE_AA)


def draw_person(canvas, index, keypoints, assessment, status):
    colour = STATUS_COLOR.get(status, (0, 0, 255) if assessment.suspected else (0, 200, 0))
    lookup = {kp.index: kp for kp in keypoints if kp.confidence >= 0.35}
    for first, second in fcf.SKELETON_EDGES:
        if first in lookup and second in lookup:
            cv2.line(
                canvas,
                (int(lookup[first].x), int(lookup[first].y)),
                (int(lookup[second].x), int(lookup[second].y)),
                colour,
                2,
            )
    for keypoint in lookup.values():
        cv2.circle(canvas, (int(keypoint.x), int(keypoint.y)), 4, colour, -1)
    if lookup:
        x = int(min(kp.x for kp in lookup.values()))
        y = max(64, int(min(kp.y for kp in lookup.values())) - 10)
        prefix = f"{status.upper()} " if status in STATUS_COLOR else ""
        label = f"#{index} {prefix}{assessment.state} {assessment.confidence:.2f}"
        (width, height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(canvas, (x - 3, y - height - 5), (x + width + 3, y + 5), (0, 0, 0), -1)
        cv2.putText(canvas, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2, cv2.LINE_AA)


def pump(args, model):
    subprocess.run([BOT, "push", str(REPO_ROOT / "scripts" / "fall_preview_streamer.py")], check=True)
    stream = subprocess.Popen(
        [BOT, "py", "-u", "/tmp/baymax/fall_preview_streamer.py", str(args.every)],
        stdout=subprocess.PIPE,
    )
    tracker = fcf.GroundAlertTracker(hold_seconds=args.hold, clear_seconds=args.hold)
    last_line = None
    try:
        while True:
            (size,) = struct.unpack(">I", read_exact(stream.stdout, 4))
            frame = cv2.imdecode(np.frombuffer(read_exact(stream.stdout, size), np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                continue
            eye = fcf.split_eye(frame, args.eye).copy()
            started = time.perf_counter()
            people = fcf.people_in_frame(model, eye, args.device, args.imgsz, args.confidence)
            elapsed = (time.perf_counter() - started) * 1000
            assessments = {
                index: fcf.monocular_assess(keypoints, args.eye, args.keypoint_confidence)
                for index, keypoints in enumerate(people)
            }
            statuses = tracker.update(assessments, time.monotonic())
            view = eye
            for index, keypoints in enumerate(people):
                draw_person(view, index, keypoints, assessments[index], statuses.get(index, "clear"))
            worst = "clear"
            for status in statuses.values():
                if status == "alert" or (status == "checking" and worst != "alert"):
                    worst = status
            if not people:
                text = f"no person   {elapsed:.0f} ms"
            else:
                top = max(assessments, key=lambda index: assessments[index].confidence)
                first = assessments[top]
                text = f"{worst.upper()}  #{top} {first.state}  score={first.confidence:.2f}  {elapsed:.0f} ms"
            banner(view, text, STATUS_COLOR.get(worst, (0, 220, 0)))
            line = text.rsplit("  ", 1)[0]
            if line != last_line:
                print(time.strftime("%H:%M:%S"), text, "|", first.reason if people else "", flush=True)
                last_line = line
            ok, encoded = cv2.imencode(".jpg", view, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ok:
                with lock:
                    latest["jpeg"] = encoded.tobytes()
                    latest["stamp"] = time.time()
                    lock.notify_all()
    finally:
        stream.kill()


PAGE = b"""<!doctype html><title>Fall preview</title>
<body style="margin:0;background:#111;color:#ccc;font:14px system-ui;text-align:center">
<p>Baymax fall-detector preview &mdash; left eye, monocular path, read-only</p>
<img src="/stream" style="max-width:100%;max-height:90vh"></body>"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def do_GET(self):
        if self.path == "/frame.jpg":
            with lock:
                jpeg = latest["jpeg"]
            self.send_response(200 if jpeg else 503)
            self.send_header("Content-Type", "image/jpeg")
            self.end_headers()
            self.wfile.write(jpeg or b"")
            return
        if self.path != "/stream":
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(PAGE)
            return
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.end_headers()
        seen = 0.0
        try:
            while True:
                with lock:
                    lock.wait_for(lambda: latest["stamp"] > seen, timeout=5)
                    jpeg, seen = latest["jpeg"], latest["stamp"]
                if jpeg:
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n")
        except (BrokenPipeError, ConnectionResetError):
            pass


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, default=8031)
    parser.add_argument("--eye", choices=("left", "right"), default="left")
    parser.add_argument("--device", default="mps", help="cpu, mps, 0")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--confidence", type=float, default=0.4)
    parser.add_argument("--keypoint-confidence", type=float, default=0.35)
    parser.add_argument("--every", type=float, default=0.15, help="robot seconds between frames")
    parser.add_argument("--hold", type=float, default=2.0)
    parser.add_argument("--no-open", action="store_true")
    args = parser.parse_args()

    model = fcf.load_model(fcf.resolve_model(None), args.device)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{args.port}/"
    print("preview at", url, flush=True)
    if not args.no_open:
        webbrowser.open(url)
    pump(args, model)


if __name__ == "__main__":
    main()
