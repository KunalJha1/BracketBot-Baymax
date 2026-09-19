"""Two live windows to watch the sim -> YOLO pipeline.

    uv run --locked --extra yolo python scripts/live_views.py [--data data/yolo] [--run artifacts/yolo/v1]

"Robot head camera (sim)": the scripted fox fold seen from the robot's measured head mount. Left is the
raw pinhole render MuJoCo makes, right is the same view bent through the robot's fisheye lens into the
table crop YOLO trains on (with clutter, textures and camera effects), corners A-D marked
(filled = visible, ring = hidden). A new seed and look every episode; appearance changes every few seconds.

"YOLO training images": 9 random images from the dataset on disk with their labels, refreshed every
3 s, plus the latest training epoch's metrics.

"YOLO on real robot frames": the newest checkpoint on the robot's real head camera (left eye, same table
crop). With --robot HOST (e.g. `bot`) the frames stream live over ssh from the camera daemon (read-only,
as fast as the link allows; scripts/live_robot.py shows only this window, for demos); otherwise it shows the newest photo in --real, picking up new ones as they land.
Crosses are predicted corners with their confidence. Where YOLO finds no sheet (it is trained on sim
only), the classical finder in bbsim/workbench/yolo/fallback.py takes over, labelled "fallback".

With a checkpoint, the head-camera window gets a third panel: the model's corners (crosses) against
the truth (circles) on the live sim frame, with the error in mm on the table. The checkpoint is
reloaded every minute, so you watch training improve.

Keys (either window): q / Esc quits, space pauses the fold, n starts a new episode.
"""

import argparse
import csv
from pathlib import Path
import random
import struct
import subprocess
import sys
import threading
import time

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bbsim.workbench.yolo.dataset import YoloScene  # noqa: E402
from bbsim.workbench.yolo.evaluate import roi_of  # noqa: E402
from bbsim.workbench.yolo.fallback import find_sheet  # noqa: E402
from bbsim.workbench.yolo.tracker import SheetTracker  # noqa: E402
from PIL import Image  # noqa: E402

COLOURS = [(64, 64, 255), (255, 160, 64), (0, 200, 255), (120, 220, 64)]  # BGR for A-D


# On the robot: the bbos venv reads the decoded head frame and cuts out the left eye's table ROI (see
# roi_of); the system python3, which has OpenCV, JPEG-encodes it (~40 kB instead of the 265 kB stereo
# JPEG). The laptop asks for each frame with one byte on stdin, so frames never queue up in ssh: what
# arrives is at most a couple of frames old, however slow the link.
GRAB = """
import sys, time
from bbos import Reader
out, ask = sys.stdout.buffer, sys.stdin.buffer
with Reader("camera.head.rgb", keeptime=False) as r:
    while ask.read(1):
        t0 = time.monotonic()
        while not r.ready():
            if time.monotonic() - t0 > 3:
                sys.exit("no camera frame: is the camera daemon running?")
            time.sleep(.002)
        out.write(r.data["rgb"][480:960, 320:960].tobytes())
        out.flush()
"""
ENCODE = """
import struct, sys, cv2, numpy as np
src, out, size = sys.stdin.buffer, sys.stdout.buffer, 480 * 640 * 3
while True:
    raw = src.read(size)
    if len(raw) < size:
        break
    ok, jpeg = cv2.imencode(".jpg", np.frombuffer(raw, np.uint8).reshape(480, 640, 3)[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, %d])
    out.write(struct.pack("<I", len(jpeg)) + jpeg.tobytes())
    out.flush()
"""


class RobotFrames:
    """Live head-camera table ROI JPEGs from the robot over ssh; `latest` is the newest (count, jpeg) or None.
    Keeps `ahead` requests in flight to hide the round trip. Runs from /tmp on the robot: in ~ the bbos
    project folder shadows the bbos package."""

    def __init__(self, host, quality=90, ahead=2):
        cmd = (f"cd /tmp && ~/.local/bin/uv run --no-sync --project ~/bbos python -u -c '{GRAB}'"
               f" | /usr/bin/python3 -u -c '{ENCODE % quality}'")
        self.proc = subprocess.Popen(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", host, cmd],
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        self.latest, self.count, self.error, self.fps = None, 0, None, 0.
        self.ask(ahead)
        threading.Thread(target=self.read, daemon=True).start()

    def ask(self, n=1):
        try:
            self.proc.stdin.write(b"f" * n)
            self.proc.stdin.flush()
        except OSError:
            pass  # the stream died; read() reports it

    def read(self):
        pipe, last = self.proc.stdout, time.monotonic()
        while True:
            head = pipe.read(4)
            if len(head) < 4:
                self.error = f"robot stream ended (ssh exit {self.proc.wait()})"
                return
            jpeg = pipe.read(struct.unpack("<I", head)[0])
            self.ask()
            now = time.monotonic()
            self.fps = .9 * self.fps + .1 / max(now - last, 1e-3)
            last = now
            self.count += 1
            self.latest = self.count, jpeg

    def close(self):
        self.proc.kill()


def draw_corners(image, px, vis):
    for k, ((x, y), v) in enumerate(zip(px, vis)):
        if not v:
            continue
        cv2.circle(image, (int(x), int(y)), 7, COLOURS[k], -1 if v == 2 else 2, cv2.LINE_AA)
        cv2.putText(image, "ABCD"[k], (int(x) + 9, int(y) - 7), cv2.FONT_HERSHEY_SIMPLEX, .6, COLOURS[k], 2, cv2.LINE_AA)


def gallery(data, rng, cells=(3, 3), size=(426, 320)):
    files = list((data / "images" / "train").glob("*.jpg"))
    tiles = []
    for f in rng.sample(files, min(len(files), cells[0] * cells[1])):
        im = cv2.imread(str(f))
        h, w = im.shape[:2]
        values = list(map(float, (data / "labels" / "train" / f"{f.stem}.txt").read_text().split()[1:]))
        cx, cy, bw, bh = values[:4]
        cv2.rectangle(im, (int((cx - bw / 2) * w), int((cy - bh / 2) * h)), (int((cx + bw / 2) * w), int((cy + bh / 2) * h)), (0, 255, 0), 2)
        kps = np.array(values[4:]).reshape(4, 3)
        draw_corners(im, kps[:, :2] * [w, h], kps[:, 2].astype(int))
        im = cv2.resize(im, size)
        cv2.putText(im, f.stem, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 255), 1, cv2.LINE_AA)
        tiles.append(im)
    while len(tiles) < cells[0] * cells[1]:
        tiles.append(np.zeros((size[1], size[0], 3), np.uint8))
    rows = [np.hstack(tiles[r * cells[1]:(r + 1) * cells[1]]) for r in range(cells[0])]
    return np.vstack(rows), len(files)


class Model:
    """The newest YOLO checkpoint, reloaded when it changes (at most once a minute)."""

    def __init__(self, run):
        self.paths = [run / "runs" / "pose" / "weights" / "last.pt", run / "pose.pt"]
        self.net, self.loaded, self.checked = None, None, 0.

    def get(self):
        if time.monotonic() - self.checked < 60 and self.net is not None:
            return self.net
        self.checked = time.monotonic()
        found = [p for p in self.paths if p.exists()]
        if found:
            newest = max(found, key=lambda p: p.stat().st_mtime)
            stamp = (newest, newest.stat().st_mtime)
            if stamp != self.loaded:
                from ultralytics import YOLO

                self.net, self.loaded = YOLO(str(newest)), stamp
                print(f"loaded {newest}", flush=True)
        return self.net

    def corners(self, bgr):
        net = self.get()
        if net is None:
            return None
        result = net.predict(bgr, verbose=False, conf=.2)[0]
        if result.keypoints is None or len(result.boxes) == 0:
            return "none"
        i = int(result.boxes.conf.argmax())
        conf = result.keypoints.conf[i].cpu().numpy() if result.keypoints.conf is not None else np.ones(4)
        return result.keypoints.xy[i].cpu().numpy(), conf, float(result.boxes.conf[i])

    def candidates(self, bgr, conf=.1):
        """Every sheet YOLO sees, weak ones too, as [(corners, per-corner confidence, score)]; the
        tracker decides which (if any) is the sheet."""
        net = self.get()
        if net is None:
            return None
        result = net.predict(bgr, verbose=False, conf=conf)[0]
        if result.keypoints is None or len(result.boxes) == 0:
            return []
        xy = result.keypoints.xy.cpu().numpy()
        kc = result.keypoints.conf.cpu().numpy() if result.keypoints.conf is not None else np.ones(xy.shape[:2])
        return list(zip(xy, kc, result.boxes.conf.cpu().numpy().astype(float)))


def real_corners(model, bgr, trust=.3):
    """YOLO on a real table ROI, or the classical fallback when YOLO finds no sheet (or a weak one):
    (prediction, source name) in draw_prediction's format."""
    pred = model.corners(bgr)
    if pred is None or pred == "none" or pred[2] < trust:
        found = find_sheet(bgr)
        if found is not None:
            return found, "fallback"
    return pred, "YOLO"


def tracked_corners(model, tracker, bgr, now, trust=.3):
    """The tracked sheet in a live frame: YOLO's sheets near the table's middle (or the track), else the
    classical fallback there, folded into `tracker`. (corners, seen, state, source) or None."""
    found = [(xy, c, s, "YOLO") for xy, c, s in model.candidates(bgr) or [] if s >= .1]
    if not any(s >= trust and tracker.near(xy) for xy, _, s, _ in found):
        sheet = find_sheet(bgr, tracker.near)
        if sheet is not None:
            found.append((*sheet, "fallback"))
    track = tracker.update(found, now)
    return None if track is None else (*track, tracker.source)


def draw_track(image, track):
    """Seen corners as crosses; corners carried along under an arm (or a held track) as rings."""
    if track is None:
        cv2.putText(image, "looking for the sheet", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 0, 255), 2, cv2.LINE_AA)
        return
    xy, seen, state, source = track
    cv2.polylines(image, [xy.astype(np.int32)], True, (255, 255, 255) if state == "tracking" else (160, 160, 160), 1, cv2.LINE_AA)
    for k, ((x, y), v) in enumerate(zip(xy, seen)):
        if v:
            cv2.drawMarker(image, (int(x), int(y)), COLOURS[k], cv2.MARKER_CROSS, 18, 2, cv2.LINE_AA)
        else:
            cv2.circle(image, (int(x), int(y)), 8, COLOURS[k], 2, cv2.LINE_AA)
        cv2.putText(image, "ABCD"[k], (int(x) + 10, int(y) + 16), cv2.FONT_HERSHEY_SIMPLEX, .6, COLOURS[k], 2, cv2.LINE_AA)
    text = f"{source}: {seen.sum()}/4 corners seen" if state == "tracking" else "held: sheet not seen"
    cv2.putText(image, text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 255, 0) if seen.sum() == 4 else (255, 200, 0), 2, cv2.LINE_AA)


def draw_prediction(image, pred, source="YOLO"):
    if pred is None:
        cv2.putText(image, "no checkpoint yet", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 0, 255), 2, cv2.LINE_AA)
        return
    if pred == "none":
        cv2.putText(image, f"{source}: no sheet found", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 0, 255), 2, cv2.LINE_AA)
        return
    xy, conf, score = pred
    for k, ((x, y), c) in enumerate(zip(xy, conf)):
        cv2.drawMarker(image, (int(x), int(y)), COLOURS[k], cv2.MARKER_CROSS, 18, 2, cv2.LINE_AA)
        cv2.putText(image, f"{'ABCD'[k]} {c:.2f}", (int(x) + 10, int(y) + 16), cv2.FONT_HERSHEY_SIMPLEX, .5, COLOURS[k], 1, cv2.LINE_AA)
    cv2.putText(image, f"{source} sheet {score:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 255, 0) if source == "YOLO" else (255, 200, 0), 2, cv2.LINE_AA)


def training_status(run):
    results = run / "runs" / "pose" / "results.csv"
    if not results.exists():
        return "training: waiting for the first epoch"
    rows = list(csv.DictReader(open(results)))
    if not rows:
        return "training: waiting for the first epoch"
    last = {k.strip(): v for k, v in rows[-1].items()}
    return (f"epoch {int(float(last['epoch']))}  pose mAP50-95 {float(last.get('metrics/mAP50-95(P)', 0)):.3f}  "
            f"box mAP50 {float(last.get('metrics/mAP50(B)', 0)):.3f}  kpt loss {float(last.get('train/pose_loss', 0)):.3f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=Path("data/yolo"))
    p.add_argument("--run", type=Path, default=Path("artifacts/yolo/v1"))
    p.add_argument("--seed", type=int, default=7000)
    p.add_argument("--real", type=Path, default=Path("data/real/head"), help="real head-camera photos to run the model on")
    p.add_argument("--robot", default=None, help="ssh host to stream live head-camera frames from (e.g. bot); overrides --real")
    a = p.parse_args()
    pick = random.Random(0)
    seed, paused, skip = a.seed, False, False
    scene = YoloScene(seed)
    rng = np.random.default_rng(seed)
    scene.reset(seed)
    scene.model.geom_pos[scene.table_geom], scene.model.geom_size[scene.table_geom] = scene.nominal_table
    scene.redraw(rng)
    last_look = last_gallery = 0.
    board = None
    cv2.namedWindow("Robot head camera (sim)", cv2.WINDOW_NORMAL)
    cv2.namedWindow("YOLO training images", cv2.WINDOW_NORMAL)
    model = Model(a.run)
    robot = RobotFrames(a.robot) if a.robot else None
    tracker = SheetTracker()
    last_real, shown = 0., None
    while True:
        if not paused:
            for _ in range(20):  # 0.1 s of sim per displayed frame
                scene.tick()
            if scene.done or skip or scene.data.time > 200:
                skip = False
                seed += 1
                rng = np.random.default_rng(seed)
                scene.reset(seed)
                scene.model.geom_pos[scene.table_geom], scene.model.geom_size[scene.table_geom] = scene.nominal_table
                scene.redraw(rng)
        now = time.monotonic()
        if now - last_look > 4:
            scene.redraw(rng)
            last_look = now
        scene.cam.renderer.update_scene(scene.data, camera=scene.cam.camera_id)
        pinhole = scene.cam.renderer.render()[:, :, ::-1]
        labelled = scene.label()
        if labelled:
            rgb, _, _, meta = labelled
            fisheye = np.ascontiguousarray(rgb[:, :, ::-1])
            draw_corners(fisheye, meta["corners_px"], meta["visible"])
        else:
            fisheye = np.ascontiguousarray(scene.cam.render(scene.data)[:, :, ::-1])
        h = fisheye.shape[0]
        left = cv2.resize(pinhole, (h, h))
        panels = [left, fisheye]
        if model.get() is not None:
            yolo = np.ascontiguousarray(rgb[:, :, ::-1]).copy() if labelled else fisheye.copy()
            pred = model.corners(yolo)
            draw_prediction(yolo, pred)
            if labelled and pred not in (None, "none"):
                for k, (tx, ty) in enumerate(meta["corners_px"]):
                    if meta["visible"][k]:
                        cv2.circle(yolo, (int(tx), int(ty)), 7, COLOURS[k], 2, cv2.LINE_AA)
                errors = []
                for k in range(4):
                    if meta["visible"][k]:
                        truth = np.array(meta["corners_world"][k])
                        guess = scene.cam.to_plane(scene.data, pred[0][k], truth[2])
                        errors.append(f"{'ABCD'[k]} {np.linalg.norm(guess[:2] - truth[:2]) * 1000:.1f}")
                cv2.putText(yolo, "error mm: " + "  ".join(errors), (10, 90), cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(yolo, "YOLO prediction (x) vs truth (o)", (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 255, 255), 2, cv2.LINE_AA)
            panels.append(yolo)
        view = np.hstack(panels)
        mount = scene.cam.mount
        text = f"seed {seed}  t {scene.data.time:5.1f}s  fold {scene.done_folds}  {scene.phase}  pitch {mount['pitch_deg']:+.1f}  lean {mount['lean_deg']:.1f}"
        cv2.putText(view, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(view, "pinhole render (MuJoCo)", (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(view, "robot fisheye, table crop (YOLO input)", (h + 10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow("Robot head camera (sim)", view)
        if now - last_gallery > 3 and (a.data / "images" / "train").exists():
            board, total = gallery(a.data, pick)
            cv2.putText(board, f"{total} training images   {training_status(a.run)}", (10, board.shape[0] - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow("YOLO training images", board)
            last_gallery = now
        if (robot or now - last_real > 2) and model.get() is not None:
            last_real = now
            if robot:
                frame = robot.latest
                name = f"live from {a.robot}, frame {frame[0]}, {robot.fps:.1f} fps" if frame else None
            else:  # the newest photo; the folder is rescanned so new captures show up
                photos = list(a.real.glob("*.jpg")) if a.real.exists() else []
                frame = max(photos, key=lambda f: f.stat().st_mtime) if photos else None
                name = frame.name if frame else None
            if frame is not None and (frame, model.loaded) != shown:  # redo a still photo when the checkpoint changes
                if robot:  # already the table ROI
                    roi = cv2.imdecode(np.frombuffer(frame[1], np.uint8), cv2.IMREAD_COLOR)
                else:
                    try:
                        roi = np.ascontiguousarray(np.asarray(roi_of(Image.open(frame)))[:, :, ::-1])
                    except Exception:  # a photo still copying, or cut off
                        roi = None
                if roi is not None:
                    shown = frame, model.loaded
                    if robot:
                        draw_track(roi, tracked_corners(model, tracker, roi, now))
                    else:
                        draw_prediction(roi, *real_corners(model, roi))
                    cv2.putText(roi, name, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 255, 255), 2, cv2.LINE_AA)
                    cv2.imshow("YOLO on real robot frames", roi)
            elif robot and robot.error and shown != robot.error:
                print(robot.error, flush=True)
                shown = robot.error
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord(" "):
            paused = not paused
        if key == ord("n"):
            skip = True
    if robot:
        robot.close()
    scene.cam.close()
    scene.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
