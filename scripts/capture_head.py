"""Save real head-camera frames for YOLO testing and labelling. Runs ON THE ROBOT, read-only.

    uv run --no-sync --project ~/bbos python capture_head.py --out ~/head_frames --count 40 --every 1.0

Each frame is the full 2560 x 960 stereo JPEG exactly as the camera daemon publishes it, named by
timestamp. Move the sheet (and the arms, by hand or teleop) between frames. Opens a Reader only.
"""

import argparse
from pathlib import Path
import time

from bbos import Reader


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=Path.home() / "head_frames")
    p.add_argument("--count", type=int, default=20)
    p.add_argument("--every", type=float, default=1.)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    with Reader("camera.head.jpeg", keeptime=False) as r:
        for i in range(a.count):
            t0 = time.monotonic()
            while not r.ready():
                if time.monotonic() - t0 > 3:
                    raise SystemExit("no camera frame: is the camera daemon running?")
                time.sleep(.005)
            path = a.out / f"head_{time.strftime('%Y%m%d_%H%M%S')}_{i:03d}.jpg"
            path.write_bytes(bytes(r.data["jpeg"][:int(r.data["jpeg_len"])]))
            print(path, flush=True)
            time.sleep(a.every)


if __name__ == "__main__":
    main()
