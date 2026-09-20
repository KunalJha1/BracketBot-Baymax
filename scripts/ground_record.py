"""Record what the ground-safety detector sees, for offline tuning. Read-only.

Runs on the robot; opens no writers, so it cannot move anything::

    scripts/bot push scripts/ground_record.py --to /tmp
    scripts/bot 'cd ~/bbos && .venv/bin/python /tmp/ground_record.py 60'
    scripts/bot 'cd /tmp && tar cf ground_rec.tar ground_rec'
    scripts/bot pull /tmp/ground_rec.tar .

Then replay it with ``scripts/ground_replay.py``. Each ``fNNNN.npz`` holds the
camera.rect frame, the raw left eye, and the vision app's published verdict.
"""

import json
import os
import shutil
import signal
import sys
import time

import numpy as np
from bbos import Reader

duration = float(sys.argv[1]) if len(sys.argv) > 1 else 60.0
out = sys.argv[2] if len(sys.argv) > 2 else "/tmp/ground_rec"
shutil.rmtree(out, ignore_errors=True)
os.makedirs(out)
signal.alarm(int(duration) + 30)  # never outlive the request, even if a topic stalls
count = 0
with Reader("camera.rect", keeptime=False) as rect, Reader("camera.head.rgb", keeptime=False) as head:
    end, due, raw = time.time() + duration, 0.0, None
    while time.time() < end:
        if head.ready():
            rgb = np.asarray(head.data["rgb"])
            raw = rgb[:, : rgb.shape[1] // 2].copy()
        if rect.ready() and raw is not None and time.time() >= due:
            try:
                verdict = json.load(open("/tmp/bracketbot_ground_alert.json"))
            except (OSError, ValueError):
                verdict = {}
            np.savez_compressed(
                f"{out}/f{count:04d}.npz", rect=np.asarray(rect.data["left"]).copy(),
                raw_left=raw, verdict=json.dumps(verdict), t=time.time(),
            )
            count += 1
            due = time.time() + 0.4
        time.sleep(0.02)
print("saved", count, "frames to", out)
