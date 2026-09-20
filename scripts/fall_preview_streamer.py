"""Robot side of scripts/fall_preview.py: length-prefixed head JPEGs on stdout.

Read-only.  Never commands motion.
"""

import struct
import sys
import time

from bbos import Reader

every = float(sys.argv[1]) if len(sys.argv) > 1 else 0.15
out = sys.stdout.buffer
with Reader("camera.head.jpeg", keeptime=False) as reader:
    while True:
        if not reader.ready():
            time.sleep(0.005)
            continue
        size = int(reader.data["jpeg_len"])
        out.write(struct.pack(">I", size) + bytes(reader.data["jpeg"][:size]))
        out.flush()
        time.sleep(every)
