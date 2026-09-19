"""Read-only camera probe. Run ON THE ROBOT, inside its BBOS environment:

    python scripts/probe_cameras.py            # writes camera_probe/ next to where it runs

It opens Readers and Configs only, never a Writer, so it cannot move anything or take a topic
from its owner. For each camera topic it reports resolution and frame rate over a few seconds and
saves one still; it also dumps every camera config it can load (intrinsics, T_base_cam).

Bring camera_probe/ back to this repo. The stills settle whether camera.left/right are a head
stereo pair or wrist cameras (preset head_stereo vs head_wrist); the configs replace the
placeholder poses in bbsim/workbench/pi0/cameras.py via --cameras cameras.json.

Topic and config names come from bracketbot-bbos-dictionary.md and may differ on your BBOS
version: anything missing is reported and skipped, not treated as an error.
"""

import io
import json
from pathlib import Path
import sys
import time

TOPICS = ["camera.head.jpeg", "camera.head.rgb", "camera.left.jpeg", "camera.right.jpeg", "camera.rect", "camera.depth"]
CONFIGS = ["cam_head", "cam_left", "cam_right", "depth", "arm_left", "arm_right", "base"]
SECONDS = 5.


def plain(value):
    """Best-effort JSON form of a BBOS config value."""
    try:
        import numpy as np

        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
    except ImportError:
        pass
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    return repr(value)


def probe_topic(Reader, topic, out):
    from PIL import Image
    import numpy as np

    row = dict(topic=topic)
    try:
        with Reader(topic, keeptime=False) as reader:
            frames, first, still, started = 0, None, None, time.monotonic()
            while time.monotonic() - started < SECONDS:
                if not reader.ready():
                    time.sleep(.002)
                    continue
                frames += 1
                first = first or time.monotonic()
                if still is None:
                    data = reader.data
                    # BBOS samples are numpy structured arrays: field names live on the dtype.
                    row["fields"] = sorted(data.dtype.names or ()) or repr(type(data))
                    if "jpeg" in topic:
                        n = int(data["jpeg_len"]) if "jpeg_len" in row["fields"] else int(data["bytesused"])
                        still = Image.open(io.BytesIO(bytes(data["jpeg"][:n]))).convert("RGB")
                    elif "rgb" in row["fields"]:
                        still = Image.fromarray(np.asarray(data["rgb"]).astype(np.uint8))
                    elif "depth" in row["fields"]:
                        depth = np.asarray(data["depth"], dtype=np.float32)
                        row["depth_range"] = [float(np.nanmin(depth)), float(np.nanmax(depth))]
                        still = Image.fromarray((255 * np.clip(depth / max(np.nanmax(depth), 1e-6), 0, 1)).astype(np.uint8))
            elapsed = time.monotonic() - first if first else 0.
            row["frames"] = frames
            row["hz"] = round((frames - 1) / elapsed, 1) if frames > 1 and elapsed > 0 else None
            if still is not None:
                row["width"], row["height"] = still.size
                still.save(out / f"{topic.replace('.', '_')}.png")
    except Exception as exc:  # a missing topic on this BBOS version: report it, keep going
        row["error"] = f"{type(exc).__name__}: {exc}"
    return row


def probe_config(Config, name):
    try:
        config = Config(name)
    except Exception as exc:
        return dict(config=name, error=f"{type(exc).__name__}: {exc}")
    fields = {}
    for key in (k for k in dir(config) if not k.startswith("_")):
        try:
            value = getattr(config, key)
        except Exception as exc:
            value = f"<unreadable: {exc}>"
        if not callable(value):
            fields[key] = plain(value)
    return dict(config=name, fields=fields)


def main():
    try:
        import bbos
        from bbos import Config, Reader
    except ImportError:
        sys.exit("bbos is not importable here: run this on the robot, in its BBOS environment")
    out = Path("camera_probe")
    out.mkdir(exist_ok=True)
    report = dict(bbos=getattr(bbos, "__file__", None), version=getattr(bbos, "__version__", None), time=time.strftime("%Y-%m-%d %H:%M:%S"))
    report["topics"] = [probe_topic(Reader, topic, out) for topic in TOPICS]
    report["configs"] = [probe_config(Config, name) for name in CONFIGS]
    (out / "probe.json").write_text(json.dumps(report, indent=2) + "\n")
    for row in report["topics"]:
        status = row.get("error") or f"{row.get('width')}x{row.get('height')} at {row.get('hz')} Hz"
        print(f"{row['topic']:22s} {status}")
    for row in report["configs"]:
        print(f"Config({row['config']!r}): {row.get('error') or ', '.join(row['fields'])}")
    print(f"wrote {out.resolve()}")


if __name__ == "__main__":
    main()
