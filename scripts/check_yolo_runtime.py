"""Smoke-test the committed YOLO checkpoint on the current machine.

The check uses a synthetic frame by default, so it verifies the framework,
checkpoint, and selected accelerator without needing a camera or personal data.
Copy this script and the checkpoint to the robot for the deployment gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import time


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description="Smoke-test a YOLO model runtime")
    parser.add_argument("--model", type=Path, default=Path("yolo11n-pose.pt"))
    parser.add_argument("--device", default="cpu", help="cpu, 0, mps, or another Ultralytics device")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--image", type=Path, help="optional real image; synthetic black frame by default")
    args = parser.parse_args()

    if not args.model.is_file():
        raise SystemExit(f"model not found: {args.model}")

    import cv2
    import numpy as np
    import torch
    import ultralytics
    from ultralytics import YOLO

    if args.image:
        frame = cv2.imread(str(args.image))
        if frame is None:
            raise SystemExit(f"could not read image: {args.image}")
    else:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

    started = time.perf_counter()
    model = YOLO(str(args.model), task="pose")
    result = model.predict(
        frame,
        classes=[0],
        imgsz=args.imgsz,
        device=args.device,
        verbose=False,
    )[0]
    elapsed = time.perf_counter() - started
    report = {
        "ok": True,
        "machine": platform.machine(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "ultralytics": ultralytics.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "requested_device": args.device,
        "model": str(args.model),
        "model_sha256": sha256(args.model),
        "task": model.task,
        "classes": model.names,
        "parameters": sum(parameter.numel() for parameter in model.model.parameters())
        if hasattr(model.model, "parameters")
        else None,
        "inference_seconds": round(elapsed, 4),
        "people": len(result.boxes),
        "keypoints_shape": list(result.keypoints.xy.shape),
        "input": str(args.image) if args.image else "synthetic 640x480 black frame",
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
