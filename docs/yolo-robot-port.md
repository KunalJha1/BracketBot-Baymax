# YOLO robot compatibility and port gate

## Verdict

The committed `yolo11n-pose.pt` is a valid, small YOLO11 pose checkpoint and
the model architecture is suitable for the BracketBot Jetson. It is **not yet
verified on the physical robot**, because the robot was intentionally offline
during this pass. Do not interpret local success as proof that the Jetson's
current Python/CUDA/TensorRT environment can load this exact checkpoint.

The safest expected deployment path is:

1. prove the `.pt` model with one synthetic inference in the Jetson's actual
   Python environment;
2. prove one inference from one half of a fresh stereo head-camera frame;
3. benchmark it;
4. if needed, build a TensorRT engine **on that Jetson** and use the engine for
   continuous operation.

A TensorRT engine should not be built on the development Mac or copied from a
different Jetson image. Engines are coupled to the target TensorRT/CUDA stack
and GPU architecture.

## Evidence gathered locally

Checked on 2026-09-19:

| Gate | Result |
| --- | --- |
| Checkpoint SHA-256 | `869e83fcdffdc7371fa4e34cd8e51c838cc729571d1635e5141e3075e9319dc0` |
| Ultralytics load | Pass (`8.4.155`) |
| Model task/classes | Pose; class `0` is `person` |
| Model size | 2,874,462 parameters in the loaded PyTorch model |
| Local PyTorch inference | Pass on a synthetic 640×480 frame |
| ONNX opset 17 export | Pass; output shape `(1, 56, 8400)` |
| ONNX Runtime inference | Pass on a synthetic 640×480 frame |

Repository evidence also shows that `bbapps/greeter/main.py` already loads a
YOLO26 TensorRT engine through `bbai.Detector`, CUDA, and PyCUDA. That is strong
evidence that this robot family has a viable accelerated detection path. It is
not proof that Ultralytics or a compatible PyTorch build is installed in the
currently deployed `~/bbos` environment. The existing `bbai.Detector` also
expects a detection-engine output, while this checkpoint emits pose output, so
the two engines are not drop-in interchangeable.

## Offline/local check

Run the same deterministic smoke test used during development:

```sh
uv run --extra vision python scripts/check_yolo_runtime.py
```

This loads the committed checkpoint and performs inference on a synthetic
frame. It does not open a camera.

## Robot port gate (when SSH is available)

These checks are read-only except for copying files to `/tmp` and creating a
model cache/export. Keep `--no-sync` on all BBOS environment commands.

1. Inspect the installed runtime before installing anything:

   ```sh
   ssh bot '~/.local/bin/uv run --no-sync --project ~/bbos python -c \
     "import torch, ultralytics; print(torch.__version__, torch.cuda.is_available(), ultralytics.__version__)"'
   ```

2. Copy and run the synthetic gate:

   ```sh
   scp yolo11n-pose.pt scripts/check_yolo_runtime.py bot:/tmp/
   ssh bot 'cd /tmp && ~/.local/bin/uv run --no-sync --project ~/bbos \
     python check_yolo_runtime.py --model yolo11n-pose.pt --device 0'
   ```

3. Capture a single head-camera JPEG with `scripts/capture_head.py`, split the
   2560×960 stereo image into a 1280×960 eye, and rerun the checker with
   `--image`. Never infer on the full side-by-side stereo image as if it were
   one normal camera frame.

4. Record warm inference latency and memory use. A single successful inference
   proves compatibility, not real-time suitability.

5. If PyTorch latency is too high, export/build TensorRT on the robot using the
   versions installed there. Preserve the `.pt` file as the source artifact and
   record the JetPack, CUDA, TensorRT, Ultralytics, input size, precision, and
   checkpoint hash beside the engine.

## Integration constraints

- The head topic is `camera.head.jpeg`; it contains two 1280×960 eyes side by
  side. Pick one eye explicitly.
- The dashboard and model must not autonomously trigger contact gestures from
  a detection. Detection should become an uncertainty-labelled observation;
  consent and the deterministic action gate remain separate.
- `people_detector.py` also runs YuNet and an expression model. Only YOLO person
  detection is part of this robot compatibility verdict. The expression stack
  has separate ONNX/OpenCV requirements and must be benchmarked independently.
- Review Ultralytics licensing before redistribution or commercial use.
