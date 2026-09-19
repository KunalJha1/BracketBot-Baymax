# Robot emotion greeter

This app runs the complete vision-to-voice path on the robot:

1. read the depth-aligned left eye from `camera.rect`;
2. detect people and body keypoints with the YOLO11 pose model;
3. associate pose joints with same-timestamp `camera.points` depth;
4. flag a sustained **possible person on ground** observation and locate it in
   the `slam.pose` map;
5. locate the primary face with YuNet;
6. estimate its visible expression by averaging two EmotiEffLib ONNX models;
   and
7. after a sustained, confidently sad-looking expression, say "Hey, why are
   you sad? What's up?", listen for the answer (whisper.cpp), and reply with
   the OpenRouter LLM for up to three turns (`check_in.py`).

If whisper.cpp, espeak-ng, or the greeter voice modules are missing, the app
falls back to playing `sad_prompt.wav`. If the LLM cannot be reached it gives
a short fixed, supportive reply. Use `--no-check-in` to keep the recorded
prompt only.

The default trigger requires 60% smoothed confidence for 1.5 seconds, clears
for 2 seconds before re-arming, and has a 30-second cooldown. A face must be
inside a YOLO person box. Face detection runs only inside the primary person
box, and expression classification runs every third scan. The app targets four
scans per second by default as a balance between viewer responsiveness and CPU
use. These estimates are fallible conversation cues, not claims about a
person's internal emotional state.

Ground-safety evidence must persist for two seconds before it becomes an alert
and must positively clear for two seconds before the alert releases. Missing
depth does not clear a confirmed observation. The app writes the atomic
`/tmp/bracketbot_ground_alert.json` interlock consumed by the navigator; it
never writes to `drive.ctrl` itself. See
[`docs/slam-ground-safety.md`](../../docs/slam-ground-safety.md) for response
policy, limitations, and the required physical calibration protocol.

## Expression accuracy

Measured on 1,750 RAF-DB faces (in-the-wild photos the models never trained
on), cropped with YuNet the same way the robot does:

| Setup | Balanced accuracy | Sad precision at 60% |
| --- | --- | --- |
| Previous: `afew` model, 12%-padded rectangle | 53% | 86% |
| `afew`, square unpadded crop | 56% | 86% |
| `afew` + `va_mtl` average, square crop (current) | 59% | 86% |

Contempt is removed before the argmax. Faces smaller than `--min-face-size`
(40 px) are skipped, because accuracy falls from 57% at 64 px to 44% at 24 px.

## Prepare the models

Export the checked-in YOLO checkpoint on a development machine:

```sh
uv run --extra vision python -c \
  'from ultralytics import YOLO; YOLO("yolo11n-pose.pt").export(format="onnx", imgsz=320, opset=17, dynamic=False)'
```

Place these files in `models/` on the robot:

- `yolo11n-pose.onnx`
- `face_detection_yunet_2026may.onnx`
- `enet_b0_8_best_afew.onnx`
- `enet_b0_8_va_mtl.onnx` (optional; without it only one model runs)

The check-in imports `local_voice.py` and `voice_router.py` from
`~/bbapps/greeter`, and reads `OPENROUTER_API_KEY` from `~/bbapps/.env`.
`scripts/run_robot_local_voice.sh` syncs those files. The robot needs internet
(or the USB proxy's `HTTPS_PROXY`) for the LLM reply. Set `LOCAL_TTS_URL` for
the natural voice; otherwise it uses espeak-ng.

## Run on the robot

```sh
cd ~/bbapps/emotion_greeter
uv run main.py
```

To verify both the BBOS speaker and head camera, play the prompt once, process
three frames, and save the annotated final frame:

```sh
uv run main.py --speak-on-start --max-frames 3 \
  --snapshot /tmp/emotion-greeter.jpg
```

The top-level `bbapps/.autostart` entry makes BBOS start this app on boot once
the directory and models have been deployed.

## Live robot view

While the app is running, open `http://10.42.0.1:8018/` when connected to the
robot's Wi-Fi, or `http://100.121.192.60:8018/` over the robot's current
Tailscale address. The page shows the annotated depth-aligned left-eye feed,
temporary person tracking IDs, ground-safety state and SLAM coordinates, mapped
area, expression confidence, model processing time, total camera-frame age, and
scan rate. The IDs last only for the current process and do not identify who a
person is.
