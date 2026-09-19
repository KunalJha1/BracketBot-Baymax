# Robot emotion greeter

This app runs the complete vision-to-voice path on the robot:

1. read the left eye from `camera.head.rgb`;
2. detect people with the repository's YOLO11 pose model;
3. locate the primary face with YuNet;
4. estimate its visible expression with EmotiEffLib's ONNX model; and
5. play `sad_prompt.wav` through `speaker.audio` after a sustained,
   confidently sad-looking expression.

The default trigger requires 60% smoothed confidence for 1.5 seconds, clears
for 2 seconds before re-arming, and has a 30-second cooldown. A face must be
inside a YOLO person box. Face detection runs only inside the primary person
box, and expression classification runs every third scan. The app targets four
scans per second by default as a balance between viewer responsiveness and CPU
use. These estimates are fallible conversation cues, not claims about a
person's internal emotional state.

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
Tailscale address. The page shows the annotated left-eye feed, temporary
person tracking IDs, expression confidence, model processing time, total
camera-frame age, and scan rate. The IDs last only for the current process and
do not identify who a person is.
