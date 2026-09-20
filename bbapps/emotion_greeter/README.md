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
7. after a confidently sad-looking expression, open with one of a handful of
   short lines ("Hey, you okay? What's going on?"), listen for the answer
   (whisper.cpp), and reply with the OpenRouter LLM for up to three turns
   (`check_in.py`).

If whisper.cpp, espeak-ng, or the greeter voice modules are missing, the app
falls back to playing `sad_prompt.wav`. If the LLM cannot be reached it gives
a short fixed, supportive reply. Use `--no-check-in` to keep the recorded
prompt only.

The default trigger requires 60% smoothed confidence for 0.6 seconds, clears
for 1.5 seconds before re-arming, and has a 30-second cooldown. A reading at
or above `--sad-instant-confidence` (85%) skips the hold and speaks on the
first frame, because waiting out a hold on a face the models are already sure
about is what made the robot feel slow. A face must be inside a YOLO person
box. Face detection runs only inside the primary person box, and expression
classification runs every third scan until a reading leans sad, after which
every scan is classified so the evidence builds at the full four-per-second
rate. These estimates are fallible conversation cues, not claims about a
person's internal emotional state.

## Response latency

The cue-to-voice path is tuned to answer a frown in about a second and a
quarter, measured by simulating the filter, trigger and speech path at the
default four scans per second:

| Frown strength (raw model sadness) | Before | Now |
| --- | --- | --- |
| 0.55 | never triggers | never triggers |
| 0.65 | 7.6 s | 1.5 s |
| 0.75 | 6.2 s | 1.3 s |
| 0.90 | 4.7 s | 1.3 s |

Three changes account for it, and none of them lowers the confidence bar — a
weak frown that never spoke before still never speaks:

- **Asymmetric smoothing.** `--expression-smoothing` (0.25) still governs
  falling evidence, but rising evidence follows `--expression-attack` (0.55).
  A symmetric 0.25 filter needs four classifications to cross 0.6 from a cold
  start; this needs two.
- **Full-rate classification once a reading leans sad.** The every-third-scan
  interval saves CPU while nothing is happening and now gets out of the way
  the moment it matters.
- **Pre-rendered openers.** Every opening line is synthesized at startup, so
  the gap between the trigger and the first word is the speaker buffer rather
  than a TTS round trip. Replies are then synthesized sentence by sentence,
  so the first sentence plays while the rest is still rendering.

The check-in also runs with web-search tools off and a 110-token reply budget,
so the LLM returns the couple of short sentences the prompt asks for without
an extra tool round trip. `--check-in-trailing-silence` (0.7 s, was 1.2 s)
decides when an answer has ended; raise it if the robot starts replying over
people who pause mid-sentence. The pause after Baymax speaks is deliberately
not shortened, because the mic must not pick up the tail of his own voice.

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
(32 px) are skipped, because accuracy falls from 57% at 64 px to 51% at 32 px
and 44% at 24 px. The robot's 512x384 `camera.rect` shows faces at roughly
30-45 px from a couple of metres, so stand close for reliable cues.

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
