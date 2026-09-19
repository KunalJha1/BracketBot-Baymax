# Vision model assets

- `/yolo11n-pose.pt` is the Ultralytics YOLO11 nano pose checkpoint used for
  person detection. It is stored at the repository root because that is the
  default Ultralytics model name. Review the upstream Ultralytics licensing
  terms before redistribution or commercial use.
  SHA-256: `869e83fcdffdc7371fa4e34cd8e51c838cc729571d1635e5141e3075e9319dc0`

- `face_detection_yunet_2026may.onnx` is OpenCV Zoo's YuNet face detector,
  distributed under the MIT License. Source:
  <https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet>
  SHA-256: `ebafce4e3c118d6554634be5c27ab333b4c047a9a8c3faf1d7cf93101c22f0f0`
- `face_landmarker.task` is the MediaPipe FaceLandmarker (478-landmark) bundle
  used to place the forehead and cheek ROI for rPPG, distributed by Google
  under Apache-2.0. Source:
  <https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task>
  SHA-256: `64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff`

- `enet_b0_8_best_afew.onnx` is an EmotiEffLib AffectNet facial-expression
  model, distributed with EmotiEffLib under Apache-2.0. Source:
  <https://github.com/sb-ai-lab/EmotiEffLib/tree/main/models/affectnet_emotions/onnx>
  SHA-256: `7aa2ea31c1311f4f8aa9d3fdb085d418dd4e7a48c4b9ed41df8c044f91d0213f`
- `enet_b0_8_va_mtl.onnx` is the EmotiEffLib multi-task (expression plus
  valence/arousal) AffectNet model, Apache-2.0, from the same source. The robot
  averages it with `enet_b0_8_best_afew.onnx`.
  SHA-256: `c43e056ad388d4a8dc911832b8291435b2af537f967e5870ebd731574ec7e812`
