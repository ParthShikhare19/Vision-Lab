# Model

`yolov8n.onnx` is **YOLOv8 nano** (trained on the COCO dataset, 80 classes) exported to
the ONNX format from the official Ultralytics weights `yolov8n.pt`.

* Source: https://github.com/ultralytics/ultralytics
* License: AGPL-3.0 (Ultralytics). Fine for coursework and research; if you ever ship a
  closed-source product, read the Ultralytics licensing terms first.
* Input: 1 x 3 x 640 x 640 float32 (RGB, 0..1). Output: 1 x 84 x 8400
  (4 box numbers + 80 class scores for each of 8400 candidate boxes).
* To rebuild or upgrade it, see `scripts/export_model.py`.
