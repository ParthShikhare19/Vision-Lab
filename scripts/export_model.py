"""OPTIONAL: re-create (or upgrade) the ONNX model from the original weights.

The project already ships with models/yolov8n.onnx, so you do NOT need this to
run the demo. Use it if you want a bigger, more accurate model:

    pip install ultralytics onnx
    python scripts/export_model.py yolov8s      # n=nano (fast), s=small, m=medium

Then point MODEL_PATH in config.py at the new file.
"""
import os
import shutil
import sys

from ultralytics import YOLO

name = sys.argv[1] if len(sys.argv) > 1 else "yolov8n"
model = YOLO(name + ".pt")                    # downloads the weights if missing
exported = model.export(format="onnx", imgsz=640, simplify=False, opset=12)
target = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "models", name + ".onnx")
shutil.copy(exported, target)
print("Saved", target)
