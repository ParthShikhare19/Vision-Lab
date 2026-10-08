"""Application settings. Change values here (or via environment variables)."""
import os

BASE_DIR = os.path.abspath(os.path.dirname(__file__))


class Config:
    MODEL_PATH = os.path.join(BASE_DIR, "models", "yolov8n.onnx")
    INPUT_SIZE = 640            # YOLOv8 was exported for 640 x 640 input
    DEFAULT_CONF = 0.40         # minimum confidence to keep a detection
    NMS_IOU = 0.45              # overlap above which duplicate boxes are removed
    MAX_DETECTIONS = 100
    ORT_THREADS = int(os.environ.get("ORT_THREADS", 0))   # 0 = let ONNX Runtime decide

    DATABASE = os.path.join(BASE_DIR, "instance", "vision.db")
    SNAPSHOT_DIR = os.path.join(BASE_DIR, "instance", "snapshots")
    SAMPLE_EVERY_SECONDS = 2.0  # how often a statistics sample is written to SQLite
    MAX_CONTENT_LENGTH = 12 * 1024 * 1024   # reject uploads larger than 12 MB
