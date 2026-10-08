import cv2
import numpy as np

from backend.detector import YoloDetector, letterbox, nms
from config import Config
from tests.conftest import SAMPLE


def test_letterbox_keeps_aspect_ratio_and_pads():
    img = np.zeros((100, 200, 3), np.uint8)          # wide image
    out, scale, pad_x, pad_y = letterbox(img, 640)
    assert out.shape == (640, 640, 3)
    assert scale == 3.2 and pad_x == 0 and pad_y == 160


def test_nms_removes_duplicates_but_keeps_separate_boxes():
    boxes = np.array([[0, 0, 100, 100], [5, 5, 105, 105], [300, 300, 400, 400]], float)
    scores = np.array([0.9, 0.8, 0.7])
    assert nms(boxes, scores, 0.45) == [0, 2]


def test_nms_empty():
    assert nms(np.zeros((0, 4)), np.zeros(0), 0.5) == []


def _detector():
    return YoloDetector(Config.MODEL_PATH)


def test_detects_people_and_bus_in_sample_photo():
    detections, ms = _detector().detect(cv2.imread(SAMPLE), conf=0.4)
    labels = [d["label"] for d in detections]
    assert labels.count("person") >= 3
    assert "bus" in labels
    assert ms > 0
    for d in detections:
        x1, y1, x2, y2 = d["box"]
        assert 0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1


def test_focus_filter_only_returns_requested_class():
    detections, _ = _detector().detect(cv2.imread(SAMPLE), conf=0.4, class_ids={5})
    assert detections and all(d["label"] == "bus" for d in detections)


def test_high_confidence_returns_fewer_boxes():
    det, img = _detector(), cv2.imread(SAMPLE)
    low, _ = det.detect(img, conf=0.2)
    high, _ = det.detect(img, conf=0.85)
    assert len(high) < len(low)
