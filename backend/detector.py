"""YOLOv8 object detector running on ONNX Runtime (CPU).

Everything between "image in" and "boxes out" is written here on purpose so it
can be explained step by step:

    1. letterbox()   resize + pad the frame to 640x640 without distorting it
    2. model.run()   one forward pass of YOLOv8 -> tensor of shape (1, 84, 8400)
    3. decode        8400 candidate boxes, each = 4 box numbers + 80 class scores
    4. threshold     drop candidates below the confidence setting
    5. nms()         Non-Maximum Suppression removes duplicate boxes
    6. un-letterbox  map boxes back to the original frame, normalise to 0..1
"""
import time

import cv2
import numpy as np
import onnxruntime as ort

from .classes import COCO_CLASSES


def letterbox(img, size=640, color=114):
    """Resize keeping aspect ratio, then pad to a square. Returns the padded
    image plus the scale and padding needed to undo the transformation."""
    h, w = img.shape[:2]
    scale = min(size / h, size / w)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    pad_x, pad_y = (size - new_w) // 2, (size - new_h) // 2
    canvas = np.full((size, size, 3), color, dtype=np.uint8)
    canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
    return canvas, scale, pad_x, pad_y


def nms(boxes, scores, iou_threshold):
    """Greedy Non-Maximum Suppression. boxes are (N, 4) as x1, y1, x2, y2.
    Returns the indices of the boxes to keep."""
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        best = order[0]
        keep.append(int(best))
        rest = order[1:]
        ix1 = np.maximum(x1[best], x1[rest])
        iy1 = np.maximum(y1[best], y1[rest])
        ix2 = np.minimum(x2[best], x2[rest])
        iy2 = np.minimum(y2[best], y2[rest])
        inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
        iou = inter / (areas[best] + areas[rest] - inter + 1e-9)
        order = rest[iou <= iou_threshold]
    return keep


class YoloDetector:
    def __init__(self, model_path, input_size=640, nms_iou=0.45,
                 max_detections=100, threads=0):
        options = ort.SessionOptions()
        if threads:
            options.intra_op_num_threads = threads
        self.session = ort.InferenceSession(
            model_path, sess_options=options,
            providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.size = input_size
        self.nms_iou = nms_iou
        self.max_detections = max_detections

    def detect(self, bgr, conf=0.4, class_ids=None):
        """Run detection on a BGR image (as read by OpenCV).

        Returns (detections, inference_ms). Each detection is a dict with
        label, class_id, confidence and box = [x1, y1, x2, y2] normalised to
        the 0..1 range of the original image.
        """
        h, w = bgr.shape[:2]
        padded, scale, pad_x, pad_y = letterbox(bgr, self.size)

        # BGR -> RGB, HWC -> CHW, 0..255 -> 0..1, add batch dimension
        blob = padded[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
        blob = np.ascontiguousarray(blob[None])

        start = time.perf_counter()
        output = self.session.run(None, {self.input_name: blob})[0]
        inference_ms = (time.perf_counter() - start) * 1000.0

        if output.shape[1] > output.shape[2]:        # tolerate (1, N, 84) exports
            output = output.transpose(0, 2, 1)
        preds = output[0].T                          # (8400, 84)
        boxes_cxcywh, class_scores = preds[:, :4], preds[:, 4:]

        if class_ids is not None:                    # "focus" filter
            blocked = np.ones(class_scores.shape[1], dtype=bool)
            blocked[list(class_ids)] = False
            class_scores = class_scores.copy()
            class_scores[:, blocked] = 0.0

        best_class = class_scores.argmax(axis=1)
        best_score = class_scores[np.arange(len(best_class)), best_class]
        keep = best_score >= conf
        if not keep.any():
            return [], inference_ms

        boxes_cxcywh = boxes_cxcywh[keep]
        scores = best_score[keep]
        classes = best_class[keep]

        # centre/size -> corner coordinates, then undo the letterbox
        cx, cy, bw, bh = boxes_cxcywh.T
        xyxy = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], axis=1)
        xyxy[:, [0, 2]] = (xyxy[:, [0, 2]] - pad_x) / scale
        xyxy[:, [1, 3]] = (xyxy[:, [1, 3]] - pad_y) / scale
        xyxy[:, [0, 2]] = np.clip(xyxy[:, [0, 2]], 0, w)
        xyxy[:, [1, 3]] = np.clip(xyxy[:, [1, 3]], 0, h)

        # class-aware NMS: shift each class far apart so boxes of different
        # classes can never suppress each other
        shifted = xyxy + (classes[:, None] * 10000.0)
        kept = nms(shifted, scores, self.nms_iou)[: self.max_detections]

        detections = []
        for i in kept:
            x1, y1, x2, y2 = xyxy[i]
            if x2 - x1 < 1 or y2 - y1 < 1:
                continue
            class_id = int(classes[i])
            detections.append({
                "label": COCO_CLASSES[class_id],
                "class_id": class_id,
                "confidence": round(float(scores[i]), 3),
                "box": [round(float(x1 / w), 4), round(float(y1 / h), 4),
                        round(float(x2 / w), 4), round(float(y2 / h), 4)],
            })
        return detections, inference_ms
