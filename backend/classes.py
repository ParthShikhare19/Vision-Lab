"""The 80 COCO classes YOLOv8 can detect, plus quick-pick presets."""

COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]

PRESETS = {
    "all": None,
    "people": ["person"],
    "vehicles": ["bicycle", "car", "motorcycle", "airplane", "bus", "train",
                 "truck", "boat"],
    "animals": ["bird", "cat", "dog", "horse", "sheep", "cow", "elephant",
                "bear", "zebra", "giraffe"],
    "everyday": ["backpack", "umbrella", "handbag", "suitcase", "bottle", "cup",
                 "chair", "laptop", "mouse", "keyboard", "cell phone", "book"],
}


def ids_for_focus(focus):
    """Turn a preset name or a comma separated list of class names into a set
    of class ids. Returns None for 'all classes'."""
    if not focus or focus == "all":
        return None
    names = PRESETS.get(focus)
    if names is None:
        names = [n.strip() for n in focus.split(",") if n.strip()]
    ids = {COCO_CLASSES.index(n) for n in names if n in COCO_CLASSES}
    return ids or None
