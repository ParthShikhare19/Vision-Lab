"""A small multi-object tracker plus a virtual "tripwire" line counter.

Detection works on single frames and knows nothing about time. The tracker
links detections between frames so the same person keeps the same ID. That
makes two features possible:

  * unique counting  ("12 different people were seen", not "3 per frame")
  * tripwire counting (how many objects crossed a line in each direction)

Matching strategy (per class label):
  1. greedy IoU matching: the biggest overlaps between old tracks and new
     detections are paired first
  2. fallback: leftover pairs whose centres are close are paired too (helps
     when an object moves fast between two frames)
All coordinates are normalised (0..1).
"""
import math


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = ((a[2] - a[0]) * (a[3] - a[1]) +
             (b[2] - b[0]) * (b[3] - b[1]) - inter)
    return inter / union if union > 0 else 0.0


def center(box):
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def _side(a, b, p):
    """Which side of the directed line a->b the point p lies on (cross product)."""
    return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])


def crossing_direction(line, prev_point, new_point):
    """Return 'forward', 'backward' or None.

    A crossing happens when the movement segment prev->new really intersects
    the line segment (not just the infinite line) and the point switches side.
    """
    a, b = (line[0], line[1]), (line[2], line[3])
    s_prev, s_new = _side(a, b, prev_point), _side(a, b, new_point)
    if s_prev == 0 or s_new == 0 or (s_prev > 0) == (s_new > 0):
        return None
    # the movement segment must also straddle the line's own end points
    t_prev = _side(prev_point, new_point, a)
    t_new = _side(prev_point, new_point, b)
    if (t_prev > 0) == (t_new > 0):
        return None
    return "forward" if s_prev < 0 < s_new else "backward"


class Track:
    def __init__(self, track_id, label, box):
        self.id = track_id
        self.label = label
        self.box = box
        self.center = center(box)
        self.prev_center = None
        self.hits = 1
        self.missed = 0
        self.confirmed = False


class IoUTracker:
    def __init__(self, iou_threshold=0.25, max_age=10, min_hits=3):
        self.iou_threshold = iou_threshold
        self.max_age = max_age          # frames a lost track is remembered
        self.min_hits = min_hits        # frames before a track counts as "real"
        self.tracks = []
        self.next_id = 1
        self.unique = {}                # label -> number of distinct objects

    def reset(self):
        self.tracks, self.next_id, self.unique = [], 1, {}

    def update(self, detections):
        """Assign an 'id' and 'confirmed' flag to every detection (in place).
        Returns the list of tracks that were matched this frame."""
        for t in self.tracks:
            t.prev_center = t.center

        unmatched_tracks = set(range(len(self.tracks)))
        unmatched_dets = set(range(len(detections)))
        pairs = []

        # pass 1: IoU
        candidates = []
        for ti, t in enumerate(self.tracks):
            for di, d in enumerate(detections):
                if t.label == d["label"]:
                    score = iou(t.box, d["box"])
                    if score >= self.iou_threshold:
                        candidates.append((score, ti, di))
        self._greedy(sorted(candidates, reverse=True),
                     unmatched_tracks, unmatched_dets, pairs)

        # pass 2: centre distance for what is left
        candidates = []
        for ti in unmatched_tracks:
            t = self.tracks[ti]
            size = max(t.box[2] - t.box[0], t.box[3] - t.box[1])
            for di in unmatched_dets:
                d = detections[di]
                if t.label == d["label"]:
                    dist = math.dist(t.center, center(d["box"]))
                    if dist <= size * 0.75:
                        candidates.append((-dist, ti, di))
        self._greedy(sorted(candidates, reverse=True),
                     unmatched_tracks, unmatched_dets, pairs)

        matched = []
        for ti, di in pairs:
            t, d = self.tracks[ti], detections[di]
            t.box, t.center = d["box"], center(d["box"])
            t.hits += 1
            t.missed = 0
            self._confirm(t)
            matched.append(t)
            d["id"], d["confirmed"] = t.id, t.confirmed

        for ti in unmatched_tracks:
            self.tracks[ti].missed += 1
        self.tracks = [t for t in self.tracks if t.missed <= self.max_age]

        for di in unmatched_dets:
            d = detections[di]
            t = Track(self.next_id, d["label"], d["box"])
            self.next_id += 1
            self._confirm(t)
            self.tracks.append(t)
            d["id"], d["confirmed"] = t.id, t.confirmed
        return matched

    @staticmethod
    def _greedy(candidates, unmatched_tracks, unmatched_dets, pairs):
        for _, ti, di in candidates:
            if ti in unmatched_tracks and di in unmatched_dets:
                unmatched_tracks.discard(ti)
                unmatched_dets.discard(di)
                pairs.append((ti, di))

    def _confirm(self, track):
        if not track.confirmed and track.hits >= self.min_hits:
            track.confirmed = True
            self.unique[track.label] = self.unique.get(track.label, 0) + 1
