"""HTTP routes. `pages` serves the UI, `api` is the JSON REST API."""
import csv
import io
import os
import re
import time
import uuid

import cv2
import numpy as np
from flask import (Blueprint, Response, current_app, jsonify, render_template,
                   request, send_from_directory)

from . import database
from .classes import COCO_CLASSES, PRESETS, ids_for_focus
from .tracker import crossing_direction

pages = Blueprint("pages", __name__)
api = Blueprint("api", __name__)

SNAPSHOT_NAME = re.compile(r"^[0-9a-f]{32}\.jpg$")


def _error(message, status=400):
    return jsonify({"error": message}), status


def _float_arg(name, default, low, high):
    try:
        value = float(request.args.get(name, default))
    except ValueError:
        value = default
    return min(max(value, low), high)


# ---------------------------------------------------------------- pages ----
@pages.route("/")
def index():
    return render_template("index.html")


@pages.route("/snapshots/<name>")
def snapshot_file(name):
    if not SNAPSHOT_NAME.match(name):
        return _error("Not found.", 404)
    return send_from_directory(current_app.config["SNAPSHOT_DIR"], name)


# ------------------------------------------------------------------ api ----
@api.get("/health")
def health():
    detector = current_app.extensions["detector"]
    return jsonify({
        "status": "ok",
        "model": os.path.basename(current_app.config["MODEL_PATH"]),
        "input_size": detector.size,
        "providers": detector.session.get_providers(),
    })


@api.get("/classes")
def classes():
    return jsonify({"classes": COCO_CLASSES, "presets": PRESETS})


@api.post("/detect")
def detect():
    """Body = raw JPEG/PNG bytes of one frame. Query parameters:
    conf, focus, track (0/1), session, source, line (x1,y1,x2,y2), flip (0/1)."""
    data = request.get_data()
    if not data:
        return _error("Send an image in the request body.")
    frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        return _error("That file could not be read as an image.")

    conf = _float_arg("conf", current_app.config["DEFAULT_CONF"], 0.05, 0.95)
    class_ids = ids_for_focus(request.args.get("focus", "all"))
    detector = current_app.extensions["detector"]
    detections, infer_ms = detector.detect(frame, conf=conf, class_ids=class_ids)

    counts = {}
    for d in detections:
        counts[d["label"]] = counts.get(d["label"], 0) + 1

    response = {
        "width": int(frame.shape[1]),
        "height": int(frame.shape[0]),
        "inference_ms": round(infer_ms, 1),
        "detections": detections,
        "counts": counts,
        "total": len(detections),
    }

    session_id = request.args.get("session", "default")[:64]
    state = current_app.extensions["sessions"].get(session_id)
    track = request.args.get("track", "0") == "1"

    with state.lock:
        if track:
            matched = state.tracker.update(detections)
            line = _parse_line(request.args.get("line"))
            if line:
                flip = request.args.get("flip", "0") == "1"
                for t in matched:
                    if not t.confirmed or t.prev_center is None:
                        continue
                    direction = crossing_direction(line, t.prev_center, t.center)
                    if direction is None:
                        continue
                    if flip:
                        direction = "backward" if direction == "forward" else "forward"
                    state.line_total[direction] += 1
                    entry = state.line_by_label.setdefault(
                        t.label, {"forward": 0, "backward": 0})
                    entry[direction] += 1
                    database.add_event("line", t.label,
                                       "crossed " + ("in" if direction == "forward" else "out"), 1)
            response["unique"] = dict(state.tracker.unique)
            response["unique_total"] = sum(state.tracker.unique.values())
            response["line"] = {
                "in": state.line_total["forward"],
                "out": state.line_total["backward"],
                "by_label": state.line_by_label,
            }

        # write one statistics sample every few seconds (not every frame)
        now = time.time()
        if detections and now - state.last_sample >= current_app.config["SAMPLE_EVERY_SECONDS"]:
            state.last_sample = now
            database.add_sample(session_id, request.args.get("source", "unknown")[:16],
                                counts, infer_ms)
    return jsonify(response)


def _parse_line(raw):
    if not raw:
        return None
    try:
        values = [float(v) for v in raw.split(",")]
    except ValueError:
        return None
    if len(values) != 4 or any(v < -0.01 or v > 1.01 for v in values):
        return None
    return values


@api.post("/session/reset")
def reset_session():
    session_id = request.args.get("session", "default")[:64]
    current_app.extensions["sessions"].get(session_id).reset()
    return jsonify({"reset": True})


@api.get("/stats")
def stats():
    return jsonify(database.get_stats())


# ---- events ----
@api.get("/events")
def events():
    try:
        limit = min(int(request.args.get("limit", 30)), 500)
    except ValueError:
        return _error("limit must be a number.")
    return jsonify(database.list_events(limit))


@api.post("/events")
def create_event():
    body = request.get_json(silent=True) or {}
    if body.get("kind") != "alert":
        return _error("Only alert events can be created from the browser.")
    label = str(body.get("label", ""))[:40]
    if label not in COCO_CLASSES:
        return _error("Unknown class.")
    detail = str(body.get("detail", ""))[:120]
    try:
        count = int(body.get("count", 0))
    except (TypeError, ValueError):
        count = 0
    return jsonify({"id": database.add_event("alert", label, detail, count)}), 201


@api.delete("/events")
def delete_events():
    database.clear_events()
    database.clear_samples()
    return jsonify({"cleared": True})


@api.get("/export/events.csv")
def export_events():
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["id", "time", "kind", "label", "detail", "count"])
    for e in reversed(database.list_events(limit=100000)):
        writer.writerow([e["id"], e["ts"], e["kind"], e["label"], e["detail"], e["count"]])
    return Response(buf.getvalue(), mimetype="text/csv",
                    headers={"Content-Disposition": "attachment; filename=detection_events.csv"})


# ---- snapshots ----
@api.get("/snapshots")
def snapshots():
    return jsonify(database.list_snapshots())


@api.post("/snapshots")
def create_snapshot():
    file = request.files.get("image")
    if file is None:
        return _error("Attach the snapshot as 'image'.")
    frame = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        return _error("That file could not be read as an image.")
    folder = current_app.config["SNAPSHOT_DIR"]
    os.makedirs(folder, exist_ok=True)
    name = uuid.uuid4().hex + ".jpg"
    cv2.imwrite(os.path.join(folder, name), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    snapshot_id = database.add_snapshot(name, request.form.get("summary", "")[:200])
    return jsonify({"id": snapshot_id, "url": "/snapshots/" + name}), 201


@api.delete("/snapshots/<int:snapshot_id>")
def remove_snapshot(snapshot_id):
    row = database.get_snapshot(snapshot_id)
    if row is None:
        return _error("Snapshot not found.", 404)
    path = os.path.join(current_app.config["SNAPSHOT_DIR"], row["filename"])
    if os.path.exists(path):
        os.remove(path)
    database.delete_snapshot(snapshot_id)
    return jsonify({"deleted": snapshot_id})
