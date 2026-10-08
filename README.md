# Vision Lab: real-time object detection, tracking and counting

A full-stack web app that finds objects in a **webcam, a video file or a photo**, follows
each one with an ID, counts **unique** objects, counts how many **cross a line you draw**,
raises **alerts**, and logs everything to a database with a live analytics panel.

**Open-source tool at the core: [YOLOv8](https://github.com/ultralytics/ultralytics)**
(state-of-the-art object detector) running on **ONNX Runtime**, with **OpenCV** for image
handling. The model file is included, so it works **offline**: no API keys, no cloud, no
GPU, and no PyTorch install (the whole setup is about 100 MB).

## What makes it more than a "detect and draw boxes" demo
| Feature | What it shows |
|---|---|
| Live detection on webcam / video / photo | Real-time computer vision, 80 object types |
| Detection pipeline written from scratch (`backend/detector.py`) | Letterboxing, tensor decoding, confidence filter, Non-Maximum Suppression |
| Multi-object **tracker** (`backend/tracker.py`) | Persistent IDs, "17 different people" instead of "3 per frame" |
| **Tripwire counting**: drag a line on the video | In/out counting with geometry (cross products, segment intersection) |
| **Alert rule** ("alert when 3+ people") | Event-driven logic, banner, optional beep, saved to DB |
| Analytics: timeline, class averages, latency | SQLite + hand-drawn SVG charts (no chart library) |
| Snapshots gallery, CSV export | Full CRUD + file handling |
| 23 automated tests | Unit tests (NMS, tracker, geometry) and API tests with a real model |

## Tech stack
| Layer | Technology |
|---|---|
| Frontend | HTML5, CSS3, vanilla JavaScript, Canvas, `getUserMedia` (webcam), SVG |
| Backend | Python 3, Flask REST API |
| AI model | YOLOv8n (Ultralytics, COCO-trained) exported to ONNX |
| Inference | ONNX Runtime (CPU), NumPy, OpenCV |
| Database | SQLite (Python standard library) |
| Testing | pytest |

## Folder structure
```
vision-lab/
├── app.py                    # entry point:  python app.py
├── config.py                 # settings (model path, thresholds, limits)
├── requirements.txt
├── README.md · LICENSE
├── models/
│   ├── yolov8n.onnx          # the YOLOv8 model (12 MB)
│   └── README.md             # where it came from + license
├── backend/
│   ├── __init__.py           # Flask app factory, loads the model once
│   ├── detector.py           # YOLOv8 + ONNX Runtime pipeline (letterbox, NMS)
│   ├── tracker.py            # IoU tracker + tripwire crossing geometry
│   ├── sessions.py           # per-browser-tab tracker state
│   ├── classes.py            # 80 COCO class names + presets
│   ├── database.py           # SQLite: samples, events, snapshots
│   └── routes.py             # pages + REST API
├── frontend/
│   ├── templates/index.html
│   └── static/
│       ├── css/style.css
│       ├── js/app.js         # camera, loop, canvas drawing, charts
│       └── samples/bus.jpg   # demo photo
├── scripts/export_model.py   # optional: rebuild / upgrade the model
├── tests/                    # pytest suite
└── instance/                 # database + snapshots are created here at run time
```

## Setup and run
Requires **Python 3.9 or newer** and a modern browser (Chrome, Edge or Firefox).

**Windows**
```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python app.py
```
**macOS / Linux**
```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python app.py
```
Open **http://127.0.0.1:5000** (use `127.0.0.1` or `localhost`: browsers only allow the
webcam on those addresses over plain HTTP).

Run the tests: `pytest -v`

## How to use it
1. **Webcam tab**: press *Start camera*, allow access. Boxes appear on live video.
2. **Video file tab**: choose any video (mp4, webm...). The file stays in your browser; only single frames are sent to the server.
3. **Photo tab**: press *Use sample photo* or choose your own image.
4. **Drag on the picture** to draw a yellow counting line. The arrow shows the "in" side. Tick *Swap direction* to reverse it.
5. Use the **Confidence** slider and **Look for** menu to filter results.
6. Set an **alert rule** (for example, person ≥ 2) and watch the banner and the event log.
7. **Save snapshot** stores the annotated frame; **Download CSV** exports the event log.

## Suggested demo script (4 minutes)
1. Open the app, point out the green *yolov8n ready* badge. Say: "The model runs locally, no internet or cloud."
2. **Photo tab → sample photo.** Show detections, then switch *Look for* to Vehicles to show filtering. Move the confidence slider.
3. **Webcam tab → Start.** Hold up a phone, a bottle, a book. Show live counts and the fps/ms badges.
4. **Drag a counting line** across the picture and walk through it (or play a street video in the Video tab). Show *crossed in/out* and the **unique** counter. Explain that tracking gives the IDs.
5. Set the **alert rule** to *person ≥ 2*, get a friend into frame, show the banner.
6. Click **Save snapshot**, then scroll to the **analytics, event log and gallery**. Download the CSV.
7. Open `backend/detector.py` and `backend/tracker.py` and show that the AI pipeline is real code, then run `pytest -v`.

> Tip: do a full dry run once before the demo (allow the camera in your browser) and keep a
> short video clip handy as a backup in case the camera or room lighting misbehaves.

## How it works (viva notes)
**Pipeline per frame**
```
browser frame (JPEG) → /api/detect → letterbox to 640×640 → YOLOv8 (ONNX Runtime)
   → 8400 candidate boxes → drop low confidence → NMS → boxes in 0..1 coordinates
   → tracker (IDs) → line-crossing check → JSON → browser draws boxes on canvas
```
* **YOLO ("You Only Look Once")** is a single-stage detector: one forward pass of a
  convolutional network predicts boxes and classes together, which is why it is fast enough for video.
* **Output tensor 1×84×8400**: 8400 candidate boxes, each with 4 numbers (centre x, y, width, height) and 80 class scores. YOLOv8 is *anchor-free*, so there is no separate objectness score.
* **Letterboxing** resizes the image to fit 640×640 without stretching it, padding the rest with grey, so shapes are not distorted. The code later undoes the scale and padding.
* **Non-Maximum Suppression (NMS)**: many candidate boxes cover the same object. Keep the highest-scoring one and delete boxes that overlap it by more than an IoU threshold (0.45).
* **IoU (Intersection over Union)** = overlap area ÷ combined area. Used by both NMS and the tracker.
* **Tracking**: detection has no memory. The tracker matches boxes between frames by highest IoU (with a centre-distance fallback), keeps the ID, and only counts an object as "unique" after it was seen in 3 frames (this filters flicker).
* **Tripwire**: for the line A→B, the sign of the cross product `(B−A)×(P−A)` says which side point P is on. If an object's centre switches sides *and* its movement segment crosses the actual line segment, it is counted as in or out, depending on the sign change.
* **ONNX** is an open model format. ONNX Runtime executes it on CPU without needing PyTorch, so installation is small and fast.
* **Why frames are sent one at a time**: the browser waits for each answer before sending the next frame, so the frame rate adapts to the CPU and the server never queues up.

**Likely questions**
* *Why not train your own model?* YOLOv8 is pre-trained on COCO (about 118,000 images, 80 classes). Training from scratch needs a GPU and days; using and extending a pre-trained model is standard practice (transfer learning).
* *Limitations?* It only knows the 80 COCO classes; small, distant or occluded objects can be missed; the simple tracker can swap IDs when people cross paths (production systems add appearance features, as in DeepSORT).
* *Privacy?* Everything runs locally. Video never leaves the machine.

## REST API
| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/detect` | Body = JPEG/PNG bytes. Query: `conf`, `focus`, `track`, `session`, `line`, `flip` |
| POST | `/api/session/reset` | Reset tracker and counters for a session |
| GET | `/api/health` | Model status |
| GET | `/api/classes` | The 80 class names and presets |
| GET | `/api/stats` | Analytics (timeline, class averages, latency) |
| GET / POST / DELETE | `/api/events` | Alert and line-crossing log |
| GET | `/api/export/events.csv` | Download the log |
| GET / POST | `/api/snapshots` | List / save annotated frames |
| DELETE | `/api/snapshots/<id>` | Delete a snapshot |

Try it without the UI:
`curl -X POST --data-binary @frontend/static/samples/bus.jpg "http://127.0.0.1:5000/api/detect?conf=0.4"`

## Performance tips
* Typical speed: **5–15 fps** on a laptop CPU with the nano model.
* Slow machine? Close other heavy apps, lower the camera resolution, or set the `ORT_THREADS` environment variable (for example `ORT_THREADS=4`).
* More accuracy? Run `python scripts/export_model.py yolov8s` and set `MODEL_PATH` in `config.py` (slower but better).

## Troubleshooting
| Problem | Fix |
|---|---|
| "Camera unavailable" | Allow camera access in the browser; close Zoom/Teams; open the app via `127.0.0.1` or `localhost`; or use the Video / Photo tab |
| Port 5000 busy (macOS AirPlay) | Change the port at the bottom of `app.py` |
| `pip install` fails on a very new Python | Use Python 3.9–3.12 |
| Boxes lag behind fast motion | Normal at low fps; boxes update as soon as the server answers |

## Future work
Custom-trained classes (transfer learning), DeepSORT-style appearance tracking, polygon zones, heat-maps, GPU execution provider, Docker deployment.

## License
Project code: MIT (see `LICENSE`). YOLOv8 model: AGPL-3.0 (see `models/README.md`).
