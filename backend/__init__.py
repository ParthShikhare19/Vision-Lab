"""Flask application factory."""
import os

from flask import Flask

from config import Config
from . import database
from .detector import YoloDetector
from .routes import api, pages
from .sessions import SessionStore


def create_app(test_config=None):
    app = Flask(
        __name__,
        template_folder=os.path.join("..", "frontend", "templates"),
        static_folder=os.path.join("..", "frontend", "static"),
    )
    app.config.from_object(Config)
    if test_config:
        app.config.update(test_config)

    os.makedirs(os.path.dirname(app.config["DATABASE"]), exist_ok=True)
    os.makedirs(app.config["SNAPSHOT_DIR"], exist_ok=True)
    database.init_app(app)

    # The model is loaded once at start-up and shared by every request.
    app.extensions["detector"] = YoloDetector(
        app.config["MODEL_PATH"],
        input_size=app.config["INPUT_SIZE"],
        nms_iou=app.config["NMS_IOU"],
        max_detections=app.config["MAX_DETECTIONS"],
        threads=app.config["ORT_THREADS"],
    )
    app.extensions["sessions"] = SessionStore()

    app.register_blueprint(pages)
    app.register_blueprint(api, url_prefix="/api")
    return app
