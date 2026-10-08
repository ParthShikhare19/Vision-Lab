import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend import create_app  # noqa: E402

SAMPLE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "frontend", "static", "samples", "bus.jpg")


@pytest.fixture(scope="session")
def sample_bytes():
    with open(SAMPLE, "rb") as f:
        return f.read()


@pytest.fixture()
def client():
    folder = tempfile.mkdtemp()
    app = create_app({
        "DATABASE": os.path.join(folder, "test.db"),
        "SNAPSHOT_DIR": os.path.join(folder, "snaps"),
        "TESTING": True,
        "SAMPLE_EVERY_SECONDS": 0,
    })
    with app.test_client() as c:
        yield c
