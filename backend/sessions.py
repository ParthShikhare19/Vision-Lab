"""Per-browser-tab state (tracker + tripwire counters).

The browser sends a random session id with every frame, so two people using
the app at once do not mix up their counts.
"""
import threading
import time

from .tracker import IoUTracker


class SessionState:
    def __init__(self):
        self.tracker = IoUTracker()
        self.line_total = {"forward": 0, "backward": 0}
        self.line_by_label = {}
        self.last_sample = 0.0
        self.last_used = time.time()
        self.lock = threading.Lock()

    def reset(self):
        self.tracker.reset()
        self.line_total = {"forward": 0, "backward": 0}
        self.line_by_label = {}


class SessionStore:
    def __init__(self, max_age_seconds=3600):
        self._sessions = {}
        self._lock = threading.Lock()
        self._max_age = max_age_seconds

    def get(self, session_id):
        now = time.time()
        with self._lock:
            for key in [k for k, s in self._sessions.items()
                        if now - s.last_used > self._max_age]:
                del self._sessions[key]
            state = self._sessions.setdefault(session_id, SessionState())
            state.last_used = now
            return state
