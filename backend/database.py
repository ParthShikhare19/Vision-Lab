"""SQLite storage (Python standard library only)."""
import json
import sqlite3

from flask import current_app, g

SCHEMA = """
CREATE TABLE IF NOT EXISTS samples (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          TEXT    NOT NULL DEFAULT (datetime('now', 'localtime')),
    session_id  TEXT    NOT NULL,
    source      TEXT    NOT NULL,
    total       INTEGER NOT NULL,
    counts      TEXT    NOT NULL,
    infer_ms    REAL    NOT NULL
);
CREATE TABLE IF NOT EXISTS events (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    ts      TEXT    NOT NULL DEFAULT (datetime('now', 'localtime')),
    kind    TEXT    NOT NULL,
    label   TEXT    NOT NULL,
    detail  TEXT    NOT NULL DEFAULT '',
    count   INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS snapshots (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    ts        TEXT NOT NULL DEFAULT (datetime('now', 'localtime')),
    filename  TEXT NOT NULL,
    summary   TEXT NOT NULL DEFAULT ''
);
"""


def get_db():
    if "db" not in g:
        g.db = sqlite3.connect(current_app.config["DATABASE"])
        g.db.row_factory = sqlite3.Row
    return g.db


def close_db(_exc=None):
    db = g.pop("db", None)
    if db is not None:
        db.close()


def init_app(app):
    app.teardown_appcontext(close_db)
    with app.app_context():
        get_db().executescript(SCHEMA)
        get_db().commit()


# ---- samples -------------------------------------------------------------
def add_sample(session_id, source, counts, infer_ms):
    db = get_db()
    db.execute(
        "INSERT INTO samples (session_id, source, total, counts, infer_ms) "
        "VALUES (?, ?, ?, ?, ?)",
        (session_id, source, sum(counts.values()), json.dumps(counts), infer_ms))
    db.commit()


def get_stats():
    db = get_db()
    rows = db.execute(
        "SELECT ts, total, counts, infer_ms FROM samples ORDER BY id DESC LIMIT 300"
    ).fetchall()
    rows = list(reversed(rows))
    class_totals = {}
    for r in rows:
        for label, n in json.loads(r["counts"]).items():
            class_totals[label] = class_totals.get(label, 0) + n
    n = len(rows)
    top = sorted(class_totals.items(), key=lambda kv: kv[1], reverse=True)[:8]
    return {
        "samples": db.execute("SELECT COUNT(*) FROM samples").fetchone()[0],
        "events": db.execute("SELECT COUNT(*) FROM events").fetchone()[0],
        "snapshots": db.execute("SELECT COUNT(*) FROM snapshots").fetchone()[0],
        "avg_infer_ms": round(sum(r["infer_ms"] for r in rows) / n, 1) if n else 0,
        "timeline": [r["total"] for r in rows[-60:]],
        "top_classes": [{"label": k, "avg_per_sample": round(v / n, 2)} for k, v in top],
    }


def clear_samples():
    db = get_db()
    db.execute("DELETE FROM samples")
    db.commit()


# ---- events --------------------------------------------------------------
def add_event(kind, label, detail="", count=0):
    db = get_db()
    cur = db.execute(
        "INSERT INTO events (kind, label, detail, count) VALUES (?, ?, ?, ?)",
        (kind, label, detail, count))
    db.commit()
    return cur.lastrowid


def list_events(limit=50):
    return [dict(r) for r in get_db().execute(
        "SELECT * FROM events ORDER BY id DESC LIMIT ?", (limit,)).fetchall()]


def clear_events():
    db = get_db()
    db.execute("DELETE FROM events")
    db.commit()


# ---- snapshots -----------------------------------------------------------
def add_snapshot(filename, summary):
    db = get_db()
    cur = db.execute("INSERT INTO snapshots (filename, summary) VALUES (?, ?)",
                     (filename, summary))
    db.commit()
    return cur.lastrowid


def list_snapshots(limit=24):
    return [dict(r) for r in get_db().execute(
        "SELECT * FROM snapshots ORDER BY id DESC LIMIT ?", (limit,)).fetchall()]


def get_snapshot(snapshot_id):
    row = get_db().execute("SELECT * FROM snapshots WHERE id = ?",
                           (snapshot_id,)).fetchone()
    return dict(row) if row else None


def delete_snapshot(snapshot_id):
    db = get_db()
    db.execute("DELETE FROM snapshots WHERE id = ?", (snapshot_id,))
    db.commit()
