import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(MODULE_DIR, "data.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS player (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS session (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    original_filename TEXT,
    source_video_path TEXT,
    num_players INTEGER NOT NULL,
    status TEXT NOT NULL DEFAULT 'uploaded',  -- uploaded / processing / done / error
    error_message TEXT,
    spike_count INTEGER
);

-- Records which players, in which order, were used for a given session's
-- spike-number -> player assignment (players themselves are reusable across
-- sessions for future cross-session history).
CREATE TABLE IF NOT EXISTS session_player (
    session_id INTEGER NOT NULL REFERENCES session(id),
    player_id INTEGER NOT NULL REFERENCES player(id),
    order_index INTEGER NOT NULL,
    PRIMARY KEY (session_id, order_index)
);

CREATE TABLE IF NOT EXISTS spike_video (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL REFERENCES session(id),
    player_id INTEGER NOT NULL REFERENCES player(id),
    spike_number INTEGER NOT NULL,
    player_spike_number INTEGER NOT NULL,
    video_path TEXT NOT NULL,
    start_time REAL,
    duration REAL
);
"""


def init_db():
    with get_connection() as conn:
        conn.executescript(SCHEMA)


@contextmanager
def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def get_or_create_player(conn, name):
    row = conn.execute("SELECT id FROM player WHERE name = ?", (name,)).fetchone()
    if row:
        return row["id"]
    cur = conn.execute("INSERT INTO player (name) VALUES (?)", (name,))
    return cur.lastrowid


def create_session(original_filename, source_video_path, player_names):
    with get_connection() as conn:
        cur = conn.execute(
            "INSERT INTO session (created_at, original_filename, source_video_path, num_players, status) "
            "VALUES (?, ?, ?, ?, 'uploaded')",
            (datetime.utcnow().isoformat(), original_filename, source_video_path, len(player_names)),
        )
        session_id = cur.lastrowid
        for order_index, name in enumerate(player_names):
            player_id = get_or_create_player(conn, name)
            conn.execute(
                "INSERT INTO session_player (session_id, player_id, order_index) VALUES (?, ?, ?)",
                (session_id, player_id, order_index),
            )
        return session_id


def set_source_video_path(session_id, source_video_path):
    with get_connection() as conn:
        conn.execute(
            "UPDATE session SET source_video_path = ? WHERE id = ?",
            (source_video_path, session_id),
        )


def update_session_status(session_id, status, error_message=None, spike_count=None):
    with get_connection() as conn:
        conn.execute(
            "UPDATE session SET status = ?, error_message = ?, "
            "spike_count = COALESCE(?, spike_count) WHERE id = ?",
            (status, error_message, spike_count, session_id),
        )


def get_session(session_id):
    with get_connection() as conn:
        row = conn.execute("SELECT * FROM session WHERE id = ?", (session_id,)).fetchone()
        return dict(row) if row else None


def get_session_players(session_id):
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT p.id, p.name, sp.order_index FROM session_player sp "
            "JOIN player p ON p.id = sp.player_id "
            "WHERE sp.session_id = ? ORDER BY sp.order_index",
            (session_id,),
        ).fetchall()
        return [dict(r) for r in rows]


def save_spike_videos(session_id, spikes):
    """spikes: list of dicts with spike_number, player_id, player_spike_number,
    video_path, start_time, duration (as produced by assign_players_to_spikes)."""
    with get_connection() as conn:
        conn.executemany(
            "INSERT INTO spike_video "
            "(session_id, player_id, spike_number, player_spike_number, video_path, start_time, duration) "
            "VALUES (:session_id, :player_id, :spike_number, :player_spike_number, :video_path, :start_time, :duration)",
            [{**s, "session_id": session_id} for s in spikes],
        )


def get_spike_videos(session_id):
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT sv.*, p.name AS player_name FROM spike_video sv "
            "JOIN player p ON p.id = sv.player_id "
            "WHERE sv.session_id = ? ORDER BY sv.spike_number",
            (session_id,),
        ).fetchall()
        return [dict(r) for r in rows]
