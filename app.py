import os
import threading

from flask import Flask, render_template, request, redirect, url_for, jsonify, send_from_directory
from werkzeug.utils import secure_filename

import db
from spike_clips import extract_spikes

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_ROOT = os.path.join(MODULE_DIR, "uploads")
OUTPUT_ROOT = os.path.join(MODULE_DIR, "output", "sessions")
ALLOWED_EXTENSIONS = {"mp4", "mov", "m4v", "avi"}
MAX_CONTENT_LENGTH = 500 * 1024 * 1024  # 500MB
MAX_PLAYERS = 10

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

# session_id -> {"stage": "detect"|"extract", "current": int, "total": int}
PROGRESS = {}
PROGRESS_LOCK = threading.Lock()


def allowed_file(filename):
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    return ext in ALLOWED_EXTENSIONS


def assign_players_to_spikes(spikes, session_players):
    """spikes: extract_spikes() output, in detection order. session_players: ordered
    list of {"id", "name", "order_index"}. Assigns players round-robin by detection
    order (spike 1 -> player[0], spike 2 -> player[1], ..., spike n+1 -> player[0]),
    and tracks each player's own spike count separately from the overall spike number.
    """
    n = len(session_players)
    per_player_count = {p["id"]: 0 for p in session_players}
    assigned = []
    for spike in spikes:
        player = session_players[(spike["spike_number"] - 1) % n]
        per_player_count[player["id"]] += 1
        assigned.append({
            "spike_number": spike["spike_number"],
            "player_id": player["id"],
            "player_spike_number": per_player_count[player["id"]],
            "video_path": os.path.basename(spike["video_path"]),
            "start_time": spike["start_time"],
            "duration": spike["duration"],
        })
    return assigned


def run_analysis(session_id, video_path):
    try:
        db.update_session_status(session_id, "processing")

        def progress_cb(stage, current, total):
            with PROGRESS_LOCK:
                PROGRESS[session_id] = {"stage": stage, "current": current, "total": total}

        output_dir = os.path.join(OUTPUT_ROOT, str(session_id))
        spikes = extract_spikes(video_path, output_dir, max_frames=None, progress_callback=progress_cb)

        session_players = db.get_session_players(session_id)
        assigned = assign_players_to_spikes(spikes, session_players)
        db.save_spike_videos(session_id, assigned)
        db.update_session_status(session_id, "done", spike_count=len(assigned))
    except Exception as exc:  # noqa: BLE001 - surface any failure to the UI
        db.update_session_status(session_id, "error", error_message=str(exc))
    finally:
        with PROGRESS_LOCK:
            PROGRESS.pop(session_id, None)


@app.route("/")
def index():
    return render_template("index.html", default_num_players=3)


@app.route("/upload", methods=["POST"])
def upload():
    file = request.files.get("video")
    if not file or file.filename == "":
        return render_template("index.html", default_num_players=3, error="動画ファイルを選択してください。")
    if not allowed_file(file.filename):
        return render_template("index.html", default_num_players=3, error="対応していない動画形式です（mp4 / mov / m4v / avi）。")

    try:
        num_players = int(request.form.get("num_players", 0))
    except ValueError:
        num_players = 0
    if not (1 <= num_players <= MAX_PLAYERS):
        return render_template("index.html", default_num_players=3, error=f"選手人数は1〜{MAX_PLAYERS}人で指定してください。")

    player_names = []
    for i in range(1, num_players + 1):
        name = request.form.get(f"player_{i}", "").strip() or f"選手{i}"
        player_names.append(name)

    original_filename = secure_filename(file.filename)
    session_id = db.create_session(original_filename, None, player_names)

    session_upload_dir = os.path.join(UPLOAD_ROOT, str(session_id))
    os.makedirs(session_upload_dir, exist_ok=True)
    video_path = os.path.join(session_upload_dir, original_filename)
    file.save(video_path)
    db.set_source_video_path(session_id, video_path)

    thread = threading.Thread(target=run_analysis, args=(session_id, video_path), daemon=True)
    thread.start()

    return redirect(url_for("session_status_page", session_id=session_id))


@app.errorhandler(413)
def too_large(_exc):
    return render_template("index.html", default_num_players=3, error="動画ファイルが大きすぎます（上限500MB）。"), 413


@app.route("/session/<int:session_id>")
def session_status_page(session_id):
    session = db.get_session(session_id)
    if not session:
        return "セッションが見つかりません。", 404
    return render_template("processing.html", session=session)


@app.route("/session/<int:session_id>/status")
def session_status(session_id):
    session = db.get_session(session_id)
    if not session:
        return jsonify({"error": "not_found"}), 404
    with PROGRESS_LOCK:
        progress = PROGRESS.get(session_id)
    return jsonify({
        "status": session["status"],
        "spike_count": session["spike_count"],
        "error_message": session["error_message"],
        "progress": progress,
    })


@app.route("/session/<int:session_id>/player")
def player_page(session_id):
    session = db.get_session(session_id)
    if not session or session["status"] != "done":
        return redirect(url_for("session_status_page", session_id=session_id))
    players = db.get_session_players(session_id)
    spikes = db.get_spike_videos(session_id)
    return render_template("player.html", session=session, players=players, spikes=spikes)


@app.route("/media/<int:session_id>/<path:filename>")
def media(session_id, filename):
    directory = os.path.join(OUTPUT_ROOT, str(session_id))
    return send_from_directory(directory, filename)


if __name__ == "__main__":
    db.init_db()
    os.makedirs(UPLOAD_ROOT, exist_ok=True)
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    app.run(debug=True, host="0.0.0.0", port=5001)
