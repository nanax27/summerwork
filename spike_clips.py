import os
import math
import subprocess

try:
    import cv2
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "OpenCV is required. Install dependencies with: python3 -m pip install -r requirements.txt"
    ) from exc

try:
    from ultralytics import YOLO
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "Ultralytics YOLO is required. Install dependencies with: python3 -m pip install -r requirements.txt"
    ) from exc

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEO_FILENAME = "video01.mp4"
BALL_MODEL_FILE = "best_ball.pt"
OUTPUT_DIR = os.path.join("output", "spikes")
# Number of frames to process, starting from the beginning of the video.
# Set to None to process the entire video.
MAX_FRAMES = 4000
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.5
DEVICE = "mps"
VIDEO_CODEC = "mp4v"
TRACKER_CFG = "botsort.yaml"
IMG_SIZE = 640
# Reference ("near the net") line, as a fraction of frame height, measured from the
# top. 1/3 splits the screen into thirds and puts the line roughly at net height. Only
# upward crossings (ball moving from below the line to above it) are detected;
# downward crossings are ignored entirely.
LINE_POSITION_RATIO = 1 / 3
# Each spike clip runs for this many seconds starting at the upward crossing. Any
# further upward crossing that happens before this duration has elapsed is ignored;
# the next clip only starts once the current one's duration has fully elapsed.
SPIKE_CLIP_DURATION_SECONDS = 4.0
# A trailing clip truncated by the end of the processed frame range can end up much
# shorter than SPIKE_CLIP_DURATION_SECONDS. Discard it if shorter than this.
MIN_SPIKE_DURATION_SECONDS = 0.3
# Two known ball positions more than this many seconds apart (e.g. the ball landing
# to end one rally, then a completely separate toss starting the next one) are almost
# certainly not the same continuous flight. Don't bridge/interpolate a crossing across
# such a gap -- treat the point after it as a fresh start instead.
MAX_DETECTION_GAP_SECONDS = 1.0
# Idle/spare balls sitting on the court are also detected every frame. A ball whose
# position has moved less than BALL_STATIC_MAX_DISPLACEMENT pixels over its last
# BALL_STATIC_WINDOW frames is treated as "resting" and ignored, so the line-crossing
# logic only reacts to the ball actually in play.
BALL_STATIC_WINDOW = 15
BALL_STATIC_MAX_DISPLACEMENT = 20


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def clear_old_spike_clips(output_dir):
    """Remove spike_*.mp4 files left over from a previous run/settings, so stale
    clips never linger alongside the current run's output."""
    if not os.path.isdir(output_dir):
        return
    for name in os.listdir(output_dir):
        if name.startswith("spike_") and name.endswith(".mp4"):
            os.remove(os.path.join(output_dir, name))


def load_model(weight_path):
    weight_path = os.path.abspath(weight_path)
    if not os.path.exists(weight_path):
        raise FileNotFoundError(f"Model file not found: {weight_path}")
    return YOLO(weight_path)


def center(box):
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def xyxy_iou(box_a, box_b):
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    inter = w * h
    if inter == 0.0:
        return 0.0
    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0.0 else 0.0


def suppress_duplicates(detections, iou_threshold):
    if not detections:
        return []
    detections = sorted(detections, key=lambda d: d.get("confidence", 0.0), reverse=True)
    keep = []
    for det in detections:
        keep_det = True
        for existing in keep:
            if det.get("track_id") is not None and existing.get("track_id") == det.get("track_id"):
                keep_det = False
                break
            if xyxy_iou(det["xyxy"], existing["xyxy"]) > iou_threshold:
                keep_det = False
                break
        if keep_det:
            keep.append(det)
    return keep


def parse_result(result):
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return []
    xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes, "xyxy") else None
    confs = boxes.conf.cpu().numpy() if hasattr(boxes, "conf") else None
    track_ids = boxes.id.cpu().numpy() if hasattr(boxes, "id") and boxes.id is not None else None
    detections = []
    if xyxy is None:
        return detections
    for i, box in enumerate(xyxy):
        coords = box.tolist() if hasattr(box, "tolist") else list(box)
        det = {
            "xyxy": [int(c) for c in coords],
            "confidence": float(confs[i]) if confs is not None and len(confs) > i else 0.0,
            "track_id": int(track_ids[i]) if track_ids is not None and len(track_ids) > i else None,
        }
        detections.append(det)
    return detections


def is_static(history, window, max_displacement):
    """A ball is "resting" once its position barely changes over its last `window`
    frames. New tracks (< window frames of history) are assumed active/in-play."""
    if len(history) < window:
        return False
    recent = history[-window:]
    xs = [p[0] for p in recent]
    ys = [p[1] for p in recent]
    displacement = math.hypot(max(xs) - min(xs), max(ys) - min(ys))
    return displacement <= max_displacement


def side_of_line(y, line_y):
    return "below" if y > line_y else "above"


def interpolate_crossing_frame(prev_frame, prev_y, frame_idx, y, line_y):
    """Detection gaps (e.g. the tracker losing the ball to motion blur mid-flight
    and picking it up again several frames later, under a new track ID) mean the
    two straddling points can be far apart in time. Rather than snapping to
    frame_idx (which may already be deep past the line), linearly interpolate
    between the two known points to estimate which frame the ball actually
    crossed line_y on.
    """
    if frame_idx == prev_frame or y == prev_y:
        return frame_idx
    t = (line_y - prev_y) / (y - prev_y)
    t = max(0.0, min(1.0, t))
    return int(round(prev_frame + t * (frame_idx - prev_frame)))


def detect_upward_crossing_frames(ball_positions, line_y, max_gap_frames):
    """ball_positions: ordered [(frame_idx, y_center), ...] for frames where the
    ball was detected. Returns the (interpolated) frame of every crossing where the
    ball moves from below the line to above it. Downward crossings are not
    detected/tracked at all -- they never mark anything.
    A gap of more than max_gap_frames between two consecutive known positions breaks
    continuity (treated as two unrelated events, e.g. one rally ending and a new one
    starting later) -- no crossing is interpolated across it.
    """
    crossings = []
    prev_side = None
    prev_frame_idx = None
    prev_y = None
    for frame_idx, y in ball_positions:
        side = side_of_line(y, line_y)
        if prev_frame_idx is not None and (frame_idx - prev_frame_idx) > max_gap_frames:
            prev_side = None
        if prev_side == "below" and side == "above":
            crossings.append(interpolate_crossing_frame(prev_frame_idx, prev_y, frame_idx, y, line_y))
        prev_side = side
        prev_frame_idx = frame_idx
        prev_y = y
    return crossings


def build_spike_ranges(crossing_frames, clip_duration_frames, last_frame):
    """Each spike starts at an upward crossing and runs for clip_duration_frames
    (clipped to the end of the processed range). Crossings that happen while a
    clip is still running are ignored -- the next clip only starts once the
    current one's duration has fully elapsed.
    Returns [(start_frame, end_frame), ...].
    """
    spikes = []
    next_allowed_frame = -1
    for start in crossing_frames:
        if start < next_allowed_frame:
            continue
        end = min(start + clip_duration_frames - 1, last_frame)
        if end > start:
            spikes.append((start, end))
        next_allowed_frame = start + clip_duration_frames
    return spikes


def transcode_for_web(raw_video_path, source_video_path, start_time, duration, output_path):
    """Re-encode a cv2-written clip (mp4v, silent) to H.264 so browsers can play it
    via <video>, muxing in the matching audio segment from the original source
    video. Falls back to keeping the raw silent mp4v file if ffmpeg is unavailable
    or fails, so a clip is never lost outright.
    """
    cmd = [
        "ffmpeg", "-y",
        "-i", raw_video_path,
        "-ss", f"{max(0.0, start_time):.3f}", "-t", f"{duration:.3f}", "-i", source_video_path,
        "-map", "0:v:0", "-map", "1:a:0?",
        "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-shortest",
        "-movflags", "+faststart",
        output_path,
    ]
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except FileNotFoundError:
        result = None
    if result is not None and result.returncode == 0 and os.path.exists(output_path):
        os.remove(raw_video_path)
        return True
    print(f"[Warning] ffmpeg transcode failed for {output_path}; keeping raw clip instead.")
    if os.path.exists(output_path):
        os.remove(output_path)
    os.rename(raw_video_path, output_path)
    return False


def extract_spikes(
    video_path,
    output_dir=OUTPUT_DIR,
    ball_model_path=None,
    max_frames=MAX_FRAMES,
    progress_callback=None,
):
    """Run the ball-detection / line-crossing / 4s-clip pipeline on video_path and
    save the resulting spike clips into output_dir. Returns a list of dicts, one
    per spike, in detection order:
      {"spike_number", "start_frame", "end_frame", "start_time", "duration", "video_path"}

    progress_callback(stage, current, total), if given, is called periodically
    during detection ("detect") and extraction ("extract") so a caller (e.g. a
    web backend) can surface progress without depending on stdout.
    """
    if ball_model_path is None:
        ball_model_path = os.path.join(MODULE_DIR, BALL_MODEL_FILE)
    ensure_dir(output_dir)
    clear_old_spike_clips(output_dir)

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    ball_model = load_model(ball_model_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    line_y = int(height * LINE_POSITION_RATIO)

    ball_positions = []
    ball_history = {}
    frame_idx = 0
    print(f"[Pass 1/2] Detecting ball on all frames (total={total_frames})...")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if max_frames and frame_idx >= max_frames:
            break

        result = ball_model.track(
            frame, persist=True, tracker=TRACKER_CFG, conf=CONF_THRESHOLD,
            imgsz=IMG_SIZE, device=DEVICE, verbose=False,
        )[0]
        dets = suppress_duplicates(parse_result(result), IOU_THRESHOLD)

        for det in dets:
            tid = det.get("track_id")
            if tid is None:
                continue
            ball_history.setdefault(tid, []).append(center(det["xyxy"]))

        # A detection with no track_id (the tracker failed to confirm/assign one
        # this frame, which happens often for a fast-moving ball) has no history to
        # judge as "resting", so treat it as active rather than silently dropping
        # it -- a stationary ball almost always keeps a stable, confirmed ID.
        active_candidates = [
            det for det in dets
            if det.get("track_id") is None
            or not is_static(ball_history[det["track_id"]], BALL_STATIC_WINDOW, BALL_STATIC_MAX_DISPLACEMENT)
        ]
        if active_candidates:
            best = max(active_candidates, key=lambda d: d["confidence"])
            ball_positions.append((frame_idx, center(best["xyxy"])[1]))

        if frame_idx % 200 == 0:
            denom = total_frames or (frame_idx + 1)
            progress = min(100.0, (frame_idx / denom) * 100.0)
            print(f"[Pass 1/2] Frame {frame_idx}/{denom} ({progress:.1f}%)")
            if progress_callback:
                progress_callback("detect", frame_idx, denom)

        frame_idx += 1

    cap.release()
    num_frames = frame_idx
    print(f"[Pass 1/2] Done. Processed {num_frames} frames. Ball detected in {len(ball_positions)} frames.")
    print(f"[Analysis] Reference line at y={line_y} (frame height={height}).")

    max_gap_frames = round(MAX_DETECTION_GAP_SECONDS * fps)
    crossing_frames = detect_upward_crossing_frames(ball_positions, line_y, max_gap_frames)
    clip_duration_frames = round(SPIKE_CLIP_DURATION_SECONDS * fps)
    spikes = build_spike_ranges(crossing_frames, clip_duration_frames, num_frames - 1)
    min_spike_frames = MIN_SPIKE_DURATION_SECONDS * fps
    kept, dropped = [], []
    for s, e in spikes:
        (kept if (e - s) >= min_spike_frames else dropped).append((s, e))
    spikes = kept
    if dropped:
        print(f"[Analysis] Discarded {len(dropped)} spike(s) shorter than {MIN_SPIKE_DURATION_SECONDS}s: {dropped}")
    print(f"[Analysis] Detected {len(spikes)} spike(s):")
    for i, (s, e) in enumerate(spikes, start=1):
        print(f"  Spike {i}: frames {s}-{e} ({(e - s) / fps:.2f}s), start_time={s / fps:.2f}s")

    if not spikes:
        print("[Complete] No spikes detected; nothing to save.")
        return []

    print("[Pass 2/2] Extracting spike clips...")
    cap2 = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*VIDEO_CODEC)

    # Map each frame to the spike it belongs to, so touching/adjacent ranges
    # (a spike's end frame equal to the next spike's start frame) can't leave the
    # writer stuck waiting for a frame index it will never see again.
    frame_to_spike = {}
    for i, (s, e) in enumerate(spikes):
        for f in range(s, e + 1):
            frame_to_spike[f] = i

    raw_paths = {}
    writer = None
    current_spike = None
    idx = 0
    while idx < num_frames:
        ok, frame = cap2.read()
        if not ok:
            break
        spike_i = frame_to_spike.get(idx)
        if spike_i != current_spike:
            if writer is not None:
                writer.release()
                writer = None
            if spike_i is not None:
                s, e = spikes[spike_i]
                raw_path = os.path.join(output_dir, f"spike_{spike_i + 1:03d}.raw.mp4")
                raw_paths[spike_i] = raw_path
                writer = cv2.VideoWriter(raw_path, fourcc, fps, (width, height))
                print(f"[Pass 2/2] Writing {raw_path} (frames {s}-{e})")
            current_spike = spike_i
        if writer is not None:
            writer.write(frame)
        if idx % 200 == 0 and progress_callback:
            progress_callback("extract", idx, num_frames)
        idx += 1

    if writer is not None:
        writer.release()

    cap2.release()

    print("[Pass 2/2] Transcoding clips for web playback (H.264 + audio)...")
    results = []
    for i, (s, e) in enumerate(spikes):
        out_path = os.path.join(output_dir, f"spike_{i + 1:03d}.mp4")
        start_time = s / fps
        duration = (e - s + 1) / fps
        transcode_for_web(raw_paths[i], video_path, start_time, duration, out_path)
        results.append({
            "spike_number": i + 1,
            "start_frame": s,
            "end_frame": e,
            "start_time": start_time,
            "duration": duration,
            "video_path": out_path,
        })

    print(f"[Complete] Saved {len(results)} spike clip(s) to {output_dir}")
    return results


def main():
    cwd = os.getcwd()
    video_path = os.path.join(cwd, VIDEO_FILENAME)
    ball_model_path = os.path.join(cwd, BALL_MODEL_FILE)
    extract_spikes(video_path, OUTPUT_DIR, ball_model_path, MAX_FRAMES)


if __name__ == "__main__":
    main()
