import os
import sys

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

import math

VIDEO_FILENAME = "video01.mp4"
PERSON_MODEL_FILE = "best_person.pt"
BALL_MODEL_FILE = "best_ball.pt"
OUTPUT_DIR = "output"
OUTPUT_VIDEO = os.path.join(OUTPUT_DIR, "ball_person_output.mp4")
# Number of frames to process, starting from the beginning of the video.
# Set to None to process the entire video.
MAX_FRAMES = 1000
# Only frames [0, RED_SELECTION_FRAMES) are used to decide which person ID moved the
# least. Once chosen, that ID stays red for every frame it appears in, even beyond
# this window.
RED_SELECTION_FRAMES = 100
MIN_TRACK_FRAMES = 30  # minimum frames (within the selection window) a person ID must appear in to be eligible
# The "least movement" heuristic assumes every eligible ID is a real person, but the
# person model occasionally locks onto a static background pattern (e.g. a court
# floor line) as a low-confidence, barely-moving false positive -- which then wins
# the "least movement" comparison trivially. Require a minimum average detection
# confidence over the selection window to filter those out before ranking by
# movement. Real players in this setup average ~0.5-0.6; the observed false-positive
# floor-line detection averaged ~0.36, so 0.45 sits between the two with margin.
MIN_AVG_CONFIDENCE = 0.45
# If the red ID briefly fails to be detected (e.g. a borderline-confidence dip),
# keep drawing its last known box for up to this many consecutive frames instead
# of letting it disappear.
RED_HOLD_MAX_FRAMES = 60
# If the red track ID is lost and the tracker later assigns a new ID to what is
# almost certainly the same person (e.g. after a brief occlusion), re-link to that
# new ID as long as it appears within this many pixels of the last known position.
RED_SWITCH_MAX_DISTANCE = 200
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.5
DEVICE = "mps"
VIDEO_CODEC = "mp4v"
TRACKER_CFG = "botsort.yaml"
IMG_SIZE = 640

COLOR_BALL = (255, 0, 0)           # blue (BGR)
COLOR_LEAST_MOVEMENT = (0, 0, 255)  # red (BGR)
COLOR_PERSON = (0, 255, 0)         # green (BGR)


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def load_model(weight_path):
    weight_path = os.path.abspath(weight_path)
    if not os.path.exists(weight_path):
        raise FileNotFoundError(f"Model file not found: {weight_path}")
    return YOLO(weight_path)


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


def center(box):
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def compute_least_movement_id(person_positions, min_frames, person_confidences=None, min_avg_confidence=0.0):
    scores = {}
    for tid, points in person_positions.items():
        if len(points) < min_frames:
            continue
        if person_confidences is not None:
            confs = person_confidences.get(tid, [])
            if not confs or (sum(confs) / len(confs)) < min_avg_confidence:
                continue
        total = 0.0
        for i in range(1, len(points)):
            x1, y1 = points[i - 1]
            x2, y2 = points[i]
            total += math.hypot(x2 - x1, y2 - y1)
        scores[tid] = total
    if not scores:
        return None, scores
    least_id = min(scores, key=scores.get)
    return least_id, scores


def resolve_red_track(person_dets_per_frame, initial_id, max_hold_frames, max_switch_distance):
    """Track the single "red" person across all frames, tolerating two failure
    modes of the detector/tracker:
      1. Brief missed detections -> hold the last known box for a few frames.
      2. The tracker assigning a new ID to the same person after an occlusion ->
         re-link to whichever new detection appears closest to the last known
         position, within max_switch_distance pixels.
    Returns one entry per frame: the matching detection dict (xyxy + track_id) to
    draw red, a synthetic held box, or None if the person cannot be located.
    """
    if initial_id is None:
        return [None] * len(person_dets_per_frame)

    current_id = initial_id
    last_box = None
    gap = 0
    red_matches = []
    for dets in person_dets_per_frame:
        match = next((d for d in dets if d.get("track_id") == current_id), None)
        if match is None and last_box is not None:
            lx, ly = center(last_box)
            candidates = []
            for d in dets:
                if d.get("track_id") is None:
                    continue  # only adopt properly tracked IDs, so current_id stays usable later
                cx, cy = center(d["xyxy"])
                dist = math.hypot(cx - lx, cy - ly)
                if dist <= max_switch_distance:
                    candidates.append((dist, d))
            if candidates:
                candidates.sort(key=lambda t: t[0])
                match = candidates[0][1]
                current_id = match.get("track_id")

        if match is not None:
            last_box = match["xyxy"]
            gap = 0
            red_matches.append(match)
        else:
            gap += 1
            if last_box is not None and gap <= max_hold_frames:
                red_matches.append({"xyxy": last_box, "track_id": current_id})
            else:
                red_matches.append(None)
    return red_matches


def draw_detections(frame, ball_dets, person_dets, red_match):
    output = frame
    for det in ball_dets:
        x1, y1, x2, y2 = det["xyxy"]
        cv2.rectangle(output, (x1, y1), (x2, y2), COLOR_BALL, 2)
        cv2.putText(output, "Ball", (x1, max(25, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_BALL, 2, cv2.LINE_AA)
    for det in person_dets:
        if det is red_match:
            continue  # drawn separately below, in red
        x1, y1, x2, y2 = det["xyxy"]
        track_id = det.get("track_id")
        label = f"ID:{track_id}" if track_id is not None else "person"
        cv2.rectangle(output, (x1, y1), (x2, y2), COLOR_PERSON, 2)
        cv2.putText(output, label, (x1, max(25, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_PERSON, 2, cv2.LINE_AA)
    if red_match is not None:
        x1, y1, x2, y2 = red_match["xyxy"]
        track_id = red_match.get("track_id")
        label = f"ID:{track_id}" if track_id is not None else "person"
        cv2.rectangle(output, (x1, y1), (x2, y2), COLOR_LEAST_MOVEMENT, 2)
        cv2.putText(output, label, (x1, max(25, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_LEAST_MOVEMENT, 2, cv2.LINE_AA)
    return output


def main():
    cwd = os.getcwd()
    video_path = os.path.join(cwd, VIDEO_FILENAME)
    person_model_path = os.path.join(cwd, PERSON_MODEL_FILE)
    ball_model_path = os.path.join(cwd, BALL_MODEL_FILE)
    ensure_dir(OUTPUT_DIR)

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    person_model = load_model(person_model_path)
    ball_model = load_model(ball_model_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    person_dets_per_frame = []
    ball_dets_per_frame = []
    person_positions = {}
    person_confidences = {}

    frame_idx = 0
    print(f"[Pass 1/2] Detecting person & ball on all frames (total={total_frames})...")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if MAX_FRAMES and frame_idx >= MAX_FRAMES:
            break

        person_result = person_model.track(
            frame, persist=True, tracker=TRACKER_CFG, conf=CONF_THRESHOLD,
            imgsz=IMG_SIZE, device=DEVICE, verbose=False,
        )[0]
        ball_result = ball_model.track(
            frame, persist=True, tracker=TRACKER_CFG, conf=CONF_THRESHOLD,
            imgsz=IMG_SIZE, device=DEVICE, verbose=False,
        )[0]

        person_dets = suppress_duplicates(parse_result(person_result), IOU_THRESHOLD)
        ball_dets = suppress_duplicates(parse_result(ball_result), IOU_THRESHOLD)

        person_dets_per_frame.append(person_dets)
        ball_dets_per_frame.append(ball_dets)

        if frame_idx < RED_SELECTION_FRAMES:
            for det in person_dets:
                tid = det.get("track_id")
                if tid is None:
                    continue
                person_positions.setdefault(tid, []).append(center(det["xyxy"]))
                person_confidences.setdefault(tid, []).append(det["confidence"])

        if frame_idx % 200 == 0:
            denom = total_frames or (frame_idx + 1)
            progress = min(100.0, (frame_idx / denom) * 100.0)
            print(f"[Pass 1/2] Frame {frame_idx}/{denom} ({progress:.1f}%)")

        frame_idx += 1

    cap.release()
    num_frames = len(person_dets_per_frame)
    print(f"[Pass 1/2] Done. Processed {num_frames} frames.")

    selection_frames = min(RED_SELECTION_FRAMES, num_frames)
    min_track_frames = min(MIN_TRACK_FRAMES, selection_frames)
    print(f"[Analysis] Computing per-ID movement from first {selection_frames} frames (min_frames={min_track_frames})...")
    least_movement_id, scores = compute_least_movement_id(
        person_positions, min_track_frames, person_confidences, MIN_AVG_CONFIDENCE
    )
    ranked = sorted(scores.items(), key=lambda kv: kv[1])
    print(f"[Analysis] Movement scores (lowest first, top 10): {ranked[:10]}")
    print(f"[Analysis] Least-movement person ID: {least_movement_id}")

    red_matches = resolve_red_track(
        person_dets_per_frame, least_movement_id, RED_HOLD_MAX_FRAMES, RED_SWITCH_MAX_DISTANCE
    )
    id_switches = sorted({m.get("track_id") for m in red_matches if m is not None} - {least_movement_id})
    if id_switches:
        print(f"[Analysis] Red identity also matched to re-assigned track IDs: {id_switches}")

    print(f"[Pass 2/2] Rendering {num_frames} frames to {OUTPUT_VIDEO}...")
    cap2 = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*VIDEO_CODEC)
    writer = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))

    idx = 0
    while idx < num_frames:
        ok, frame = cap2.read()
        if not ok:
            break
        out = draw_detections(frame, ball_dets_per_frame[idx], person_dets_per_frame[idx], red_matches[idx])
        writer.write(out)
        if idx % 200 == 0:
            progress = (idx / num_frames) * 100.0
            print(f"[Pass 2/2] Frame {idx}/{num_frames} ({progress:.1f}%)")
        idx += 1

    cap2.release()
    writer.release()
    print(f"[Complete] Saved output video to {OUTPUT_VIDEO}")
    print(f"[Summary] Least-movement person ID (red): {least_movement_id}")


if __name__ == "__main__":
    main()
