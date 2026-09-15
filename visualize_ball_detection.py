"""Debug/verification tool: renders a video with ONLY the ball bbox drawn, using
the exact same detection functions and "active vs resting ball" logic as
extract_spikes() in spike_clips.py. No person detection is involved -- this
exists to visually confirm that the web app's analysis pipeline only ever
looks at the ball.
"""
import os

import cv2

from spike_clips import (
    MODULE_DIR, BALL_MODEL_FILE, CONF_THRESHOLD, IOU_THRESHOLD, DEVICE,
    TRACKER_CFG, IMG_SIZE, VIDEO_CODEC,
    BALL_STATIC_WINDOW, BALL_STATIC_MAX_DISPLACEMENT,
    load_model, parse_result, suppress_duplicates, center, is_static,
)

VIDEO_FILENAME = "video01.mp4"
OUTPUT_VIDEO = os.path.join("output", "ball_only_output.mp4")
# Set to None to process the entire video; kept short here for a quick check.
MAX_FRAMES = 1000

COLOR_ACTIVE_BALL = (0, 0, 255)     # red -- the ball the spike algorithm reacts to
COLOR_RESTING_BALL = (128, 128, 128)  # gray -- spare/idle balls, ignored by the algorithm
# The raw model bbox is often cropped a bit tight around the ball. Pad it outward
# (by this fraction of the box's own width/height) purely for display, so the drawn
# box visibly encloses the whole ball. This does not affect detection/tracking --
# only the box drawn on this debug video.
BBOX_DISPLAY_PADDING_RATIO = 0.5
BBOX_LINE_THICKNESS = 30  # 15x the original thickness of 2


def pad_box(x1, y1, x2, y2, frame_w, frame_h, ratio):
    pad_x = int((x2 - x1) * ratio)
    pad_y = int((y2 - y1) * ratio)
    return (
        max(0, x1 - pad_x),
        max(0, y1 - pad_y),
        min(frame_w, x2 + pad_x),
        min(frame_h, y2 + pad_y),
    )


def draw(frame, dets, active_det):
    frame_h, frame_w = frame.shape[:2]
    for det in dets:
        x1, y1, x2, y2 = pad_box(*det["xyxy"], frame_w, frame_h, BBOX_DISPLAY_PADDING_RATIO)
        is_active = det is active_det
        color = COLOR_ACTIVE_BALL if is_active else COLOR_RESTING_BALL
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, BBOX_LINE_THICKNESS)
        cv2.putText(frame, "Ball", (x1, max(25, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
    return frame


def main():
    video_path = os.path.join(MODULE_DIR, VIDEO_FILENAME)
    ball_model_path = os.path.join(MODULE_DIR, BALL_MODEL_FILE)
    os.makedirs("output", exist_ok=True)

    ball_model = load_model(ball_model_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*VIDEO_CODEC)
    writer = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))

    ball_history = {}
    frame_idx = 0
    print(f"[Ball-only] Detecting ball and drawing bbox -> {OUTPUT_VIDEO}")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if MAX_FRAMES and frame_idx >= MAX_FRAMES:
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

        # Same "which ball is actually in play" filter used by extract_spikes().
        active_candidates = [
            det for det in dets
            if det.get("track_id") is None
            or not is_static(ball_history[det["track_id"]], BALL_STATIC_WINDOW, BALL_STATIC_MAX_DISPLACEMENT)
        ]
        active_det = max(active_candidates, key=lambda d: d["confidence"]) if active_candidates else None

        writer.write(draw(frame, dets, active_det))

        if frame_idx % 200 == 0:
            print(f"[Ball-only] frame {frame_idx}")
        frame_idx += 1

    cap.release()
    writer.release()
    print(f"[Ball-only] Saved {OUTPUT_VIDEO}")


if __name__ == "__main__":
    main()
