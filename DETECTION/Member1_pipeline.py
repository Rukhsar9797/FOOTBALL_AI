import os
import sys
import cv2
import json
import math
import time
import argparse
import numpy as np
from collections import defaultdict, Counter
from datetime import datetime
# Handle optional YOLO import to support clean mock execution on CPU environments
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
# =====================================================================
# Configuration & Constants
# =====================================================================
# Default field bounding polygon (normalized coordinates, relative to frame width/height)
# Coordinates represent a standard perspective-warped football pitch (TL, TR, BR, BL)
DEFAULT_PITCH_POLYGON_PCT = [
    [0.15, 0.28],  # Top-Left (TL)
    [0.85, 0.28],  # Top-Right (TR)
    [0.94, 0.94],  # Bottom-Right (BR)
    [0.06, 0.94]   # Bottom-Left (BL)
]
# HSV Color Mask Parameters
# Red Team jersey colors (wraps around 0 and 180 degrees in Hue)
RED_LOWER1 = np.array([0, 70, 50])
RED_UPPER1 = np.array([10, 255, 255])
RED_LOWER2 = np.array([170, 70, 50])
RED_UPPER2 = np.array([180, 255, 255])
# White Team jersey colors (low saturation, high value -> "washed out" pixels)
WHITE_LOWER = np.array([0, 0, 150])
WHITE_UPPER = np.array([180, 60, 255])
# Referee jersey colors (dark/black kit only -> low value)
REF_DARK_LOWER = np.array([0, 0, 0])
REF_DARK_UPPER = np.array([180, 255, 55])
# Grass/pitch background color, used to explicitly exclude background bleed
# from torso crops so a few green edge-pixels can't tip a jersey vote.
# NOTE: the old "neon referee" range (H 25-80) overlapped this grass hue,
# which is what was causing almost every track to be misclassified as referee.
GRASS_LOWER = np.array([35, 40, 40])
GRASS_UPPER = np.array([85, 255, 255])
# Minimum dominant-color share required within a torso crop before we trust
# the classification; below this we treat the read as noise, not a jersey.
JERSEY_DOMINANCE_THRESHOLD = 0.12
# Team naming and styling
TEAM_RED = "Red Team"
TEAM_WHITE = "White Team"
ROLE_PLAYER = "player"
ROLE_REFEREE = "referee"
ROLE_BALL = "ball"
# Hex Colors for JSON output
HEX_RED = "#FF0000"
HEX_WHITE = "#FFFFFF"
# =====================================================================
# Homography Mapping (Physical Telemetry Calculation)
# =====================================================================
class HomographyMapper:
    """
    Translates screen pixel coordinates to a normalized 0-100 pitch coordinate grid
    using 2D perspective Homography transformation.
    """
    def __init__(self, pitch_poly):
        # src_pts: pixels of the pitch boundary polygon
        self.src_pts = np.float32(pitch_poly)
        # dst_pts: normalized coordinate system representing the pitch grid [0-100]
        self.dst_pts = np.float32([
            [0, 0],       # TL
            [100, 0],     # TR
            [100, 100],   # BR
            [0, 100]      # BL
        ])
        self.M = cv2.getPerspectiveTransform(self.src_pts, self.dst_pts)
    def to_pitch_coords(self, px, py):
        """Maps a pixel coordinate to [0, 100] pitch coordinates."""
        pts = np.array([[[px, py]]], dtype=np.float32)
        warped = cv2.perspectiveTransform(pts, self.M)
        x_pitch = float(np.clip(warped[0][0][0], 0.0, 100.0))
        y_pitch = float(np.clip(warped[0][0][1], 0.0, 100.0))
        return x_pitch, y_pitch
# =====================================================================
# Speed Telemetry Calculator
# =====================================================================
class SpeedCalculator:
    """
    Computes smoothed physical speed telemetry (km/h) for players
    based on normalized pitch coordinate displacements over time.
    """
    def __init__(self, fps):
        self.fps = fps
        self.prev_pitch_coords = {}  # track_id -> (x_pitch, y_pitch)
        self.speed_history = defaultdict(list)  # track_id -> list of speed samples
    def calculate_speed(self, track_id, x_pitch, y_pitch):
        dt = 1.0 / self.fps if self.fps > 0 else 0.033
        if track_id not in self.prev_pitch_coords:
            self.prev_pitch_coords[track_id] = (x_pitch, y_pitch)
            return 0.0
        x_prev, y_prev = self.prev_pitch_coords[track_id]
        dx = x_pitch - x_prev
        dy = y_pitch - y_prev
        # Assuming standard pitch dimensions of 105m length and 68m width
        # Map 0-100 coordinate changes to meters
        dx_meters = dx * 1.05
        dy_meters = dy * 0.68
        dist_meters = math.hypot(dx_meters, dy_meters)
        raw_speed_mps = dist_meters / dt
        raw_speed_kmh = raw_speed_mps * 3.6
        # Update coordinate state
        self.prev_pitch_coords[track_id] = (x_pitch, y_pitch)
        # Apply a 5-frame moving average filter to smooth speed spikes
        self.speed_history[track_id].append(raw_speed_kmh)
        if len(self.speed_history[track_id]) > 5:
            self.speed_history[track_id].pop(0)
        smoothed_speed = float(np.mean(self.speed_history[track_id]))
        return round(smoothed_speed, 2)
# =====================================================================
# HSV Color Classifier
# =====================================================================
def classify_jersey_hsv(crop):
    """
    Analyzes jersey color distribution within a cropped bounding box using
    HSV color-masking. Resolves Red Team vs. White Team vs. Referee (black kit).

    The three candidate masks (red / white / black-referee) are scored by
    their share of the crop, and the highest-scoring class wins - rather than
    the old nested if/else which let a "referee" ratio win on a technicality.
    A grass-bleed check guards against edge pixels from the green pitch
    background (which previously overlapped the referee "neon" hue range and
    caused almost everything to be misclassified as referee).
    """
    if crop is None or crop.size == 0:
        return TEAM_WHITE
    total_pixels = crop.shape[0] * crop.shape[1]
    if total_pixels == 0:
        return TEAM_WHITE
    # Convert the jersey crop from BGR to HSV
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    # Red Team mask (handling wrap-around range)
    mask_red1 = cv2.inRange(hsv, RED_LOWER1, RED_UPPER1)
    mask_red2 = cv2.inRange(hsv, RED_LOWER2, RED_UPPER2)
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)
    red_ratio = cv2.countNonZero(mask_red) / total_pixels
    # White Team mask (low saturation, high value)
    mask_white = cv2.inRange(hsv, WHITE_LOWER, WHITE_UPPER)
    white_ratio = cv2.countNonZero(mask_white) / total_pixels
    # Referee mask (dark/black kit only - low value)
    mask_black = cv2.inRange(hsv, REF_DARK_LOWER, REF_DARK_UPPER)
    black_ratio = cv2.countNonZero(mask_black) / total_pixels
    # Grass background bleed (green pitch behind/around the torso box)
    mask_grass = cv2.inRange(hsv, GRASS_LOWER, GRASS_UPPER)
    grass_ratio = cv2.countNonZero(mask_grass) / total_pixels

    scores = {
        TEAM_RED: red_ratio,
        TEAM_WHITE: white_ratio,
        ROLE_REFEREE: black_ratio,
    }
    best_cls = max(scores, key=scores.get)
    best_ratio = scores[best_cls]

    # If nothing clearly dominates, or the crop is mostly grass bleed, this
    # single-frame read is unreliable noise - let the majority vote across
    # frames (in process_video) correct it, rather than defaulting to referee.
    if best_ratio < JERSEY_DOMINANCE_THRESHOLD or grass_ratio > 0.5:
        return TEAM_WHITE
    return best_cls
# =====================================================================
# Ball Trajectory Processor
# =====================================================================
class BallTrajectoryProcessor:
    """
    Performs frame-gap interpolation and rolling-window coordinate smoothing
    on the ball trajectory to resolve occlusions and erratic tracking.
    """
    def __init__(self, max_gap_frames=3, smoothing_window=5):
        self.max_gap_frames = max_gap_frames
        self.smoothing_window = smoothing_window
        self.raw_detections = {}  # frame_idx -> (x, y)
    def add_detection(self, frame_idx, cx, cy):
        self.raw_detections[frame_idx] = (cx, cy)
    def process_trajectory(self):
        """
        Runs gap interpolation (1-3 frames) followed by a moving average smoothing
        window across the entire trajectory set.
        """
        interpolated = {}
        frames = sorted(self.raw_detections.keys())
        if not frames:
            return interpolated
        # Step 1: Linear interpolation over brief gaps (1 to max_gap_frames)
        for i in range(len(frames) - 1):
            f_start = frames[i]
            f_end = frames[i + 1]
            gap = f_end - f_start
            x_start, y_start = self.raw_detections[f_start]
            x_end, y_end = self.raw_detections[f_end]
            interpolated[f_start] = (x_start, y_start)
            if 1 < gap <= self.max_gap_frames + 1:
                for f_mid in range(f_start + 1, f_end):
                    t = (f_mid - f_start) / gap
                    x_mid = x_start + t * (x_end - x_start)
                    y_mid = y_start + t * (y_end - y_start)
                    interpolated[f_mid] = (int(round(x_mid)), int(round(y_mid)))

        # Add the last detected frame
        interpolated[frames[-1]] = self.raw_detections[frames[-1]]
        # Step 2: Smoothing using a rolling-window moving average
        smoothed = {}
        interp_frames = sorted(interpolated.keys())
        for idx, f in enumerate(interp_frames):
            window_coords = []
            for offset in range(-self.smoothing_window // 2 + 1, self.smoothing_window // 2 + 1):
                curr_idx = idx + offset
                if 0 <= curr_idx < len(interp_frames):
                    # Only include contiguous/near frames
                    if abs(interp_frames[curr_idx] - f) <= self.smoothing_window:
                        window_coords.append(interpolated[interp_frames[curr_idx]])
            if window_coords:
                avg_x = int(np.mean([pt[0] for pt in window_coords]))
                avg_y = int(np.mean([pt[1] for pt in window_coords]))
                smoothed[f] = (avg_x, avg_y)
            else:
                smoothed[f] = interpolated[f]
        return smoothed
# =====================================================================
# Synthetic Match Video Generator (Mock Mode Support)
# =====================================================================
def generate_mock_video(filepath, width=1280, height=720, total_frames=150):
    """
    Generates a 2D synthetic football match video and returns ground-truth
    tracking detections to feed into the pipeline for validation.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(filepath, fourcc, 30.0, (width, height))
    # Field boundaries mapping
    pitch_poly = np.array([
        [int(p[0] * width), int(p[1] * height)] for p in DEFAULT_PITCH_POLYGON_PCT
    ], dtype=np.int32)
    mock_tracks = []
    # Define paths for synthetic objects (players, referee, ball)
    for frame_idx in range(total_frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        # Draw green turf background
        frame[:] = (34, 139, 34)
        # Draw pitch white border line
        cv2.polylines(frame, [pitch_poly], isClosed=True, color=(255, 255, 255), thickness=3)
        frame_detections = []
        # Red Player 1 (track ID 1): moving diagonally across the pitch
        p1_x = int(300 + frame_idx * 2.5)
        p1_y = int(300 + frame_idx * 1.5)
        cv2.rectangle(frame, (p1_x - 15, p1_y - 45), (p1_x + 15, p1_y), (0, 0, 255), -1)  # Red jersey
        frame_detections.append({"track_id": 1, "class": 0, "bbox": [p1_x - 15, p1_y - 45, p1_x + 15, p1_y]})
        # Red Player 2 (track ID 2): moving horizontally
        p2_x = int(400 + frame_idx * 1.8)
        p2_y = int(250 + frame_idx * 0.2)
        cv2.rectangle(frame, (p2_x - 15, p2_y - 45), (p2_x + 15, p2_y), (0, 0, 255), -1)  # Red jersey
        frame_detections.append({"track_id": 2, "class": 0, "bbox": [p2_x - 15, p2_y - 45, p2_x + 15, p2_y]})
        # White Player 3 (track ID 3): moving leftwards
        p3_x = int(900 - frame_idx * 2.2)
        p3_y = int(400 + frame_idx * 0.5)
        cv2.rectangle(frame, (p3_x - 15, p3_y - 45), (p3_x + 15, p3_y), (255, 255, 255), -1)  # White jersey
        frame_detections.append({"track_id": 3, "class": 0, "bbox": [p3_x - 15, p3_y - 45, p3_x + 15, p3_y]})
        # White Player 4 (track ID 4): moving left-upwards
        p4_x = int(1000 - frame_idx * 2.0)
        p4_y = int(550 - frame_idx * 1.2)
        cv2.rectangle(frame, (p4_x - 15, p4_y - 45), (p4_x + 15, p4_y), (255, 255, 255), -1)  # White jersey
        frame_detections.append({"track_id": 4, "class": 0, "bbox": [p4_x - 15, p4_y - 45, p4_x + 15, p4_y]})
        # Referee (track ID 5): standing/moving slowly in center (black outfit)
        ref_x = int(640 + frame_idx * 0.5)
        ref_y = int(320 + frame_idx * 0.3)
        cv2.rectangle(frame, (ref_x - 15, ref_y - 45), (ref_x + 15, ref_y), (15, 15, 15), -1)  # Black jersey
        frame_detections.append({"track_id": 5, "class": 0, "bbox": [ref_x - 15, ref_y - 45, ref_x + 15, ref_y]})
        # Sports Ball (track ID 6): flying across the pitch.
        # Intentionally introduce tracking gaps (frames 40-42) to validate interpolation
        ball_x = int(350 + frame_idx * 4.2)
        ball_y = int(280 + frame_idx * 2.1)
        cv2.circle(frame, (ball_x, ball_y), 8, (0, 165, 255), -1)  # Orange ball
        if not (40 <= frame_idx <= 42):
            frame_detections.append({"track_id": 6, "class": 32, "bbox": [ball_x - 8, ball_y - 8, ball_x + 8, ball_y + 8]})
        # Outside boundary Player (track ID 7): Out of bounds spectator (dropped by pipeline)
        spec_x = int(50)
        spec_y = int(150 + frame_idx * 0.1)
        cv2.rectangle(frame, (spec_x - 15, spec_y - 45), (spec_x + 15, spec_y), (0, 0, 255), -1)
        frame_detections.append({"track_id": 7, "class": 0, "bbox": [spec_x - 15, spec_y - 45, spec_x + 15, spec_y]})
        out.write(frame)
        mock_tracks.append(frame_detections)
    out.release()
    print(f"[Mock Video] Generated synthetic video at {filepath} ({total_frames} frames)")
    return mock_tracks
# =====================================================================
# Main Tracking Pipeline (reusable module entry point)
# =====================================================================
def process_video(input_video_path: str, output_dir: str) -> dict:
    """
    Runs YOLO detection + ByteTrack tracking on the given video.
    Parameters:
        input_video_path (str): Path to the input football video.
        output_dir (str): Folder where outputs should be saved.
    Returns:
        dict containing:
        {
            "tracked_video": "<path>",
            "detections_json": "<path>",
            "tracking_summary": "<path>"
        }
    """
    print("=" * 60)
    print("SmartMatch AI Tracker Pipeline Launching")
    print("=" * 60)

    os.makedirs(output_dir, exist_ok=True)

    print("[Mode] Running in live video processing mode...")
    if not YOLO_AVAILABLE:
        print("CRITICAL ERROR: 'ultralytics' YOLO package not found.")
        print("Please run: pip install ultralytics")
        sys.exit(1)
    if not os.path.exists(input_video_path):
        print(f"CRITICAL ERROR: Input video file not found at '{input_video_path}'")
        sys.exit(1)

    # Initialize video capture
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"CRITICAL ERROR: Failed to open video source at '{input_video_path}'")
        sys.exit(1)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Input Specs  : Resolution {width}x{height} | FPS: {fps:.2f} | Frame Count: {total_frames}")
    # Build the field boundary PITCH_POLYGON
    pitch_poly = np.array([
        [int(p[0] * width), int(p[1] * height)] for p in DEFAULT_PITCH_POLYGON_PCT
    ], dtype=np.int32)
    # Instantiate mapping & telemetry utilities
    mapper = HomographyMapper(pitch_poly)
    speed_calc = SpeedCalculator(fps)
    ball_processor = BallTrajectoryProcessor(max_gap_frames=3, smoothing_window=5)
    # Data collection dictionaries
    track_votes = defaultdict(list)      # track_id -> list of classifications ("Red Team", "White Team", "referee")
    player_tracks = defaultdict(list)    # track_id -> frame-by-frame data entries
    ball_detections_raw = {}            # frame_idx -> (cx, cy, conf)
    raw_frame_objects = []              # frame_idx -> list of raw detected bboxes
    # Load YOLO Model
    print("[Model] Loading YOLO model...")
    model = YOLO("yolo11s.pt")
    print("[Model] YOLO loaded successfully.")
    # =================================================================
    # PASS 1: Tracking & Team Color Classification Sampling
    # =================================================================
    print("\n[Pass 1/2] Initiating object tracking and jersey color sampling...")
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_detections = []
        # Run live YOLO detection
        results = model.track(
            frame, persist=True, tracker="bytetrack.yaml",
            conf=0.10, imgsz=1280, device="cpu", verbose=False
        )
        detections = []
        boxes = results[0].boxes
        if boxes.id is not None:
            ids = boxes.id.int().cpu().tolist()
            xyxy = boxes.xyxy.cpu().tolist()
            classes = boxes.cls.int().cpu().tolist()
            confidences = boxes.conf.cpu().tolist()
            for tid, box, cls_idx, conf in zip(ids, xyxy, classes, confidences):
                # COCO Person class is 0, Sports Ball is 32
                if cls_idx in [0, 32]:
                    detections.append({
                        "track_id": int(tid),
                        "class": int(cls_idx),
                        "bbox": [int(v) for v in box],
                        "confidence": float(conf)
                    })
        for det in detections:
            tid = det["track_id"]
            c_idx = det["class"]
            bbox = det["bbox"]
            conf = det.get("confidence", 0.90)
            x1, y1, x2, y2 = bbox
            # Bounding box center coordinates
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            # Evaluate footprint / bottom-center coordinates for out-of-bounds boundary filtering
            # For the ball, use center pixels; for players, use bottom feet line.
            eval_x, eval_y = (cx, cy) if c_idx == 32 else (cx, y2)
            # Drop any tracking entry instantly if it falls outside the field lines
            dist = cv2.pointPolygonTest(pitch_poly, (eval_x, eval_y), False)
            if dist < 0:
                continue
            if c_idx == 32:
                # Store ball coordinate
                ball_detections_raw[frame_idx] = (cx, cy, conf)
            else:
                # Extract upper torso crop (jersey region), inset slightly from the
                # left/right bbox edges to reduce grass/background bleed at the
                # silhouette edges before classification.
                torso_y2 = y1 + max(1, (y2 - y1) // 3)
                box_w = x2 - x1
                inset = max(0, int(box_w * 0.15))
                crop_x1 = max(0, x1 + inset)
                crop_y1 = max(0, y1)
                crop_x2 = min(width - 1, x2 - inset)
                crop_y2 = min(height - 1, torso_y2)
                if crop_x2 <= crop_x1:
                    crop_x1 = max(0, x1)
                    crop_x2 = min(width - 1, x2)
                if crop_x2 > crop_x1 and crop_y2 > crop_y1:
                    jersey_crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
                    color_cls = classify_jersey_hsv(jersey_crop)
                    track_votes[tid].append(color_cls)
                frame_detections.append({
                    "track_id": tid,
                    "bbox": bbox,
                    "confidence": conf
                })
        raw_frame_objects.append(frame_detections)
        frame_idx += 1
        if frame_idx % 30 == 0:
            print(f"  Processed {frame_idx}/{total_frames} frames...")
    cap.release()
    print(f"[Pass 1/2] Processing completed. Detections collected across {frame_idx} frames.")
    # =================================================================
    # Stabilize Classification: Run Majority Voting per Stable ID
    # =================================================================
    print("\n[Stabilize] Executing global majority vote classification for player IDs...")
    final_role_team = {}  # track_id -> (role, team)
    for tid, votes in track_votes.items():
        if not votes:
            final_role_team[tid] = (ROLE_PLAYER, TEAM_WHITE)
            continue
        winning_cls = Counter(votes).most_common(1)[0][0]
        if winning_cls == ROLE_REFEREE:
            final_role_team[tid] = (ROLE_REFEREE, None)
        elif winning_cls == TEAM_RED:
            final_role_team[tid] = (ROLE_PLAYER, TEAM_RED)
        else:
            final_role_team[tid] = (ROLE_PLAYER, TEAM_WHITE)
    # Compile Ball Trajectory
    print("[Stabilize] Resolving ball trajectory gaps and smoothing...")
    for f, (cx, cy, _) in ball_detections_raw.items():
        ball_processor.add_detection(f, cx, cy)
    smoothed_ball_path = ball_processor.process_trajectory()
    # =================================================================
    # PASS 2: Telemetry calculations & Annotation Rendering
    # =================================================================
    print("\n[Pass 2/2] Calculating physical telemetry, writing detections.json & rendering tracked_video.mp4...")
    cap = cv2.VideoCapture(input_video_path)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    # Output file paths (now rooted at the caller-supplied output_dir)
    output_video_path = os.path.join(output_dir, "tracked_video.mp4")
    output_json_path = os.path.join(output_dir, "detections.json")
    output_summary_path = os.path.join(output_dir, "tracking_summary.json")
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    if not out.isOpened():
        print(f"CRITICAL ERROR: Failed to instantiate output video file at '{output_video_path}'")
        sys.exit(1)
    json_records = []
    frame_idx = 0
    # Color Palette definitions for visualization overlays (BGR)
    bgr_red = (0, 0, 255)
    bgr_white = (255, 255, 255)
    bgr_referee = (0, 255, 255)  # Bright yellow outline for better contrast
    bgr_ball = (0, 165, 255)     # Orange circle for soccer ball
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        timestamp = float(frame_idx / fps) if fps > 0 else 0.0
        # Draw the pitch boundary polygon (semi-transparent overlay)
        overlay = frame.copy()
        cv2.polylines(overlay, [pitch_poly], isClosed=True, color=(255, 255, 255), thickness=2)
        cv2.fillPoly(overlay, [pitch_poly], color=(0, 255, 0))
        cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)
        # Draw complete ball trajectory trail up to current frame
        ball_trail = [smoothed_ball_path[f] for f in sorted(smoothed_ball_path.keys()) if f <= frame_idx]
        for i in range(1, len(ball_trail)):
            cv2.line(frame, ball_trail[i - 1], ball_trail[i], (0, 140, 255), 2)
        # Check if ball exists in current frame
        if frame_idx in smoothed_ball_path:
            bcx, bcy = smoothed_ball_path[frame_idx]

            # Map ball telemetry
            ball_x_pitch, ball_y_pitch = mapper.to_pitch_coords(bcx, bcy)

            # Draw ball indicator
            cv2.circle(frame, (bcx, bcy), 8, bgr_ball, -1)
            cv2.putText(frame, "Ball", (bcx - 15, bcy - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            # Strict key schema enforcement for ball detection output
            json_records.append({
                "frame": frame_idx,
                "timestamp": round(timestamp, 3),
                "player_id": 999,  # Constant standard ID representing the ball
                "class": ROLE_BALL,
                "bbox": [bcx - 8, bcy - 8, bcx + 8, bcy + 8],
                "confidence": 1.0,
                "x": float(bcx),
                "y": float(bcy),
                "x_pitch": round(ball_x_pitch, 2),
                "y_pitch": round(ball_y_pitch, 2),
                "current_speed": 0.0  # Speed calculations applied strictly to player/referee runs
            })
        # Process players and referees detected in this frame
        for player in raw_frame_objects[frame_idx]:
            tid = player["track_id"]
            x1, y1, x2, y2 = player["bbox"]
            conf = player["confidence"]
            # Footprint anchor point
            cx = (x1 + x2) // 2

            # Retrieve classification role/team
            role, team = final_role_team.get(tid, (ROLE_PLAYER, TEAM_WHITE))
            # Retrieve telemetry mapping
            x_pitch, y_pitch = mapper.to_pitch_coords(cx, y2)
            speed_val = speed_calc.calculate_speed(tid, x_pitch, y_pitch)
            # Determine colors and labels based on classification
            if role == ROLE_REFEREE:
                team_color_hex = None  # Omitted based on data schema requirements
                bgr_color = bgr_referee
                label = f"Ref {tid}"
            elif team == TEAM_RED:
                team_color_hex = HEX_RED
                bgr_color = bgr_red
                label = f"Red {tid}"
            else:
                team_color_hex = HEX_WHITE
                bgr_color = bgr_white
                label = f"White {tid}"
            # Visual overlay rendering
            cv2.rectangle(frame, (x1, y1), (x2, y2), bgr_color, 2)
            cv2.circle(frame, (cx, y2), 4, (0, 0, 255), -1)  # foot location dot
            cv2.putText(frame, f"{label} | {speed_val}kmh", (x1, max(y1 - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2)
            # Strict JSON schema formatting
            record = {
                "frame": frame_idx,
                "timestamp": round(timestamp, 3),
                "player_id": tid,
                "class": role,
                "bbox": [x1, y1, x2, y2],
                "confidence": round(conf, 2),
                "x": float(cx),
                "y": float(y2),
                "x_pitch": round(x_pitch, 2),
                "y_pitch": round(y_pitch, 2),
                "current_speed": speed_val
            }
            # Team and team_color keys are strictly omitted if class is referee
            if role != ROLE_REFEREE:
                record["team"] = team
                record["team_color"] = team_color_hex
            json_records.append(record)
        out.write(frame)
        frame_idx += 1
    cap.release()
    out.release()
    print(f"[Pass 2/2] Rendering finished. Output files generated successfully.")
    # Write detections JSON file
    with open(output_json_path, "w") as jf:
        json.dump(json_records, jf, indent=2)

    # Write a tracking summary file (module-level output requirement; does not
    # affect or alter any detection/tracking logic above).
    summary = {
        "input_video": input_video_path,
        "output_video": output_video_path,
        "detections_json": output_json_path,
        "resolution": {"width": width, "height": height},
        "fps": fps,
        "total_frames_processed": frame_idx,
        "total_detections_logged": len(json_records),
        "unique_player_ids": len(track_votes),
        "ball_trajectory_points": len(smoothed_ball_path),
    }
    with open(output_summary_path, "w") as sf:
        json.dump(summary, sf, indent=2)

    print("\n" + "=" * 60)
    print("SmartMatch AI Telemetry Summary:")
    print("=" * 60)
    print(f"Annotated Video Output : {output_video_path}")
    print(f"Detections JSON Log    : {output_json_path}")
    print(f"Tracking Summary       : {output_summary_path}")
    print(f"Total Detections Logged: {len(json_records)}")
    print(f"Unique Player IDs      : {len(track_votes)}")
    print(f"Ball Trajectory Points : {len(smoothed_ball_path)}")
    print("=" * 60)

    return {
        "tracked_video": output_video_path,
        "detections_json": output_json_path,
        "tracking_summary": output_summary_path,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SmartMatch AI Football Analysis Pipeline")
    parser.add_argument("--input", type=str, default="data/input.mp4", help="Path to input video file")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Folder where outputs should be saved")
    args = parser.parse_args()
    process_video(args.input, args.output_dir)