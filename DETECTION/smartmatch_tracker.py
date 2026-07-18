import os
import sys
import cv2
import json
import math
import argparse
import numpy as np
from collections import defaultdict

# Handle optional YOLO import to support clean mock execution on CPU environments
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# =====================================================================
# Configuration & Constants
# =====================================================================
DEFAULT_PITCH_POLYGON_PCT = [
    [0.15, 0.28],  # Top-Left (TL)
    [0.85, 0.28],  # Top-Right (TR)
    [0.94, 0.94],  # Bottom-Right (BR)
    [0.06, 0.94]   # Bottom-Left (BL)
]

# Grass/pitch background color -> excluded from jersey-color sampling so
# green edge-pixels around a torso crop can't bias the team color read.
GRASS_LOWER = np.array([35, 40, 40])
GRASS_UPPER = np.array([85, 255, 255])

ROLE_PLAYER = "player"
ROLE_REFEREE = "referee"
ROLE_BALL = "ball"

# Small named-color palette (BGR) used only to give a human-readable label
# to whatever two jersey colors k-means finds - not used for classification.
COLOR_NAME_PALETTE_BGR = {
    "Red": (0, 0, 255), "White": (255, 255, 255), "Black": (20, 20, 20),
    "Yellow": (0, 230, 255), "Blue": (255, 0, 0), "Sky Blue": (235, 206, 135),
    "Green": (0, 200, 0), "Orange": (0, 140, 255), "Purple": (128, 0, 128),
    "Pink": (203, 150, 255), "Gray": (150, 150, 150), "Navy": (128, 0, 0),
}


def nearest_color_name(bgr):
    b, g, r = [int(v) for v in bgr]
    best_name, best_dist = "Team", float("inf")
    for name, (pb, pg, pr) in COLOR_NAME_PALETTE_BGR.items():
        d = (pb - b) ** 2 + (pg - g) ** 2 + (pr - r) ** 2
        if d < best_dist:
            best_dist, best_name = d, name
    return best_name


def bgr_to_hex(bgr):
    b, g, r = [int(v) for v in bgr]
    return "#{:02X}{:02X}{:02X}".format(r, g, b)


# =====================================================================
# Homography Mapping (Physical Telemetry Calculation)
# =====================================================================
class HomographyMapper:
    """Translates screen pixels to a normalized 0-100 pitch coordinate grid."""

    def __init__(self, pitch_poly):
        self.src_pts = np.float32(pitch_poly)
        self.dst_pts = np.float32([[0, 0], [100, 0], [100, 100], [0, 100]])
        self.M = cv2.getPerspectiveTransform(self.src_pts, self.dst_pts)

    def to_pitch_coords(self, px, py):
        pts = np.array([[[px, py]]], dtype=np.float32)
        warped = cv2.perspectiveTransform(pts, self.M)
        x_pitch = float(np.clip(warped[0][0][0], 0.0, 100.0))
        y_pitch = float(np.clip(warped[0][0][1], 0.0, 100.0))
        return x_pitch, y_pitch


# =====================================================================
# Speed Telemetry Calculator
# =====================================================================
class SpeedCalculator:
    """Smoothed physical speed (km/h) from normalized pitch displacement."""

    def __init__(self, fps):
        self.fps = fps
        self.prev_pitch_coords = {}
        self.speed_history = defaultdict(list)

    def calculate_speed(self, track_id, x_pitch, y_pitch):
        dt = 1.0 / self.fps if self.fps > 0 else 0.033
        if track_id not in self.prev_pitch_coords:
            self.prev_pitch_coords[track_id] = (x_pitch, y_pitch)
            return 0.0
        x_prev, y_prev = self.prev_pitch_coords[track_id]
        dx_meters = (x_pitch - x_prev) * 1.05
        dy_meters = (y_pitch - y_prev) * 0.68
        dist_meters = math.hypot(dx_meters, dy_meters)
        raw_speed_kmh = (dist_meters / dt) * 3.6
        self.prev_pitch_coords[track_id] = (x_pitch, y_pitch)
        hist = self.speed_history[track_id]
        hist.append(raw_speed_kmh)
        if len(hist) > 5:
            hist.pop(0)
        return round(float(np.mean(hist)), 2)


# =====================================================================
# Dynamic Team / Referee Classifier (replaces fixed red/white HSV rules)
# =====================================================================
class TeamClassifier:
    """
    Learns jersey colors directly from the footage instead of assuming a
    fixed Red-vs-White palette (which silently misclassifies every match
    that isn't literally red vs. white). Referees are separated first via
    a low-lightness / low-chroma heuristic (dark kit), then the remaining
    tracks are grouped into exactly two clusters with k-means on CIE-Lab
    color - Lab is far more lighting-robust than raw BGR/HSV thresholds,
    so this holds up across broadcast footage, floodlights, and shadows.
    """

    REFEREE_L_MAX = 70
    REFEREE_CHROMA_MAX = 18

    def __init__(self):
        self.samples = defaultdict(list)  # track_id -> list[Lab color]

    def add_sample(self, track_id, crop):
        color = self._dominant_lab_color(crop)
        if color is not None:
            self.samples[track_id].append(color)

    @staticmethod
    def _dominant_lab_color(crop):
        if crop is None or crop.size == 0:
            return None
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        grass_mask = cv2.inRange(hsv, GRASS_LOWER, GRASS_UPPER)
        keep_mask = cv2.bitwise_not(grass_mask).reshape(-1)
        bgr_pixels = crop.reshape(-1, 3)
        filtered = bgr_pixels[keep_mask > 0]
        # If grass-masking strips almost everything (e.g. a green-kit team),
        # fall back to the raw crop rather than starving the sample.
        if filtered.shape[0] < max(10, int(bgr_pixels.shape[0] * 0.1)):
            filtered = bgr_pixels
        if filtered.shape[0] == 0:
            return None
        lab_pixels = cv2.cvtColor(filtered.reshape(-1, 1, 3), cv2.COLOR_BGR2LAB).reshape(-1, 3)
        return np.median(lab_pixels, axis=0)

    def classify_all(self, referee_ids=None):
        """
        Args:
          referee_ids: optional set of track_ids already identified as
            referees (e.g. by RefereeClassifier). When given, these are
            pulled out before clustering and the built-in dark-kit Lab
            heuristic below is skipped for them entirely - this keeps a
            dark navy/charcoal *team* kit from being clustered against
            actual referees. When omitted, falls back to the built-in
            lightness/chroma heuristic.
        Returns:
          role_team: track_id -> (role, team_name_or_None, team_hex_or_None, bgr_color)
          median_lab: track_id -> median Lab color (used by the stitcher)
        """
        track_ids = [tid for tid, s in self.samples.items() if s]
        if not track_ids:
            return {}, {}
        median_lab = {tid: np.median(np.array(self.samples[tid]), axis=0) for tid in track_ids}

        role_team = {}
        player_ids, player_lab = [], []
        for tid in track_ids:
            if referee_ids is not None:
                is_referee = tid in referee_ids
            else:
                L, a, b = median_lab[tid]
                chroma = math.hypot(a - 128, b - 128)
                is_referee = L < self.REFEREE_L_MAX and chroma < self.REFEREE_CHROMA_MAX
            if is_referee:
                role_team[tid] = (ROLE_REFEREE, None, None, (0, 255, 255))
            else:
                player_ids.append(tid)
                player_lab.append(median_lab[tid])

        if player_ids:
            data = np.float32(player_lab)
            if len(player_ids) >= 2:
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.5)
                _, labels, centers = cv2.kmeans(data, 2, None, criteria, 8, cv2.KMEANS_PP_CENTERS)
                labels = labels.flatten()
            else:
                labels = np.zeros(len(player_ids), dtype=int)
                centers = data

            cluster_meta = {}
            for k in range(len(centers)):
                lab_center = np.uint8([[np.clip(centers[k], 0, 255)]])
                bgr_center = cv2.cvtColor(lab_center, cv2.COLOR_LAB2BGR)[0][0]
                name = nearest_color_name(bgr_center)
                cluster_meta[k] = (f"{name} Team", bgr_to_hex(bgr_center), tuple(int(v) for v in bgr_center))

            for tid, cl in zip(player_ids, labels):
                team_name, team_hex, bgr_color = cluster_meta[int(cl)]
                role_team[tid] = (ROLE_PLAYER, team_name, team_hex, bgr_color)

        return role_team, median_lab


# =====================================================================
# Referee Classifier - identifies the dark/black kit separately from,
# and before, the team color clustering
# =====================================================================
class RefereeClassifier:
    """
    Identifies referees by their solid black/charcoal kit using a
    dedicated HSV dark-pixel mask, kept completely separate from the
    team k-means clustering. Two reasons this is its own classifier
    rather than folded into TeamClassifier:

      1. A dark navy or black *team* kit would otherwise get pulled into
         the same "low lightness" bucket as an actual referee and
         confuse the two-cluster team split.
      2. It can be tuned/debugged on its own (DARK_V_MAX, DARK_S_MAX,
         DOMINANCE_THRESHOLD) without touching how team colors are read.

    Per detection it measures what fraction of the (grass-excluded)
    jersey crop is "dark" (low saturation, low value in HSV - covers
    black, charcoal, near-black navy). Votes are averaged per track
    across every frame that track was seen, so a single bad-lighting
    frame can't flip the call - the referee kit has to be dark on
    average across the whole track.
    """

    DARK_V_MAX = 60          # value (brightness) ceiling for "dark" pixels
    DARK_S_MAX = 120         # saturation ceiling - excludes vivid dark team colors (e.g. deep red)
    DOMINANCE_THRESHOLD = 0.35  # fraction of the crop that must read as dark

    def __init__(self, dark_v_max=None, dark_s_max=None, dominance_threshold=None):
        self.dark_v_max = dark_v_max if dark_v_max is not None else self.DARK_V_MAX
        self.dark_s_max = dark_s_max if dark_s_max is not None else self.DARK_S_MAX
        self.dominance_threshold = (
            dominance_threshold if dominance_threshold is not None else self.DOMINANCE_THRESHOLD
        )
        self.dark_ratio_samples = defaultdict(list)

    def add_sample(self, track_id, crop):
        ratio = self._dark_pixel_ratio(crop)
        if ratio is not None:
            self.dark_ratio_samples[track_id].append(ratio)

    def _dark_pixel_ratio(self, crop):
        if crop is None or crop.size == 0:
            return None
        total = crop.shape[0] * crop.shape[1]
        if total == 0:
            return None
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        grass_mask = cv2.inRange(hsv, GRASS_LOWER, GRASS_UPPER)
        dark_mask = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, self.dark_s_max, self.dark_v_max]))
        dark_mask = cv2.bitwise_and(dark_mask, cv2.bitwise_not(grass_mask))
        return cv2.countNonZero(dark_mask) / total

    def identify_referees(self):
        """Returns the set of track_ids whose kit reads as dark on average."""
        referee_ids = set()
        for tid, ratios in self.dark_ratio_samples.items():
            if ratios and float(np.mean(ratios)) >= self.dominance_threshold:
                referee_ids.add(tid)
        return referee_ids


# =====================================================================
# Track Stitcher - merges fragmented IDs back into one physical person
# =====================================================================
class UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, x):
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x, y):
        rx, ry = self.find(x), self.find(y)
        if rx != ry:
            self.parent[rx] = ry


class TrackStitcher:
    """
    Trackers like ByteTrack assign a brand-new ID whenever a track is lost
    for too long - a collision, a brief occlusion by another player, or a
    dip below the confidence threshold. Left alone this inflates the
    tracked-person count well past the number of real people on the pitch
    (22 players can easily surface as 30-40 raw IDs).

    This stitches broken tracks back together: track A (which just ended)
    is merged into track B (which starts shortly after) when the gap is
    short, the position is plausible for that gap (bounded by a per-frame
    max speed), the role matches (referee only merges with referee), and -
    for players - the jersey color is close enough that it's very unlikely
    to be a different person.
    """

    def __init__(self, max_frame_gap=20, max_pixel_dist_per_frame=12, base_pixel_dist=40, color_dist_thresh=20):
        self.max_frame_gap = max_frame_gap
        self.max_pixel_dist_per_frame = max_pixel_dist_per_frame
        self.base_pixel_dist = base_pixel_dist
        self.color_dist_thresh = color_dist_thresh

    def stitch(self, track_meta, median_lab, role_team):
        tids = list(track_meta.keys())
        uf = UnionFind()
        for tid in tids:
            uf.find(tid)

        ends_sorted = sorted(tids, key=lambda t: track_meta[t]["last_frame"])
        used_as_start = set()

        for e in ends_sorted:
            e_meta = track_meta[e]
            e_role = role_team.get(e, (ROLE_PLAYER,))[0]
            best_s, best_cost = None, None
            for s in tids:
                if s == e or s in used_as_start:
                    continue
                s_meta = track_meta[s]
                gap = s_meta["first_frame"] - e_meta["last_frame"]
                if gap <= 0 or gap > self.max_frame_gap:
                    continue
                s_role = role_team.get(s, (ROLE_PLAYER,))[0]
                if e_role != s_role:
                    continue
                dist_px = math.hypot(
                    s_meta["first_pos"][0] - e_meta["last_pos"][0],
                    s_meta["first_pos"][1] - e_meta["last_pos"][1],
                )
                allowed_dist = self.base_pixel_dist + self.max_pixel_dist_per_frame * gap
                if dist_px > allowed_dist:
                    continue
                if e in median_lab and s in median_lab:
                    color_dist = float(np.linalg.norm(median_lab[e] - median_lab[s]))
                    if color_dist > self.color_dist_thresh:
                        continue
                else:
                    color_dist = 0.0
                cost = dist_px + gap * 3 + color_dist
                if best_cost is None or cost < best_cost:
                    best_cost, best_s = cost, s
            if best_s is not None:
                uf.union(e, best_s)
                used_as_start.add(best_s)

        return {tid: uf.find(tid) for tid in tids}


# =====================================================================
# Ball Trajectory Processor
# =====================================================================
class BallTrajectoryProcessor:
    """Frame-gap interpolation + rolling-window smoothing for the ball."""

    def __init__(self, max_gap_frames=3, smoothing_window=5):
        self.max_gap_frames = max_gap_frames
        self.smoothing_window = smoothing_window
        self.raw_detections = {}

    def add_detection(self, frame_idx, cx, cy):
        self.raw_detections[frame_idx] = (cx, cy)

    def process_trajectory(self):
        interpolated = {}
        frames = sorted(self.raw_detections.keys())
        if not frames:
            return interpolated
        for i in range(len(frames) - 1):
            f_start, f_end = frames[i], frames[i + 1]
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
        interpolated[frames[-1]] = self.raw_detections[frames[-1]]

        smoothed = {}
        interp_frames = sorted(interpolated.keys())
        for idx, f in enumerate(interp_frames):
            window_coords = []
            for offset in range(-self.smoothing_window // 2 + 1, self.smoothing_window // 2 + 1):
                curr_idx = idx + offset
                if 0 <= curr_idx < len(interp_frames):
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
# Custom ByteTrack config - longer track_buffer means the tracker itself
# survives brief occlusions instead of minting a new ID, so the stitcher
# has less work to do and merges are more likely to be correct.
# =====================================================================
def write_custom_tracker_yaml(path, track_buffer=60):
    content = f"""tracker_type: bytetrack
track_high_thresh: 0.25
track_low_thresh: 0.1
new_track_thresh: 0.25
track_buffer: {track_buffer}
match_thresh: 0.8
fuse_score: True
"""
    with open(path, "w") as f:
        f.write(content)
    return path


# =====================================================================
# Synthetic Match Video Generator (Mock Mode Support)
# =====================================================================
def generate_mock_video(filepath, width=1280, height=720, total_frames=150):
    """
    Generates a synthetic football video and returns ground-truth
    per-frame detections so the full pipeline (color clustering, track
    stitching, ball interpolation) can be exercised without a YOLO model.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(filepath, fourcc, 30.0, (width, height))
    pitch_poly = np.array(
        [[int(p[0] * width), int(p[1] * height)] for p in DEFAULT_PITCH_POLYGON_PCT], dtype=np.int32
    )
    mock_tracks = []
    for frame_idx in range(total_frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:] = (34, 139, 34)
        cv2.polylines(frame, [pitch_poly], isClosed=True, color=(255, 255, 255), thickness=3)
        frame_detections = []

        def draw_player(tid, x, y, bgr, drop=False):
            if not drop:
                cv2.rectangle(frame, (x - 15, y - 45), (x + 15, y), bgr, -1)
                frame_detections.append({"track_id": tid, "class": 0, "bbox": [x - 15, y - 45, x + 15, y]})

        # Red Player 1 - deliberately loses tracking for 5 frames mid-clip
        # (simulating an occlusion) to exercise the track stitcher: it
        # reappears under a NEW id (101) a few frames later at a nearby spot.
        p1_x, p1_y = int(300 + frame_idx * 2.5), int(300 + frame_idx * 1.5)
        occluded = 60 <= frame_idx <= 64
        if not occluded:
            tid = 1 if frame_idx < 60 else 101
            draw_player(tid, p1_x, p1_y, (0, 0, 255))

        p2_x, p2_y = int(400 + frame_idx * 1.8), int(250 + frame_idx * 0.2)
        draw_player(2, p2_x, p2_y, (0, 0, 255))

        p3_x, p3_y = int(900 - frame_idx * 2.2), int(400 + frame_idx * 0.5)
        draw_player(3, p3_x, p3_y, (255, 255, 255))

        p4_x, p4_y = int(1000 - frame_idx * 2.0), int(550 - frame_idx * 1.2)
        draw_player(4, p4_x, p4_y, (255, 255, 255))

        ref_x, ref_y = int(640 + frame_idx * 0.5), int(320 + frame_idx * 0.3)
        draw_player(5, ref_x, ref_y, (15, 15, 15))

        ball_x, ball_y = int(350 + frame_idx * 4.2), int(280 + frame_idx * 2.1)
        cv2.circle(frame, (ball_x, ball_y), 8, (0, 165, 255), -1)
        if not (40 <= frame_idx <= 42):
            frame_detections.append({"track_id": 6, "class": 32, "bbox": [ball_x - 8, ball_y - 8, ball_x + 8, ball_y + 8]})

        # Out-of-bounds spectator - must be dropped by the pitch-polygon filter
        spec_x, spec_y = 50, int(150 + frame_idx * 0.1)
        draw_player(7, spec_x, spec_y, (0, 0, 255))

        out.write(frame)
        mock_tracks.append(frame_detections)
    out.release()
    print(f"[Mock Video] Generated synthetic video at {filepath} ({total_frames} frames)")
    return mock_tracks


# =====================================================================
# Main Tracking Pipeline
# =====================================================================
def process_video(
    input_video_path: str,
    output_dir: str,
    mock: bool = False,
    expected_players: int = None,
    stitch_max_gap: int = 20,
    stitch_max_dist_per_frame: int = 12,
    stitch_color_thresh: float = 20.0,
    referee_dark_v_max: int = RefereeClassifier.DARK_V_MAX,
    referee_dark_s_max: int = RefereeClassifier.DARK_S_MAX,
    referee_dominance_thresh: float = RefereeClassifier.DOMINANCE_THRESHOLD,
) -> dict:
    print("=" * 60)
    print("SmartMatch AI Tracker Pipeline Launching")
    print("=" * 60)

    os.makedirs(output_dir, exist_ok=True)

    mock_frame_detections = None
    if mock:
        print("[Mode] Running in MOCK mode (synthetic video, no YOLO required)...")
        mock_video_path = os.path.join(output_dir, "mock_input.mp4")
        mock_frame_detections = generate_mock_video(mock_video_path)
        input_video_path = mock_video_path
    else:
        print("[Mode] Running in live video processing mode...")
        if not YOLO_AVAILABLE:
            print("CRITICAL ERROR: 'ultralytics' YOLO package not found.")
            print("Please run: pip install ultralytics")
            sys.exit(1)
        if not os.path.exists(input_video_path):
            print(f"CRITICAL ERROR: Input video file not found at '{input_video_path}'")
            sys.exit(1)

    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"CRITICAL ERROR: Failed to open video source at '{input_video_path}'")
        sys.exit(1)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Input Specs  : Resolution {width}x{height} | FPS: {fps:.2f} | Frame Count: {total_frames}")

    pitch_poly = np.array(
        [[int(p[0] * width), int(p[1] * height)] for p in DEFAULT_PITCH_POLYGON_PCT], dtype=np.int32
    )
    mapper = HomographyMapper(pitch_poly)
    speed_calc = SpeedCalculator(fps)
    ball_processor = BallTrajectoryProcessor(max_gap_frames=3, smoothing_window=5)
    team_classifier = TeamClassifier()
    referee_classifier = RefereeClassifier(
        dark_v_max=referee_dark_v_max, dark_s_max=referee_dark_s_max,
        dominance_threshold=referee_dominance_thresh,
    )

    track_meta = {}          # track_id -> {first_frame,last_frame,first_pos,last_pos}
    player_tracks = defaultdict(list)
    ball_detections_raw = {}
    raw_frame_objects = []

    model = None
    device = "cpu"
    tracker_yaml = None
    if not mock:
        if TORCH_AVAILABLE and torch.cuda.is_available():
            device = "0"
        tracker_yaml = write_custom_tracker_yaml(os.path.join(output_dir, "bytetrack_custom.yaml"))
        print(f"[Model] Loading YOLO model (device={device})...")
        model = YOLO("yolo11s.pt")
        print("[Model] YOLO loaded successfully.")

    # =================================================================
    # PASS 1: Tracking & jersey-color sampling
    # =================================================================
    print("\n[Pass 1/2] Initiating object tracking and jersey color sampling...")
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detections = []
        if mock:
            detections = [dict(d, confidence=0.95) for d in mock_frame_detections[frame_idx]]
        else:
            results = model.track(
                frame, persist=True, tracker=tracker_yaml,
                conf=0.10, imgsz=1280, device=device, verbose=False, classes=[0, 32],
            )
            boxes = results[0].boxes
            if boxes.id is not None:
                ids = boxes.id.int().cpu().tolist()
                xyxy = boxes.xyxy.cpu().tolist()
                classes = boxes.cls.int().cpu().tolist()
                confidences = boxes.conf.cpu().tolist()
                for tid, box, cls_idx, conf in zip(ids, xyxy, classes, confidences):
                    detections.append({
                        "track_id": int(tid), "class": int(cls_idx),
                        "bbox": [int(v) for v in box], "confidence": float(conf),
                    })

        frame_detections = []
        for det in detections:
            tid = det["track_id"]
            c_idx = det["class"]
            x1, y1, x2, y2 = det["bbox"]
            conf = det.get("confidence", 0.90)
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            eval_x, eval_y = (cx, cy) if c_idx == 32 else (cx, y2)

            # Drop anything outside the pitch boundary immediately
            if cv2.pointPolygonTest(pitch_poly, (eval_x, eval_y), False) < 0:
                continue

            if c_idx == 32:
                ball_detections_raw[frame_idx] = (cx, cy, conf)
                continue

            # Upper-torso jersey crop, inset from the left/right edges to
            # cut down on background bleed at the silhouette boundary
            torso_y2 = y1 + max(1, (y2 - y1) // 3)
            box_w = x2 - x1
            inset = max(0, int(box_w * 0.15))
            crop_x1 = max(0, x1 + inset)
            crop_y1 = max(0, y1)
            crop_x2 = min(width - 1, x2 - inset)
            crop_y2 = min(height - 1, torso_y2)
            if crop_x2 <= crop_x1:
                crop_x1, crop_x2 = max(0, x1), min(width - 1, x2)
            if crop_x2 > crop_x1 and crop_y2 > crop_y1:
                jersey_crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
                team_classifier.add_sample(tid, jersey_crop)
                referee_classifier.add_sample(tid, jersey_crop)

            if tid not in track_meta:
                track_meta[tid] = {"first_frame": frame_idx, "last_frame": frame_idx,
                                    "first_pos": (cx, y2), "last_pos": (cx, y2)}
            else:
                track_meta[tid]["last_frame"] = frame_idx
                track_meta[tid]["last_pos"] = (cx, y2)

            frame_detections.append({"track_id": tid, "bbox": [x1, y1, x2, y2], "confidence": conf})

        raw_frame_objects.append(frame_detections)
        frame_idx += 1
        if frame_idx % 30 == 0:
            print(f"  Processed {frame_idx}/{total_frames} frames...")
    cap.release()
    print(f"[Pass 1/2] Processing completed. Detections collected across {frame_idx} frames.")

    # =================================================================
    # Stabilize: classify (dynamic color clustering), then stitch broken
    # tracks back into single physical people, then re-classify on the
    # merged (canonical) ids for a cleaner team-color read.
    # =================================================================
    print("\n[Stabilize] Identifying referees by dark-kit ratio (separate from team colors)...")
    prelim_referee_ids = referee_classifier.identify_referees()
    print(f"[Stabilize] {len(prelim_referee_ids)} track(s) flagged as referee by kit darkness.")

    print("[Stabilize] Classifying remaining tracks by jersey color (k-means, not fixed thresholds)...")
    prelim_role_team, prelim_median_lab = team_classifier.classify_all(referee_ids=prelim_referee_ids)

    print("[Stabilize] Stitching fragmented track IDs back into single players...")
    stitcher = TrackStitcher(
        max_frame_gap=stitch_max_gap,
        max_pixel_dist_per_frame=stitch_max_dist_per_frame,
        color_dist_thresh=stitch_color_thresh,
    )
    id_map = stitcher.stitch(track_meta, prelim_median_lab, prelim_role_team)
    num_merges = sum(1 for tid, canon in id_map.items() if tid != canon)
    print(f"[Stabilize] Merged {num_merges} fragmented track id(s) into existing players.")

    # Remap everything through the canonical id
    canonical_samples = defaultdict(list)
    for tid, samples in team_classifier.samples.items():
        canonical_samples[id_map.get(tid, tid)].extend(samples)
    canonical_referee_ids = {id_map.get(tid, tid) for tid in prelim_referee_ids}
    merged_classifier = TeamClassifier()
    merged_classifier.samples = canonical_samples
    final_role_team, _ = merged_classifier.classify_all(referee_ids=canonical_referee_ids)

    for frame_dets in raw_frame_objects:
        for det in frame_dets:
            det["track_id"] = id_map.get(det["track_id"], det["track_id"])

    num_players = sum(1 for r in final_role_team.values() if r[0] == ROLE_PLAYER)
    num_referees = sum(1 for r in final_role_team.values() if r[0] == ROLE_REFEREE)
    print(f"[Stabilize] Final unique identities: {num_players} player(s), {num_referees} referee(s).")
    if expected_players is not None and num_players != expected_players:
        print(f"[Stabilize] WARNING: expected {expected_players} players but found {num_players}. "
              f"Tune --stitch-gap / --stitch-dist / --stitch-color-thresh and re-run.")

    print("[Stabilize] Resolving ball trajectory gaps and smoothing...")
    for f, (cx, cy, _) in ball_detections_raw.items():
        ball_processor.add_detection(f, cx, cy)
    smoothed_ball_path = ball_processor.process_trajectory()

    # =================================================================
    # PASS 2: Telemetry calculations & annotation rendering
    # =================================================================
    print("\n[Pass 2/2] Calculating physical telemetry, writing detections.json & rendering tracked_video.mp4...")
    cap = cv2.VideoCapture(input_video_path)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    output_video_path = os.path.join(output_dir, "tracked_video.mp4")
    output_json_path = os.path.join(output_dir, "detections.json")
    output_summary_path = os.path.join(output_dir, "tracking_summary.json")
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    if not out.isOpened():
        print(f"CRITICAL ERROR: Failed to instantiate output video file at '{output_video_path}'")
        sys.exit(1)

    json_records = []
    frame_idx = 0
    bgr_ball = (0, 165, 255)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        timestamp = float(frame_idx / fps) if fps > 0 else 0.0

        overlay = frame.copy()
        cv2.polylines(overlay, [pitch_poly], isClosed=True, color=(255, 255, 255), thickness=2)
        cv2.fillPoly(overlay, [pitch_poly], color=(0, 255, 0))
        cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)

        ball_trail = [smoothed_ball_path[f] for f in sorted(smoothed_ball_path.keys()) if f <= frame_idx]
        for i in range(1, len(ball_trail)):
            cv2.line(frame, ball_trail[i - 1], ball_trail[i], (0, 140, 255), 2)

        if frame_idx in smoothed_ball_path:
            bcx, bcy = smoothed_ball_path[frame_idx]
            ball_x_pitch, ball_y_pitch = mapper.to_pitch_coords(bcx, bcy)
            cv2.circle(frame, (bcx, bcy), 8, bgr_ball, -1)
            cv2.putText(frame, "Ball", (bcx - 15, bcy - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            json_records.append({
                "frame": frame_idx, "timestamp": round(timestamp, 3), "player_id": 999,
                "class": ROLE_BALL, "bbox": [bcx - 8, bcy - 8, bcx + 8, bcy + 8], "confidence": 1.0,
                "x": float(bcx), "y": float(bcy),
                "x_pitch": round(ball_x_pitch, 2), "y_pitch": round(ball_y_pitch, 2),
                "current_speed": 0.0,
            })

        for player in raw_frame_objects[frame_idx]:
            tid = player["track_id"]
            x1, y1, x2, y2 = player["bbox"]
            conf = player["confidence"]
            cx = (x1 + x2) // 2

            role, team, team_hex, bgr_color = final_role_team.get(
                tid, (ROLE_PLAYER, "Unclassified Team", None, (255, 255, 255))
            )
            x_pitch, y_pitch = mapper.to_pitch_coords(cx, y2)
            speed_val = speed_calc.calculate_speed(tid, x_pitch, y_pitch)
            label = f"Ref {tid}" if role == ROLE_REFEREE else f"{team.split()[0]} {tid}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), bgr_color, 2)
            cv2.circle(frame, (cx, y2), 4, (0, 0, 255), -1)
            cv2.putText(frame, f"{label} | {speed_val}kmh", (x1, max(y1 - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2)

            record = {
                "frame": frame_idx, "timestamp": round(timestamp, 3), "player_id": tid,
                "class": role, "bbox": [x1, y1, x2, y2], "confidence": round(conf, 2),
                "x": float(cx), "y": float(y2),
                "x_pitch": round(x_pitch, 2), "y_pitch": round(y_pitch, 2),
                "current_speed": speed_val,
            }
            if role != ROLE_REFEREE:
                record["team"] = team
                record["team_color"] = team_hex
            json_records.append(record)

        out.write(frame)
        frame_idx += 1
    cap.release()
    out.release()
    print("[Pass 2/2] Rendering finished. Output files generated successfully.")

    with open(output_json_path, "w") as jf:
        json.dump(json_records, jf, indent=2)

    summary = {
        "input_video": input_video_path,
        "output_video": output_video_path,
        "detections_json": output_json_path,
        "resolution": {"width": width, "height": height},
        "fps": fps,
        "total_frames_processed": frame_idx,
        "total_detections_logged": len(json_records),
        "unique_player_ids": num_players,
        "unique_referee_ids": num_referees,
        "fragmented_ids_merged": num_merges,
        "ball_trajectory_points": len(smoothed_ball_path),
        "teams": sorted({v[1] for v in final_role_team.values() if v[0] == ROLE_PLAYER}),
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
    print(f"Unique Player IDs      : {num_players}  (merged {num_merges} fragmented id(s))")
    print(f"Unique Referee IDs     : {num_referees}")
    print(f"Ball Trajectory Points : {len(smoothed_ball_path)}")
    print(f"Teams Detected         : {summary['teams']}")
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
    parser.add_argument("--mock", action="store_true", help="Run on a generated synthetic video, no YOLO needed")
    parser.add_argument("--expected-players", type=int, default=None, help="Sanity-check the final player count")
    parser.add_argument("--stitch-gap", type=int, default=20, help="Max frame gap to bridge when stitching IDs")
    parser.add_argument("--stitch-dist", type=int, default=12, help="Max px/frame of travel allowed when stitching")
    parser.add_argument("--stitch-color-thresh", type=float, default=20.0, help="Max Lab color distance to stitch")
    parser.add_argument("--referee-dark-v-max", type=int, default=RefereeClassifier.DARK_V_MAX,
                         help="HSV Value ceiling for a pixel to count as 'dark' (0-255, lower = stricter black)")
    parser.add_argument("--referee-dark-s-max", type=int, default=RefereeClassifier.DARK_S_MAX,
                         help="HSV Saturation ceiling for a pixel to count as 'dark' (excludes vivid dark colors)")
    parser.add_argument("--referee-dominance-thresh", type=float, default=RefereeClassifier.DOMINANCE_THRESHOLD,
                         help="Fraction of a track's jersey crops that must read as dark to call it a referee")
    args = parser.parse_args()
    process_video(
        args.input, args.output_dir, mock=args.mock, expected_players=args.expected_players,
        stitch_max_gap=args.stitch_gap, stitch_max_dist_per_frame=args.stitch_dist,
        stitch_color_thresh=args.stitch_color_thresh,
        referee_dark_v_max=args.referee_dark_v_max, referee_dark_s_max=args.referee_dark_s_max,
        referee_dominance_thresh=args.referee_dominance_thresh,
    )
