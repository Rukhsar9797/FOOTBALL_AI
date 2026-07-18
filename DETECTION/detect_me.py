"""
SmartMatch AI - Member 1 (Detection & Tracking)  [v2 - optimized]
-------------------------------------------------------------------
Detects players, referees, and the ball in a football video, assigns
each a STABLE, human-readable id across frames (ByteTrack + custom
re-id layer), splits players into Team A / Team B by jersey-color
clustering (referee = the darkest / near-black cluster, not "the
smallest one"), draws the BALL'S FULL TRAJECTORY across the whole
video (interpolating through frames where the ball is flying too
fast/blurred to detect), and reports tracking-accuracy + performance
metrics.

ARCHITECTURE (why two passes):
  Pass 1 (analysis) - run YOLO+ByteTrack ONCE over every frame,
      collect raw per-frame detections + jersey-color samples for
      every track. No drawing, no video writing -> this is the
      expensive GPU/CPU step and it only happens once.
  Finalize - with the *whole video's* data in hand: cluster jersey
      colors, decide Team A / Team B / Referee per track (majority
      vote, so a track can't flicker teams mid-video), assign
      sequential "Player N" / "Referee N" labels in order of first
      appearance, and interpolate the ball's path through gaps.
  Pass 2 (render) - re-read the video and draw the now-finalized
      labels/teams/trajectory onto every frame. This pass is cheap
      (no model inference), so re-reading the video from disk here
      instead of caching every frame in RAM keeps memory flat
      regardless of video length.

This is both faster (single inference pass) and more correct (team
assignment and the ball path are computed from full-video context
instead of a small rolling window).

Output:
    outputs/tracked_video.mp4
    outputs/detections.json
    outputs/tracking_summary.json
"""

from ultralytics import YOLO
import cv2
import json
import os
import math
import time
import numpy as np
from collections import defaultdict, Counter
from datetime import datetime

try:
    import torch
    CUDA_AVAILABLE = torch.cuda.is_available()
except ImportError:
    CUDA_AVAILABLE = False

# ==================================================
# Config
# ==================================================
INPUT_VIDEO = "data/input.mp4"
MODEL_PATH = "yolo11s.pt"          # swap for a football-fine-tuned model if you get one
IMG_SIZE = 1280 if CUDA_AVAILABLE else 960   # bigger helps small/fast ball detection; drop
                                              # this first if you need more CPU speed
PERSON_CONF = 0.30                 # confidence threshold for players/referees
BALL_CONF = 0.08                   # kept low on purpose - ball is small, fast, and often
                                    # motion-blurred; interpolation (below) covers the rest
BALL_CLASS_NAME = "sports ball"    # COCO class name used by yolo11s.pt

# ---- Model optimization ----
DEVICE = 0 if CUDA_AVAILABLE else "cpu"
USE_HALF_PRECISION = CUDA_AVAILABLE       # fp16 - ~2x faster on supported GPUs
cv2.setNumThreads(max(1, os.cpu_count() or 1))
# NOTE: if you need real-time on CPU, drop IMG_SIZE (e.g. 960 -> 736) rather than
# skipping frames - skipping frames loses ball trajectory data.

# ---- ID stabilization ----
MAX_LOST_FRAMES = 30
MAX_MATCH_DISTANCE = 150
APPEARANCE_THRESHOLD = 0.5

# ---- Team / referee classification (jersey-color clustering) ----
N_COLOR_CLUSTERS = 3          # Team A, Team B, Referee
MIN_SAMPLES_TO_CLUSTER = 20   # need at least this many jersey-color samples total

# ---- Ball trajectory ----
MAX_BALL_GAP_FRAMES = 45      # bridge gaps up to ~1.5s @30fps (ball flying / blurred)
TRAJECTORY_THICKNESS = 2
TRAJECTORY_FADE_TAIL = None   # set an int (e.g. 150) to only draw the last N points of the
                               # trail instead of the whole match; None = draw entire path

os.makedirs("outputs", exist_ok=True)


# ==================================================
# ID Stabilizer  (ByteTrack -> stable unique ids)
# ==================================================
# ByteTrack occasionally loses an object for a few frames (occlusion,
# overlap, motion blur) and re-assigns a brand new raw track id when it
# reappears. This keeps short-term memory of lost tracks (position + an
# HSV color-histogram "fingerprint") so a reappearing object gets its
# OLD stable id back instead of a new one.
# ==================================================
class IDStabilizer:
    def __init__(self, max_lost_frames=30, max_distance=150, appearance_threshold=0.5):
        self.next_stable_id = 1
        self.raw_to_stable = {}
        self.active = {}
        self.lost = {}
        self.max_lost_frames = max_lost_frames
        self.max_distance = max_distance
        self.appearance_threshold = appearance_threshold
        self.track_frame_counts = {}
        self.first_seen_frame = {}     # stable_id -> first frame number (for ordering)
        self.reassignment_count = 0

    @staticmethod
    def _compute_feature(frame, bbox):
        x1, y1, x2, y2 = bbox
        x1, y1 = max(0, x1), max(0, y1)
        x2 = min(frame.shape[1] - 1, x2)
        y2 = min(frame.shape[0] - 1, y2)
        if x2 <= x1 or y2 <= y1:
            return None
        crop = frame[y1:y2, x1:x2]
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [16, 16], [0, 180, 0, 256])
        cv2.normalize(hist, hist)
        return hist

    @staticmethod
    def _similarity(feat_a, feat_b):
        if feat_a is None or feat_b is None:
            return 0.0
        return float(cv2.compareHist(feat_a, feat_b, cv2.HISTCMP_CORREL))

    def _match_lost(self, center, feature, frame_number):
        best_sid, best_score = None, -1.0
        for sid, info in self.lost.items():
            if frame_number - info["last_frame"] > self.max_lost_frames:
                continue
            dist = math.hypot(center[0] - info["center"][0], center[1] - info["center"][1])
            if dist > self.max_distance:
                continue
            sim = self._similarity(feature, info["feature"])
            if sim < self.appearance_threshold:
                continue
            score = sim - (dist / self.max_distance) * 0.3
            if score > best_score:
                best_score, best_sid = score, sid
        return best_sid

    def update(self, frame, frame_number, raw_ids, bboxes):
        current_raw_set = set(raw_ids)
        stable_ids_out = []

        for raw_id, bbox in zip(raw_ids, bboxes):
            feature = self._compute_feature(frame, bbox)
            center = ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)

            if raw_id in self.raw_to_stable:
                stable_id = self.raw_to_stable[raw_id]
            else:
                stable_id = self._match_lost(center, feature, frame_number)
                if stable_id is None:
                    stable_id = self.next_stable_id
                    self.next_stable_id += 1
                else:
                    self.reassignment_count += 1
                self.raw_to_stable[raw_id] = stable_id
                self.lost.pop(stable_id, None)

            if stable_id not in self.first_seen_frame:
                self.first_seen_frame[stable_id] = frame_number

            self.active[stable_id] = {
                "center": center, "bbox": bbox,
                "feature": feature, "last_frame": frame_number
            }
            self.track_frame_counts[stable_id] = self.track_frame_counts.get(stable_id, 0) + 1
            stable_ids_out.append(stable_id)

        for raw_id in [r for r in list(self.raw_to_stable.keys()) if r not in current_raw_set]:
            stable_id = self.raw_to_stable.pop(raw_id)
            info = self.active.pop(stable_id, None)
            if info:
                self.lost[stable_id] = info

        expired = [sid for sid, info in self.lost.items()
                   if frame_number - info["last_frame"] > self.max_lost_frames]
        for sid in expired:
            del self.lost[sid]

        return stable_ids_out

    def accuracy_report(self):
        if not self.track_frame_counts:
            return {
                "avg_track_length_frames": 0, "shortest_track_frames": 0,
                "longest_track_frames": 0, "fragmented_tracks_under_5_frames": 0,
                "successful_id_recoveries": self.reassignment_count
            }
        lengths = list(self.track_frame_counts.values())
        return {
            "avg_track_length_frames": round(sum(lengths) / len(lengths), 1),
            "shortest_track_frames": min(lengths),
            "longest_track_frames": max(lengths),
            "fragmented_tracks_under_5_frames": sum(1 for l in lengths if l < 5),
            "successful_id_recoveries": self.reassignment_count
        }


# ==================================================
# Team / Referee Classifier (jersey-color clustering)
# ==================================================
# COCO-pretrained YOLO only outputs "person" - no team/referee info.
# We sample the upper-third of every person bbox (the jersey) across
# the WHOLE video, then run k-means (k=3) once at the end over every
# sample. The cluster with the lowest mean lightness (L in Lab space)
# is the referee - referees wear black/very dark kits, and this holds
# regardless of how many referee detections there are, unlike the old
# "smallest cluster = referee" heuristic. The other two clusters are
# Team A and Team B, colored by their own centroid color.
#
# Each track's final role/team is a MAJORITY VOTE across all of its
# own samples, so a player can't flicker between teams frame-to-frame
# because of a shadow or motion blur on a single frame.
# ==================================================
class TeamClassifier:
    def __init__(self, k=3, min_samples=20):
        self.k = k
        self.min_samples = min_samples
        self._sample_colors = []      # Lab colors
        self._sample_track_ids = []
        self.centroids = None         # Lab
        self.referee_cluster = None
        self.team_clusters = []       # the two non-referee cluster indices, in order
        self.cluster_bgr = {}         # cluster_idx -> representative BGR for drawing
        self.track_cluster_votes = defaultdict(Counter)   # stable_id -> Counter(cluster_idx)
        self.track_role = {}          # stable_id -> "player" / "referee"
        self.track_team = {}          # stable_id -> "Team A" / "Team B" / None

    @staticmethod
    def _shirt_color(frame, bbox):
        x1, y1, x2, y2 = bbox
        x1, y1 = max(0, x1), max(0, y1)
        x2 = min(frame.shape[1] - 1, x2)
        y2 = min(frame.shape[0] - 1, y2)
        if x2 <= x1 or y2 <= y1:
            return None
        torso_y2 = y1 + max(1, (y2 - y1) // 3)   # upper third = torso/jersey, skips shorts/grass
        crop = frame[y1:torso_y2, x1:x2]
        if crop.size == 0:
            return None
        lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
        return lab.reshape(-1, 3).mean(axis=0)

    def add_sample(self, frame, bbox, stable_id):
        color = self._shirt_color(frame, bbox)
        if color is not None:
            self._sample_colors.append(color)
            self._sample_track_ids.append(stable_id)

    def finalize(self):
        """Call once after pass 1. Clusters every sample collected across the whole
        video and resolves a final role/team per stable_id via majority vote."""
        if len(self._sample_colors) < self.min_samples:
            print(f"[TeamClassifier] Only {len(self._sample_colors)} jersey samples "
                  f"collected - not enough to cluster reliably. All tracks default to "
                  f"'player' / unassigned team.")
            return

        data = np.array(self._sample_colors, dtype=np.float32)
        k = min(self.k, len(data))
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.1)
        _, labels, centers = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
        labels = labels.flatten()
        self.centroids = centers

        # referee = darkest cluster (lowest mean L / lightness channel), i.e. closest to black
        self.referee_cluster = int(np.argmin(centers[:, 0]))
        self.team_clusters = [i for i in range(k) if i != self.referee_cluster]

        for idx in range(k):
            lab_pixel = np.uint8([[centers[idx]]])
            bgr = cv2.cvtColor(lab_pixel, cv2.COLOR_LAB2BGR)[0, 0].tolist()
            self.cluster_bgr[idx] = tuple(int(v) for v in bgr)

        # tally which cluster each track's samples fell into
        for sid, lbl in zip(self._sample_track_ids, labels):
            self.track_cluster_votes[sid][int(lbl)] += 1

        team_names = {self.team_clusters[0]: "Team A"}
        if len(self.team_clusters) > 1:
            team_names[self.team_clusters[1]] = "Team B"

        for sid, votes in self.track_cluster_votes.items():
            winning_cluster = votes.most_common(1)[0][0]
            if winning_cluster == self.referee_cluster:
                self.track_role[sid] = "referee"
                self.track_team[sid] = None
            else:
                self.track_role[sid] = "player"
                self.track_team[sid] = team_names.get(winning_cluster, "Team B")

        counts = {i: sum(1 for l in labels if l == i) for i in range(k)}
        print(f"[TeamClassifier] Calibrated on {len(data)} jersey samples across "
              f"{len(self.track_cluster_votes)} tracks. Cluster sizes: {counts} -> "
              f"referee cluster = {self.referee_cluster} (darkest).")

    def role_and_team(self, stable_id):
        """Returns (role, team) - role is 'player'/'referee', team is 'Team A'/'Team B'/None.
        Defaults to 'player'/None for tracks with too few samples to have voted."""
        role = self.track_role.get(stable_id, "player")
        team = self.track_team.get(stable_id)
        return role, team

    def color_for(self, stable_id):
        role, team = self.role_and_team(stable_id)
        if role == "referee":
            return (0, 255, 255)   # bright yellow outline - actual jersey is near-black
                                    # and wouldn't be visible as a box color on the video
        if team == "Team A" and self.team_clusters:
            return self.cluster_bgr.get(self.team_clusters[0], (0, 255, 0))
        if team == "Team B" and len(self.team_clusters) > 1:
            return self.cluster_bgr.get(self.team_clusters[1], (255, 0, 0))
        return (0, 255, 0)


# ==================================================
# Ball Trajectory  (full-video path, interpolated through gaps)
# ==================================================
# The ball is small, fast, and motion-blurs when kicked hard, so raw
# detection has gaps ("flying" frames with no detection at all). We
# collect every frame where the ball WAS detected, then linearly
# interpolate the missing frames in between (up to MAX_BALL_GAP_FRAMES -
# beyond that the ball is treated as genuinely out of frame, not just
# blurred, and the gap is left undrawn rather than guessed at).
# ==================================================
class BallTrajectory:
    def __init__(self, max_gap_frames=45):
        self.max_gap_frames = max_gap_frames
        self.raw_points = {}   # frame_number -> (cx, cy)
        self.full_path = {}    # frame_number -> (cx, cy), filled in after interpolate()

    def add_detection(self, frame_number, center, confidence):
        # only one ball on the pitch - if two are detected in one frame (rare double
        # detection), keep the higher-confidence one
        existing = self.raw_points.get(frame_number)
        if existing is None or confidence > existing[2]:
            self.raw_points[frame_number] = (center[0], center[1], confidence)

    def interpolate(self):
        frames = sorted(self.raw_points.keys())
        self.full_path = {}
        if not frames:
            return

        for f in frames:
            x, y, _ = self.raw_points[f]
            self.full_path[f] = (x, y)

        for prev_f, next_f in zip(frames, frames[1:]):
            gap = next_f - prev_f
            if gap <= 1 or gap > self.max_gap_frames:
                continue   # too big a gap - ball was likely off-screen, don't guess
            x1, y1, _ = self.raw_points[prev_f]
            x2, y2, _ = self.raw_points[next_f]
            for step in range(1, gap):
                t = step / gap
                self.full_path[prev_f + step] = (
                    int(round(x1 + (x2 - x1) * t)),
                    int(round(y1 + (y2 - y1) * t)),
                )

    def path_up_to(self, frame_number, tail=None):
        """Ordered list of (x, y) points from the start of the match through
        frame_number, in frame order (gaps beyond max_gap_frames create separate
        unconnected segments, which is desired - no false lines across real gaps)."""
        pts = [self.full_path[f] for f in sorted(self.full_path.keys()) if f <= frame_number]
        if tail:
            pts = pts[-tail:]
        return pts

    def detection_rate(self, total_frames):
        return round(100 * len(self.raw_points) / max(total_frames, 1), 2)

    def interpolated_frame_count(self):
        return len(self.full_path) - len(self.raw_points)


# ==================================================
# Sequential label assignment
# ==================================================
def assign_sequential_labels(stabilizer, classifier):
    """Player 1, Player 2, ... in order of first appearance (global, across both
    teams); Referee 1, Referee 2, ... in its own counter."""
    ordered_ids = sorted(stabilizer.first_seen_frame.keys(),
                          key=lambda sid: stabilizer.first_seen_frame[sid])
    labels = {}
    player_counter = 0
    referee_counter = 0
    for sid in ordered_ids:
        role, team = classifier.role_and_team(sid)
        if role == "referee":
            referee_counter += 1
            labels[sid] = f"Referee {referee_counter}"
        else:
            player_counter += 1
            team_suffix = f" ({team})" if team else ""
            labels[sid] = f"Player {player_counter}{team_suffix}"
    return labels


# ==================================================
# Load Model
# ==================================================
print("Loading model...")
model = YOLO(MODEL_PATH)
print("CUDA detected - using half-precision (fp16) inference." if USE_HALF_PRECISION
      else "Running on CPU - lower IMG_SIZE first if you need more speed.")
print("Model loaded successfully!\n")

stabilizer = IDStabilizer(MAX_LOST_FRAMES, MAX_MATCH_DISTANCE, APPEARANCE_THRESHOLD)
classifier = TeamClassifier(k=N_COLOR_CLUSTERS, min_samples=MIN_SAMPLES_TO_CLUSTER)
ball_trajectory = BallTrajectory(max_gap_frames=MAX_BALL_GAP_FRAMES)

cap = cv2.VideoCapture(INPUT_VIDEO)
if not cap.isOpened():
    print("Error: Could not open input video.")
    exit()

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames_hint = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"Video Width  : {width}")
print(f"Video Height : {height}")
print(f"FPS          : {fps}")
print("-" * 50)

# ==================================================
# PASS 1 - detect, track, collect samples (no drawing/writing)
# ==================================================
print("[Pass 1/2] Running detection + tracking...")
frame_number = 0
inference_times = []
# per-frame raw results, kept lightweight (ints/floats only) so long videos don't
# blow up memory - the actual frame pixels are re-read from disk in pass 2
per_frame_objects = []   # list of lists: [{stable_id, class, bbox, conf}, ...]

while True:
    ret, frame = cap.read()
    if not ret:
        break

    t0 = time.time()
    results = model.track(
        frame, persist=True, tracker="bytetrack.yaml",
        conf=BALL_CONF, imgsz=IMG_SIZE, device=DEVICE,
        half=USE_HALF_PRECISION, verbose=False
    )
    inference_times.append(time.time() - t0)

    frame_objects = []
    boxes = results[0].boxes

    if boxes.id is not None:
        raw_ids = boxes.id.int().cpu().tolist()
        xyxy = boxes.xyxy.cpu().tolist()
        cls = boxes.cls.int().cpu().tolist()
        conf = boxes.conf.cpu().tolist()

        kept_raw_ids, kept_bboxes, kept_names, kept_conf = [], [], [], []
        for raw_id, box, c, score in zip(raw_ids, xyxy, cls, conf):
            object_name = model.names[c]
            threshold = BALL_CONF if object_name == BALL_CLASS_NAME else PERSON_CONF
            if score < threshold:
                continue
            kept_raw_ids.append(raw_id)
            kept_bboxes.append([int(v) for v in box])
            kept_conf.append(score)
            kept_names.append(object_name)

        stable_ids = stabilizer.update(frame, frame_number, kept_raw_ids, kept_bboxes)

        for stable_id, bbox, object_name, score in zip(stable_ids, kept_bboxes, kept_names, kept_conf):
            if object_name == BALL_CLASS_NAME:
                cx, cy = (bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2
                ball_trajectory.add_detection(frame_number, (cx, cy), score)
            else:
                classifier.add_sample(frame, bbox, stable_id)

            frame_objects.append({
                "stable_id": stable_id, "class": object_name,
                "bbox": bbox, "confidence": round(score, 3)
            })

    per_frame_objects.append(frame_objects)
    frame_number += 1
    if frame_number % 200 == 0:
        print(f"  ...{frame_number} frames analyzed")

cap.release()
total_frames = frame_number
print(f"[Pass 1/2] Done - {total_frames} frames analyzed.\n")

# ==================================================
# Finalize - team clustering + ball interpolation + labels
# ==================================================
print("[Finalize] Clustering jerseys, interpolating ball path, assigning labels...")
classifier.finalize()
ball_trajectory.interpolate()
sequential_labels = assign_sequential_labels(stabilizer, classifier)
print(f"[Finalize] Ball: {len(ball_trajectory.raw_points)} raw detections + "
      f"{ball_trajectory.interpolated_frame_count()} interpolated frames "
      f"= {len(ball_trajectory.full_path)} total trajectory points.\n")

# ==================================================
# PASS 2 - render annotated video
# ==================================================
print("[Pass 2/2] Rendering annotated video...")
cap = cv2.VideoCapture(INPUT_VIDEO)
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter("outputs/tracked_video.mp4", fourcc, fps, (width, height))
if not out.isOpened():
    print("Error: Could not create output video.")
    exit()

BALL_COLOR = (0, 165, 255)
CENTER_COLOR = (0, 0, 255)
TEXT_COLOR = (255, 255, 255)
TRAJECTORY_COLOR = (0, 140, 255)

tracking_data = []
unique_stable_ids = set()

frame_number = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    annotated = frame.copy()
    frame_info = {"frame": frame_number, "objects": []}

    # full accumulated ball trajectory, drawn every frame
    trail = ball_trajectory.path_up_to(frame_number, tail=TRAJECTORY_FADE_TAIL)
    for i in range(1, len(trail)):
        cv2.line(annotated, trail[i - 1], trail[i], TRAJECTORY_COLOR, TRAJECTORY_THICKNESS)
    if trail:
        cv2.circle(annotated, trail[-1], 5, BALL_COLOR, -1)

    for obj in per_frame_objects[frame_number]:
        stable_id = obj["stable_id"]
        x1, y1, x2, y2 = obj["bbox"]
        object_name = obj["class"]
        score = obj["confidence"]
        center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

        if object_name == BALL_CLASS_NAME:
            role, team, label = "ball", None, "Ball"
            color = BALL_COLOR
        else:
            unique_stable_ids.add(stable_id)
            role, team = classifier.role_and_team(stable_id)
            label = sequential_labels.get(stable_id, f"ID {stable_id}")
            color = classifier.color_for(stable_id)

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        cv2.circle(annotated, (center_x, center_y), 4, CENTER_COLOR, -1)
        cv2.putText(annotated, f"{label} | {score:.2f}", (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_COLOR, 2)

        frame_info["objects"].append({
            "label": label, "role": role, "team": team,
            "track_id": stable_id, "bbox": [x1, y1, x2, y2],
            "center": [center_x, center_y], "confidence": score
        })

    tracking_data.append(frame_info)
    out.write(annotated)
    frame_number += 1

cap.release()
out.release()
print("[Pass 2/2] Done.\n")

# ==================================================
# Tracking Summary
# ==================================================
avg_inference_time = sum(inference_times) / max(len(inference_times), 1)
processing_fps = 1.0 / avg_inference_time if avg_inference_time > 0 else 0.0

team_counts = Counter(t for t in classifier.track_team.values() if t)
referee_count = sum(1 for r in classifier.track_role.values() if r == "referee")

summary = {
    "project": "SmartMatch AI",
    "video_name": os.path.basename(INPUT_VIDEO),
    "detector": "YOLO11s",
    "tracker": "ByteTrack + ID Stabilizer",
    "classifier": "Jersey-color k-means, whole-video majority vote (darkest cluster = referee)",
    "person_confidence_threshold": PERSON_CONF,
    "ball_confidence_threshold": BALL_CONF,
    "inference_imgsz": IMG_SIZE,
    "device": "GPU" if CUDA_AVAILABLE else "CPU",
    "half_precision": USE_HALF_PRECISION,
    "video_fps": round(fps, 2),
    "resolution": {"width": width, "height": height},
    "total_frames": total_frames,
    "total_unique_stable_ids": len(unique_stable_ids),
    "team_composition": dict(team_counts),
    "referee_count": referee_count,
    "ball_trajectory": {
        "raw_detected_frames": len(ball_trajectory.raw_points),
        "interpolated_frames": ball_trajectory.interpolated_frame_count(),
        "total_trajectory_points": len(ball_trajectory.full_path),
        "raw_detection_rate_pct": ball_trajectory.detection_rate(total_frames),
        "max_bridged_gap_frames": MAX_BALL_GAP_FRAMES
    },
    "tracking_accuracy": stabilizer.accuracy_report(),
    "performance": {
        "avg_inference_time_sec": round(avg_inference_time, 4),
        "avg_processing_fps": round(processing_fps, 2),
        "real_time_capable": processing_fps >= fps
    },
    "tracking_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
}

with open("outputs/detections.json", "w") as f:
    json.dump(tracking_data, f, indent=4)

with open("outputs/tracking_summary.json", "w") as f:
    json.dump(summary, f, indent=4)

print("=" * 50)
print("Tracking Completed Successfully!")
print("=" * 50)
print(f"Frames Processed      : {total_frames}")
print(f"Unique Player/Ref IDs : {len(unique_stable_ids)}  "
      f"(Team A: {team_counts.get('Team A', 0)}, Team B: {team_counts.get('Team B', 0)}, "
      f"Referees: {referee_count})")
print(f"Ball trajectory       : {len(ball_trajectory.raw_points)} detected + "
      f"{ball_trajectory.interpolated_frame_count()} interpolated "
      f"({summary['ball_trajectory']['raw_detection_rate_pct']}% raw detection rate)")
print(f"Avg track length      : {summary['tracking_accuracy']['avg_track_length_frames']} frames")
print(f"ID recoveries         : {summary['tracking_accuracy']['successful_id_recoveries']}")
print(f"Processing speed      : {processing_fps:.2f} FPS "
      f"(video is {fps:.2f} FPS -> "
      f"{'real-time capable' if processing_fps >= fps else 'slower than real-time'})")
print("\nGenerated Files:")
print("  outputs/tracked_video.mp4")
print("  outputs/detections.json")
print("  outputs/tracking_summary.json")