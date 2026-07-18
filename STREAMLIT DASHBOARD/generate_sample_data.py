"""
generate_sample_data.py
-----------------------
Run once to generate realistic sample detections.json and analytics.json
in the data/ directory. Also creates placeholder heatmap images.

Usage: python generate_sample_data.py
"""

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import json
import random
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

# ── Paths ────────────────────────────────────────────────────────────────────
BASE = Path(__file__).parent
DATA_DIR    = BASE / "data"
OUTPUTS_DIR = BASE / "outputs"
ASSETS_DIR  = BASE / "assets"

DATA_DIR.mkdir(exist_ok=True)
OUTPUTS_DIR.mkdir(exist_ok=True)
ASSETS_DIR.mkdir(exist_ok=True)

random.seed(42)
np.random.seed(42)

# ── Config ───────────────────────────────────────────────────────────────────
PITCH_W, PITCH_H = 1920, 1080      # pixel dims of video
PITCH_LEN = 105.0                  # metres
PITCH_WID = 68.0

TOTAL_FRAMES = 5400                # 3 min at 30fps (demo)
FPS = 30
N_BLUE = 11
N_RED  = 11

# Player ID ranges
BLUE_IDS = list(range(1, 12))
RED_IDS  = list(range(12, 23))
REF_ID   = 23

# ─────────────────────────────────────────────────────────────────────────────
# 1. Generate player motion trails
# ─────────────────────────────────────────────────────────────────────────────

def make_trail(x0: float, y0: float, n: int = TOTAL_FRAMES) -> list:
    """Brownian-motion trail clamped to pitch."""
    x, y = x0, y0
    trail = []
    for _ in range(n):
        x = float(np.clip(x + np.random.randn() * 1.5, 5, PITCH_LEN - 5))
        y = float(np.clip(y + np.random.randn() * 1.0, 3, PITCH_WID - 3))
        trail.append({"x": round(x, 2), "y": round(y, 2)})
    return trail


# Assign home positions
team1_homes = [
    (10, 34), (20, 10), (20, 25), (20, 43), (20, 58),
    (35, 15), (35, 34), (35, 53), (50, 20), (50, 48), (50, 34),
]
team2_homes = [
    (95, 34), (85, 10), (85, 25), (85, 43), (85, 58),
    (70, 15), (70, 34), (70, 53), (55, 20), (55, 48), (55, 34),
]

print("Generating player trails …")
team1_trails = {pid: make_trail(*pos, TOTAL_FRAMES) for pid, pos in zip(BLUE_IDS, team1_homes)}
team2_trails  = {pid: make_trail(*pos, TOTAL_FRAMES) for pid, pos in zip(RED_IDS,  team2_homes)}

# Ball trail (follows players roughly)
ball_trail = make_trail(52, 34, TOTAL_FRAMES)

# ─────────────────────────────────────────────────────────────────────────────
# 2. detections.json  (sample: every 5th frame for file size)
# ─────────────────────────────────────────────────────────────────────────────
print("Building detections.json …")

SAMPLE_FRAMES = list(range(0, TOTAL_FRAMES, 5))
detections = []

def pitch_to_pixel(px: float, py: float):
    return (px / PITCH_LEN) * PITCH_W, (py / PITCH_WID) * PITCH_H

for fi in SAMPLE_FRAMES:
    for pid in BLUE_IDS:
        t = team1_trails[pid][fi]
        px, py = pitch_to_pixel(t["x"], t["y"])
        bw, bh = random.randint(50, 80), random.randint(100, 160)
        detections.append({
            "frame":     fi,
            "player_id": pid,
            "class":     "player",
            "team":      "Team 1",
            "bbox":      [round(px - bw/2), round(py - bh/2),
                          round(px + bw/2), round(py + bh/2)],
            "confidence": round(random.uniform(0.82, 0.99), 3),
            "x":         round(px, 1),
            "y":         round(py, 1),
            "x_pitch":   round(t["x"], 2),
            "y_pitch":   round(t["y"], 2),
        })
    for pid in RED_IDS:
        t = team2_trails[pid][fi]
        px, py = pitch_to_pixel(t["x"], t["y"])
        bw, bh = random.randint(50, 80), random.randint(100, 160)
        detections.append({
            "frame":     fi,
            "player_id": pid,
            "class":     "player",
            "team":      "Team 2",
            "bbox":      [round(px - bw/2), round(py - bh/2),
                          round(px + bw/2), round(py + bh/2)],
            "confidence": round(random.uniform(0.80, 0.99), 3),
            "x":         round(px, 1),
            "y":         round(py, 1),
            "x_pitch":   round(t["x"], 2),
            "y_pitch":   round(t["y"], 2),
        })
    # Ball
    bt = ball_trail[fi]
    bpx, bpy = pitch_to_pixel(bt["x"], bt["y"])
    detections.append({
        "frame":     fi,
        "player_id": 0,
        "class":     "ball",
        "team":      "None",
        "bbox":      [round(bpx-15), round(bpy-15), round(bpx+15), round(bpy+15)],
        "confidence": round(random.uniform(0.75, 0.98), 3),
        "x":         round(bpx, 1),
        "y":         round(bpy, 1),
        "x_pitch":   round(bt["x"], 2),
        "y_pitch":   round(bt["y"], 2),
    })
    # Referee
    rt = team1_trails[BLUE_IDS[0]][fi]  # Ref moves near centre
    rpx, rpy = pitch_to_pixel(52 + np.random.randn()*3, 34 + np.random.randn()*3)
    detections.append({
        "frame":     fi,
        "player_id": REF_ID,
        "class":     "referee",
        "team":      "Referee",
        "bbox":      [round(rpx-30), round(rpy-70), round(rpx+30), round(rpy+70)],
        "confidence": round(random.uniform(0.85, 0.99), 3),
        "x":         round(rpx, 1),
        "y":         round(rpy, 1),
        "x_pitch":   round(52.0 + random.gauss(0, 3), 2),
        "y_pitch":   round(34.0 + random.gauss(0, 3), 2),
    })

with open(DATA_DIR / "detections.json", "w") as f:
    json.dump(detections, f, indent=2)
print(f"  -> {len(detections):,} detection records written.")

# ─────────────────────────────────────────────────────────────────────────────
# 3. analytics.json
# ─────────────────────────────────────────────────────────────────────────────
print("Building analytics.json …")

def compute_distance(trail):
    total = 0.0
    for i in range(1, len(trail)):
        dx = trail[i]["x"] - trail[i-1]["x"]
        dy = trail[i]["y"] - trail[i-1]["y"]
        total += math.sqrt(dx*dx + dy*dy)
    return total * 1.05   # scale to realistic metres

players_data = []
for pid, home, trail in [(p, h, team1_trails[p]) for p, h in zip(BLUE_IDS, team1_homes)]:
    dist    = compute_distance(trail)
    speeds  = [random.uniform(6, 32) for _ in range(50)]
    avg_spd = float(np.mean(speeds))
    max_spd = float(max(speeds))
    players_data.append({
        "player_id":      pid,
        "team":           "Team 1",
        "distance_m":     round(dist, 2),
        "avg_speed_kmh":  round(avg_spd, 2),
        "max_speed_kmh":  round(max_spd, 2),
        "touches":        random.randint(15, 90),
        "possession_pct": round(random.uniform(2, 18), 2),
        "trail":          trail[::60],   # 1 point per 2 seconds for storage
    })

for pid, home, trail in [(p, h, team2_trails[p]) for p, h in zip(RED_IDS, team2_homes)]:
    dist    = compute_distance(trail)
    speeds  = [random.uniform(6, 30) for _ in range(50)]
    avg_spd = float(np.mean(speeds))
    max_spd = float(max(speeds))
    players_data.append({
        "player_id":      pid,
        "team":           "Team 2",
        "distance_m":     round(dist, 2),
        "avg_speed_kmh":  round(avg_spd, 2),
        "max_speed_kmh":  round(max_spd, 2),
        "touches":        random.randint(12, 80),
        "possession_pct": round(random.uniform(2, 15), 2),
        "trail":          trail[::60],
    })

# Team aggregates
team1_p = [p for p in players_data if p["team"] == "Team 1"]
team2_p  = [p for p in players_data if p["team"] == "Team 2"]

def team_stats(players, poss_pct):
    return {
        "player_count":       len(players),
        "possession_pct":     round(poss_pct, 2),
        "total_distance_m":   round(sum(p["distance_m"] for p in players), 2),
        "avg_speed_kmh":      round(sum(p["avg_speed_kmh"] for p in players) / len(players), 2),
        "max_speed_kmh":      round(max(p["max_speed_kmh"] for p in players), 2),
        "avg_touches":        round(sum(p["touches"] for p in players) / len(players), 1),
    }

team1_poss = 54.3
team2_poss  = 45.7

analytics = {
    "match_info": {
        "duration_seconds": TOTAL_FRAMES / FPS,
        "total_frames":     TOTAL_FRAMES,
        "fps":              FPS,
        "video_resolution": f"{PITCH_W}x{PITCH_H}",
        "model":            "YOLOv8x + ByteTrack",
    },
    "team_stats": {
        "Team 1": team_stats(team1_p, team1_poss),
        "Team 2":  team_stats(team2_p,  team2_poss),
    },
    "players": players_data,
    "insights": [
        {
            "text": "🎯 <strong>Team 1</strong> dominated ball possession with <strong>54.3%</strong> — "
                    "indicating superior positional control and build-up play.",
            "variant": "default",
        },
        {
            "text": "⚡ Player 7 (Team 1) recorded the highest sprint speed of <strong>31.2 km/h</strong> "
                    "during a counter-attack sequence in the second half.",
            "variant": "info",
        },
        {
            "text": "📏 Player 4 (Team 1) covered the most ground at <strong>11.4 km</strong>, "
                    "reflecting their high work rate and pressing responsibility.",
            "variant": "default",
        },
        {
            "text": "🔴 Team 2 showed a compact defensive shape with lower average speed "
                    "but higher intensity in the final third.",
            "variant": "warning",
        },
        {
            "text": "⚽ Player 8 had the most ball contacts (<strong>87 touches</strong>), "
                    "serving as the primary ball-playing midfielder.",
            "variant": "info",
        },
    ],
}

with open(DATA_DIR / "analytics.json", "w") as f:
    json.dump(analytics, f, indent=2)
print("  -> analytics.json written.")

# ─────────────────────────────────────────────────────────────────────────────
# 4. Heatmap Images (synthetic KDE-style)
# ─────────────────────────────────────────────────────────────────────────────
print("Generating heatmap images …")

def make_heatmap(trails_list, width=1050, height=680, title=""):
    """Create a synthetic KDE-style heatmap as a PIL image."""
    hmap = np.zeros((height, width), dtype=np.float32)
    for trail in trails_list:
        for pt in trail[::10]:
            px = int(np.clip((pt["x"] / PITCH_LEN) * width,  0, width  - 1))
            py = int(np.clip((pt["y"] / PITCH_WID) * height, 0, height - 1))
            # Gaussian splat
            r = 30
            x0, x1 = max(0, px - r), min(width,  px + r)
            y0, y1 = max(0, py - r), min(height, py + r)
            for xi in range(x0, x1):
                for yi in range(y0, y1):
                    d2 = (xi - px)**2 + (yi - py)**2
                    if d2 < r*r:
                        hmap[yi, xi] += math.exp(-d2 / (2 * (r/2)**2))

    # Normalise to 0-255
    mx = hmap.max()
    if mx > 0:
        hmap = hmap / mx
    heat_u8 = (hmap * 255).astype(np.uint8)

    # Colourize: black -> green -> yellow -> red
    img = Image.fromarray(heat_u8, mode="L").convert("RGB")
    arr = np.array(img, dtype=np.float32) / 255.0
    r_ch = np.clip(arr[:,:,0] * 2.0, 0, 1)
    g_ch = np.clip(1 - abs(arr[:,:,0] * 2 - 1), 0, 1)
    b_ch = np.zeros_like(r_ch)
    rgb = (np.stack([r_ch, g_ch, b_ch], axis=2) * 255).astype(np.uint8)
    result = Image.fromarray(rgb)
    result = result.filter(ImageFilter.GaussianBlur(8))
    # Dark background
    bg = Image.new("RGB", (width, height), (10, 14, 26))
    bg.paste(result, (0, 0), mask=Image.fromarray(heat_u8).filter(ImageFilter.GaussianBlur(4)))
    return bg

all_trails = list(team1_trails.values()) + list(team2_trails.values())
player_hm = make_heatmap(all_trails)
player_hm.save(OUTPUTS_DIR / "player_heatmap.png")
print("  -> player_heatmap.png written.")

ball_hm = make_heatmap([ball_trail])
ball_hm.save(OUTPUTS_DIR / "ball_heatmap.png")
print("  -> ball_heatmap.png written.")

# ─────────────────────────────────────────────────────────────────────────────
# 5. Logo placeholder
# ─────────────────────────────────────────────────────────────────────────────
print("Generating logo …")

def make_logo(path, size=128):
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    # Circle background
    draw.ellipse([4, 4, size-4, size-4], fill=(26, 35, 50, 230))
    draw.ellipse([4, 4, size-4, size-4], outline=(0, 255, 135, 200), width=3)
    # Football symbol (text)
    draw.text((size//2 - 18, size//2 - 22), "⚽", fill=(255, 255, 255, 230))
    img.save(path)

try:
    make_logo(ASSETS_DIR / "logo.png")
    print("  -> logo.png written.")
except Exception as e:
    # Some PIL builds don't support emoji in text — create a simpler logo
    img = Image.new("RGB", (128, 128), (26, 35, 50))
    draw = ImageDraw.Draw(img)
    draw.ellipse([4, 4, 124, 124], fill=(0, 30, 20), outline=(0, 255, 135), width=3)
    draw.rectangle([54, 44, 74, 84], fill=(0, 255, 135))
    draw.rectangle([44, 54, 84, 74], fill=(0, 255, 135))
    img.save(ASSETS_DIR / "logo.png")
    print(f"  -> logo.png written (simple fallback, emoji error: {e})")

print("\nOK All sample data generated successfully!")
print(f"   data/detections.json ({(DATA_DIR / 'detections.json').stat().st_size // 1024} KB)")
print(f"   data/analytics.json  ({(DATA_DIR / 'analytics.json').stat().st_size // 1024} KB)")
print(f"   outputs/player_heatmap.png")
print(f"   outputs/ball_heatmap.png")
print(f"   assets/logo.png")
