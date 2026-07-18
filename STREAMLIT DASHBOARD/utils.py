"""
utils.py
--------
Shared utility functions used across the dashboard:
 - Data loaders (JSON, image, video)
 - Formatters (speed, distance, time)
 - File-existence helpers
 - Pitch coordinate converters
"""

import json
import base64
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st

from config import (
    DETECTIONS_JSON,
    ANALYTICS_JSON,
    BLUE_TEAM_HEATMAP,
    RED_TEAM_HEATMAP,
    BALL_HEATMAP,
    TRACKED_VIDEO,
    LOGO_PATH,
    CUSTOM_CSS,
    PITCH_LENGTH,
    PITCH_WIDTH,
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS LOADER
# ─────────────────────────────────────────────────────────────────────────────

def load_css(path: Path = CUSTOM_CSS) -> None:
    """Inject custom CSS into Streamlit app."""
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADERS
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=60)
def load_detections() -> Optional[List[Dict]]:
    """Load detections.json. Returns None if missing / malformed."""
    return _load_json(DETECTIONS_JSON)


@st.cache_data(ttl=60)
def load_analytics() -> Optional[Dict]:
    """Load analytics.json. Returns None if missing / malformed."""
    return _load_json(ANALYTICS_JSON)


def _load_json(path: Path) -> Optional[Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return None
    except json.JSONDecodeError:
        st.error(f"⚠️ JSON parse error in {path.name}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# IMAGE HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def load_image(path: Path) -> Optional[Image.Image]:
    """Return PIL Image or None if file is missing."""
    if path.exists():
        return Image.open(path)
    return None


def image_to_base64(path: Path) -> Optional[str]:
    """Encode an image file to base64 for inline HTML embedding."""
    if path.exists():
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    return None


def logo_html(path: Path = LOGO_PATH, height: int = 48) -> str:
    """Return an <img> HTML tag for the logo, or text fallback."""
    b64 = image_to_base64(path)
    if b64:
        return f'<img src="data:image/png;base64,{b64}" height="{height}" style="margin-right:10px;">'
    return "⚽"


# ─────────────────────────────────────────────────────────────────────────────
# ANALYTICS → DATAFRAME
# ─────────────────────────────────────────────────────────────────────────────

def analytics_to_dataframe(analytics: Dict) -> pd.DataFrame:
    """Convert analytics.json player list into a tidy DataFrame."""
    players = analytics.get("players", [])
    if not players:
        return pd.DataFrame()
    df = pd.DataFrame(players)
    numeric_cols = ["distance_m", "avg_speed_kmh", "max_speed_kmh", "touches", "possession_pct"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").round(2)
    return df


def detections_to_dataframe(detections: List[Dict]) -> pd.DataFrame:
    """Flatten detections list into a DataFrame (one row per detection)."""
    if not detections:
        return pd.DataFrame()
    return pd.DataFrame(detections)


# ─────────────────────────────────────────────────────────────────────────────
# FORMATTERS
# ─────────────────────────────────────────────────────────────────────────────

def fmt_speed(value: float) -> str:
    return f"{value:.1f} km/h"


def fmt_distance(value: float) -> str:
    if value >= 1000:
        return f"{value / 1000:.2f} km"
    return f"{value:.0f} m"


def fmt_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"


def fmt_pct(value: float) -> str:
    return f"{value:.1f}%"


# ─────────────────────────────────────────────────────────────────────────────
# PITCH COORDINATE HELPER
# ─────────────────────────────────────────────────────────────────────────────

def normalize_pitch_coords(
    x: float,
    y: float,
    frame_w: int = 1920,
    frame_h: int = 1080,
) -> Tuple[float, float]:
    """
    Convert pixel coordinates to pitch-relative metres.
    Assumes camera covers full pitch width × height.
    """
    px = (x / frame_w) * PITCH_LENGTH
    py = (y / frame_h) * PITCH_WIDTH
    return px, py


# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY STATISTICS
# ─────────────────────────────────────────────────────────────────────────────

def compute_summary(analytics: Dict) -> Dict:
    """Derive top-level KPIs from the analytics payload."""
    players = analytics.get("players", [])
    team_stats = analytics.get("team_stats", {})
    match_info = analytics.get("match_info", {})

    total_players = len(players)
    df = pd.DataFrame(players) if players else pd.DataFrame()

    most_active = "—"
    top_distance = 0.0
    top_speed = 0.0

    if not df.empty:
        if "distance_m" in df.columns:
            idx = df["distance_m"].idxmax()
            most_active = f"Player {df.loc[idx, 'player_id']}"
            top_distance = df["distance_m"].max()
        if "max_speed_kmh" in df.columns:
            top_speed = df["max_speed_kmh"].max()

    team1_poss = team_stats.get("Team 1", {}).get("possession_pct", 50.0)
    match_duration = match_info.get("duration_seconds", 0)

    return {
        "total_players": total_players,
        "most_active_player": most_active,
        "top_distance": top_distance,
        "top_speed": top_speed,
        "team1_possession": team1_poss,
        "team2_possession": 100.0 - team1_poss,
        "match_duration": match_duration,
    }


# ─────────────────────────────────────────────────────────────────────────────
# FILE-EXISTENCE GUARD
# ─────────────────────────────────────────────────────────────────────────────

def check_file(path: Path, label: str) -> bool:
    """Return True if file exists; display a styled warning card if not."""
    if path.exists():
        return True
    st.markdown(
        f"""
        <div class="warning-card">
            <span class="warning-icon">⚠️</span>
            <div>
                <strong>{label}</strong> not found.<br>
                <small style="color:#94A3B8;">Expected path: <code>{path}</code></small>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    return False
# ─────────────────────────────────────────────────────────────────────────────
# FILE SAVE HELPER
# ─────────────────────────────────────────────────────────────────────────────

def save_uploaded_file(uploaded_file, destination: Path) -> bool:
    """
    Save a Streamlit uploaded file to disk.
    Returns True on success, False otherwise.
    """
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)

        with open(destination, "wb") as f:
            f.write(uploaded_file.getbuffer())

        return True

    except Exception as e:
        st.error(f"❌ Failed to save uploaded file: {e}")
        return False