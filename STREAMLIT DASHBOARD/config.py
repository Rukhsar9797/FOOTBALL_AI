"""
config.py
---------
Central configuration for the Football Analytics Dashboard.
Defines paths, color palette, theme settings, and global constants.
"""

import os
from pathlib import Path

# ─────────────────────────────────────────────
# BASE PATHS
# ─────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
OUTPUTS_DIR = BASE_DIR / "outputs"
ASSETS_DIR = BASE_DIR / "assets"
STYLES_DIR = BASE_DIR / "styles"

# ─────────────────────────────────────────────
# FILE PATHS
# ─────────────────────────────────────────────
DETECTIONS_JSON = DATA_DIR / "detections.json"
ANALYTICS_JSON = DATA_DIR / "analytics.json"
TRACKED_VIDEO = OUTPUTS_DIR / "tracked_video.mp4"
ORIGINAL_VIDEO = DATA_DIR / "original_video.mp4"
COMPARISON_VIDEO = OUTPUTS_DIR / "comparison.mp4"
# Heatmaps
BLUE_TEAM_HEATMAP = OUTPUTS_DIR / "blue_team_heatmap.png"
RED_TEAM_HEATMAP  = OUTPUTS_DIR / "red_team_heatmap.png"
BALL_HEATMAP      = OUTPUTS_DIR / "ball_heatmap.png"

# Optional: keep this for backward compatibility
PLAYER_HEATMAP = BLUE_TEAM_HEATMAP
LOGO_PATH = ASSETS_DIR / "logo.png"
BACKGROUND_PATH = ASSETS_DIR / "football_background.png"
CUSTOM_CSS = STYLES_DIR / "custom.css"

# ─────────────────────────────────────────────
# APP SETTINGS
# ─────────────────────────────────────────────
APP_TITLE = "FootballIQ Analytics"
APP_SUBTITLE = "AI-Powered Football Match Intelligence"
APP_ICON = "⚽"
APP_VERSION = "1.0.0"
HACKATHON_NAME = "24-Hour Computer Vision Hackathon"

# ─────────────────────────────────────────────
# TEAM CONFIG
# ─────────────────────────────────────────────
TEAM_COLORS = {
    "Team 1": "#EF4444",
    "Team 2": "#FFFFFF",
    "Referee":   "#F59E0B",
    "Unknown":   "#6B7280",
}

TEAM_ACCENT = {
    "Team 1": "#F87171",
    "Team 2": "#E2E8F0",
}

# ─────────────────────────────────────────────
# DARK THEME PALETTE
# ─────────────────────────────────────────────
COLORS = {
    # Backgrounds
    "bg_primary":    "#0A0E1A",
    "bg_secondary":  "#111827",
    "bg_card":       "#1A2332",
    "bg_card_hover": "#1F2B3E",
    "bg_sidebar":    "#0D1321",

    # Accents / Brand
    "accent_green":  "#00FF87",
    "accent_blue":   "#3B82F6",
    "accent_gold":   "#F59E0B",
    "accent_teal":   "#14B8A6",
    "accent_purple": "#8B5CF6",
    "accent_red":    "#EF4444",

    # Text
    "text_primary":   "#F1F5F9",
    "text_secondary": "#94A3B8",
    "text_muted":     "#475569",

    # Borders
    "border":         "#1E293B",
    "border_accent":  "#334155",
}

# ─────────────────────────────────────────────
# CHART DEFAULTS
# ─────────────────────────────────────────────
PLOTLY_TEMPLATE = "plotly_dark"
CHART_BG = "rgba(17, 24, 39, 0.0)"
CHART_PAPER_BG = "rgba(17, 24, 39, 0.0)"
CHART_FONT_COLOR = "#94A3B8"
CHART_GRID_COLOR = "rgba(51, 65, 85, 0.5)"

# ─────────────────────────────────────────────
# VIDEO SETTINGS
# ─────────────────────────────────────────────
VIDEO_FPS = 25
FRAME_STEP = 1

# ─────────────────────────────────────────────
# PITCH DIMENSIONS (standard in meters)
# ─────────────────────────────────────────────
PITCH_LENGTH = 105.0
PITCH_WIDTH  = 68.0
