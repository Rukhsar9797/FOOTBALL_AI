"""
app.py
------
FootballIQ Analytics Dashboard — Entry Point
Configures Streamlit page, loads CSS, and routes to page modules.

Run:
    streamlit run app.py
"""

import sys
from pathlib import Path

# ── Make project root importable ────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st

from config import APP_TITLE, APP_ICON, APP_SUBTITLE, DETECTIONS_JSON, ANALYTICS_JSON, TRACKED_VIDEO
from utils import load_css
from components.sidebar import render_sidebar

# ── Dashboard Pages ──────────────────────────────────────────────────────────
from dashboard.home           import render_home
from dashboard.live_match     import render_live_match
from dashboard.team_analysis  import render_team_analysis
from dashboard.player_analysis import render_player_analysis
from dashboard.heatmaps       import render_heatmaps
from dashboard.insights       import render_insights
from dashboard.downloads      import render_downloads
from dashboard.about          import render_about

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": None,
        "Report a bug": None,
        "About": f"**{APP_TITLE}** — {APP_SUBTITLE}",
    },
)

# ── Inject Custom CSS ────────────────────────────────────────────────────────
#load_css()

# ─────────────────────────────────────────────────────────────────────────────
# ROUTING
# ─────────────────────────────────────────────────────────────────────────────
PAGE_REGISTRY = {
    "Home":            render_home,
    "Live Match":      render_live_match,
    "Team Analysis":   render_team_analysis,
    "Player Analysis": render_player_analysis,
    "Heatmaps":        render_heatmaps,
    "Match Insights":  render_insights,
    "Downloads":       render_downloads,
    "About Project":   render_about,
}

# ── Render Sidebar & Get Selected Page ──────────────────────────────────────
selected_page, is_live_mode = render_sidebar()

# Store mode in session state if needed globally
st.session_state["live_mode"] = is_live_mode

if (
    selected_page != "Home"
    and not TRACKED_VIDEO.exists()
):
    st.warning("⚠️ Please upload the football video and generate the tracked video first.")
    st.stop()

# ── Render Selected Page ─────────────────────────────────────────────────────
render_fn = PAGE_REGISTRY.get(selected_page, render_home)
render_fn()

