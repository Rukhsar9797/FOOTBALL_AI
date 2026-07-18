"""
dashboard/downloads.py
----------------------
Downloads page: styled download cards for each output file.
"""

import json
import streamlit as st

from components.download import render_download_card
from components.metrics import section_header
from config import TRACKED_VIDEO, ANALYTICS_JSON, DETECTIONS_JSON


def render_downloads() -> None:
    """Render the Downloads page."""
    st.markdown(
        """
        <div class="page-header">
            <div class="page-title">⬇️ Downloads</div>
            <div class="page-subtitle">Export tracked video, analytics data, and raw detections</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    section_header("Available Outputs", "📦")

    render_download_card(
        icon="🎬",
        title="Tracked Match Video",
        description=(
            "Annotated match footage with bounding boxes, player IDs, team colors, "
            "and trajectory overlays. Produced by Member 1 using YOLOv8 + ByteTrack."
        ),
        file_path=TRACKED_VIDEO,
        file_name="tracked_video.mp4",
        mime="video/mp4",
    )

    render_download_card(
        icon="📊",
        title="Analytics Report (JSON)",
        description=(
            "Full per-player statistics including distance, speed, touches, "
            "possession percentage, team assignments, and match summary. Produced by Member 2."
        ),
        file_path=ANALYTICS_JSON,
        file_name="analytics.json",
        mime="application/json",
    )

    render_download_card(
        icon="🔍",
        title="Frame Detections (JSON)",
        description=(
            "Raw detection output — one record per object per frame. Includes "
            "bounding boxes, class labels, confidence scores, and pitch coordinates."
        ),
        file_path=DETECTIONS_JSON,
        file_name="detections.json",
        mime="application/json",
    )

    # ── Export info ──────────────────────────────────────────────────────────
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class="insight-card info">
            <strong>📌 File Location Reference</strong><br>
            <div style="margin-top:8px; font-size:0.82rem; line-height:1.8; color:#94A3B8;">
                <code>outputs/tracked_video.mp4</code> — Annotated video (Member 1)<br>
                <code>data/analytics.json</code> — Player &amp; team stats (Member 2)<br>
                <code>data/detections.json</code> — Raw detections (Member 1)<br>
                <code>outputs/player_heatmap.png</code> — Heatmap image (Member 2)<br>
                <code>outputs/ball_heatmap.png</code> — Ball heatmap image (Member 2)
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
