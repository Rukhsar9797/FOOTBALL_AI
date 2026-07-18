"""
dashboard/live_match.py
-----------------------
Live Match page: video playback, frame inspection, live detection overlay,
current possession, and player/ball position display.
"""

from typing import Dict, List, Optional

import streamlit as st

from utils import load_detections, load_analytics
from components.video_player import render_video_player, render_frame_inspector, get_frame_detections
from components.charts import player_positions_scatter
from components.metrics import section_header, metric_row, possession_bar


def render_live_match() -> None:
    """Render the Live Match page."""
    st.markdown(
        """
        <div class="page-header">
            <div class="page-title">📡 Live Match</div>
            <div class="page-subtitle">Tracked match video with real-time detection overlay</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    detections = load_detections()
    analytics  = load_analytics()

    # ── Tabs: Video vs Frame Inspector ──────────────────────────────────────
    tab1, tab2 = st.tabs(["▶ Match Video", "🔍 Frame Inspector"])

    with tab1:
        # ── Video Player ─────────────────────────────────────────────────
        section_header("Tracked Match Video", "🎬")
        render_video_player()

        st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

        # ── Summary stats below video ─────────────────────────────────────
        if analytics:
            ts = analytics.get("team_stats", {})
            blue = ts.get("Team 1", {})
            red  = ts.get("Team 2",  {})
            mi   = analytics.get("match_info", {})

            st.markdown("<hr>", unsafe_allow_html=True)
            section_header("Match Overview", "📋")

            metric_row([
                {
                    "icon": "⏱️",
                    "label": "Match Duration",
                    "value": _fmt_seconds(mi.get("duration_seconds", 0)),
                    "accent": "linear-gradient(90deg,#8B5CF6,#3B82F6)",
                    "delta_color": "#8B5CF6",
                },
                {
                    "icon": "👥",
                    "label": "Total Detected Players",
                    "value": str(len(analytics.get("players", []))),
                    "accent": "linear-gradient(90deg,#3B82F6,#00FF87)",
                    "delta_color": "#00FF87",
                },
                {
                    "icon": "📹",
                    "label": "Total Frames",
                    "value": f"{mi.get('total_frames', 0):,}",
                    "accent": "linear-gradient(90deg,#14B8A6,#3B82F6)",
                    "delta_color": "#14B8A6",
                },
                {
                    "icon": "🎯",
                    "label": "White Possession",
                    "value": f"{blue.get('possession_pct', 50):.0f}%",
                    "delta": "vs Red",
                    "accent": "linear-gradient(90deg,#3B82F6,#60A5FA)",
                    "delta_color": "#60A5FA",
                },
            ])

    with tab2:
        # ── Frame-by-frame inspector ──────────────────────────────────────
        frame_idx = render_frame_inspector(detections)

        if frame_idx is not None and detections:
            frame_dets = get_frame_detections(detections, frame_idx)
            st.markdown("<hr>", unsafe_allow_html=True)

            col1, col2 = st.columns([3, 2])
            with col1:
                section_header("Player Positions", "📍")
                fig = player_positions_scatter(frame_dets)
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                section_header("Frame Detections", "📊")
                players = [d for d in frame_dets if d.get("class") == "player"]
                referees = [d for d in frame_dets if d.get("class") == "referee"]
                balls = [d for d in frame_dets if d.get("class") == "ball"]

                metric_row([
                    {"icon": "🏃", "label": "Players",   "value": str(len(players))},
                    {"icon": "🟡", "label": "Referees",  "value": str(len(referees))},
                    {"icon": "⚽", "label": "Ball Det.", "value": str(len(balls))},
                ])

                # Ball position
                if balls:
                    b = balls[0]
                    st.markdown(
                        f"""
                        <div class="insight-card info" style="margin-top:1rem;">
                            <strong>⚽ Ball Position</strong><br>
                            X: {b.get('x', 0):.0f}px &nbsp;|&nbsp; Y: {b.get('y', 0):.0f}px<br>
                            <span style="color:#64748B; font-size:0.78rem;">
                                Pitch: {b.get('x_pitch', 0):.1f}m × {b.get('y_pitch', 0):.1f}m
                            </span>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                # Possession in this frame
                blue_count = sum(1 for d in players if d.get("team") == "Team 1")
                red_count  = sum(1 for d in players if d.get("team") == "Team 2")
                if blue_count + red_count > 0:
                    st.markdown("<div style='margin-top:1rem;'></div>", unsafe_allow_html=True)
                    section_header("Frame Possession", "⬜")
                    possession_bar(
                        (blue_count / (blue_count + red_count)) * 100,
                        (red_count  / (blue_count + red_count)) * 100,
                    )

                # Detection table
                if frame_dets:
                    st.markdown("<div style='margin-top:1rem;'></div>", unsafe_allow_html=True)
                    section_header("Detection List", "📋")
                    for det in frame_dets[:15]:
                        team  = det.get("team", "?")
                        pid   = det.get("player_id", "?")
                        cls   = det.get("class", "player")
                        badge_class = "team-2" if team == "Team 1" else \
                                      "team-1" if team == "Team 2" else ""
                        icon  = "⚽" if cls == "ball" else "🟡" if cls == "referee" else "🏃"
                        st.markdown(
                            f"""
                            <div style="display:flex; justify-content:space-between;
                                        align-items:center; padding:5px 0;
                                        border-bottom:1px solid #1E293B;">
                                <span style="color:#94A3B8; font-size:0.82rem;">
                                    {icon} {cls.title()} {pid}
                                </span>
                                <span class="team-badge {badge_class}">{team}</span>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )


def _fmt_seconds(s: float) -> str:
    m, sec = divmod(int(s), 60)
    return f"{m:02d}:{sec:02d}"
