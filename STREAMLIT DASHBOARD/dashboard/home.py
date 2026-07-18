"""
dashboard/home.py
-----------------
Home page: hero banner, hackathon info, and KPI summary cards.
"""

import streamlit as st
from typing import Optional, Dict

from utils import load_analytics, load_detections, compute_summary, fmt_distance, fmt_speed, fmt_time, fmt_pct
from components.metrics import metric_row, section_header, possession_bar
from config import ORIGINAL_VIDEO, OUTPUTS_DIR, DETECTIONS_JSON
from pipeline.run_pipeline import run_detection, run_analytics

def render_home() -> None:
    """Render the Home page."""

    st.markdown(
        """
        ...
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.subheader("📤 Football Match Input")

    uploaded_video = st.file_uploader(
        "Upload Original Football Match",
        type=["mp4"],
        key="football_video",
    )

    if uploaded_video is not None:
        with open(ORIGINAL_VIDEO, "wb") as f:
            f.write(uploaded_video.read())

        st.success("✅ Original video uploaded successfully!")

        if st.button("▶ Run Detection + Analytics Pipeline"):
            with st.spinner("Running detection..."):
                det_res = run_detection(ORIGINAL_VIDEO, OUTPUTS_DIR)
            
            if not det_res.get("success"):
                st.warning(det_res.get("message", "Detection pipeline failed."))
                st.stop()
                
            with st.spinner("Computing analytics..."):
                ana_res = run_analytics(DETECTIONS_JSON, OUTPUTS_DIR)
                
            if not ana_res.get("success"):
                st.warning(ana_res.get("message", "Analytics pipeline failed."))
                st.stop()
                
            st.success("✅ Pipeline completed successfully!")
            st.rerun()

    # ── Load data and parse with the ACTUAL analytics.json schema ───────────
    analytics  = load_analytics()
    detections = load_detections() or []

    # Derive real values from the new analytics.json schema where possible.
    # Fallbacks are shown only when data is not yet available.
    if analytics:
        possession_data = analytics.get("possession", {})
        white_poss      = float(possession_data.get("White Team", 0.0))
        red_poss        = float(possession_data.get("Red Team",   0.0))
        total_p         = white_poss + red_poss
        if total_p > 0:
            white_poss = round(white_poss / total_p * 100, 1)
            red_poss   = round(100.0 - white_poss, 1)

        ma_info     = analytics.get("most_active_player", {})
        hs_info     = analytics.get("highest_speed",      {})
        top_dist_l  = analytics.get("top_distance_players", [])

        most_active = f"Player {ma_info.get('player_id', '—')}"
        top_speed   = float(hs_info.get("speed", 0.0))
        top_dist    = float(top_dist_l[0]["distance_meters"]) if top_dist_l else 0.0
        team1_poss  = white_poss
        team2_poss  = red_poss
    else:
        most_active = "—"
        top_speed   = 0.0
        top_dist    = 0.0
        team1_poss  = 50.0
        team2_poss  = 50.0

    # Total player count — derived live from detections.json (distinct player_ids)
    if detections:
        total_players = len({d.get("player_id") for d in detections if d.get("class") == "player"})
    else:
        total_players = 0

    duration = 0  # not yet in new analytics schema; placeholder

    # ── Row 1 Metrics ────────────────────────────────────────────────────────
    section_header("Quick Statistics", "📊")
    metric_row([
        {
            "icon": "👥",
            "label": "Total Players Tracked",
            "value": str(total_players),
            "delta": "YOLOv8 Detection",
            "accent": "linear-gradient(90deg,#3B82F6,#00FF87)",
            "delta_color": "#00FF87",
        },
        {
            "icon": "⏱️",
            "label": "Match Duration",
            "value": fmt_time(duration),
            "delta": "Full Match",
            "accent": "linear-gradient(90deg,#8B5CF6,#3B82F6)",
            "delta_color": "#8B5CF6",
        },
        {
            "icon": "⚡",
            "label": "Top Speed Recorded",
            "value": fmt_speed(top_speed),
            "delta": "Sprint Peak",
            "accent": "linear-gradient(90deg,#F59E0B,#EF4444)",
            "delta_color": "#F59E0B",
        },
    ])

    st.markdown("<div style='height:0.75rem;'></div>", unsafe_allow_html=True)

    metric_row([
        {
            "icon": "🏃",
            "label": "Most Active Player",
            "value": most_active,
            "delta": "Highest Distance",
            "accent": "linear-gradient(90deg,#14B8A6,#3B82F6)",
            "delta_color": "#14B8A6",
        },
        {
            "icon": "📏",
            "label": "Top Distance Covered",
            "value": fmt_distance(top_dist),
            "delta": "Outfield Player",
            "accent": "linear-gradient(90deg,#00FF87,#14B8A6)",
            "delta_color": "#00FF87",
        },
        {
            "icon": "🎯",
            "label": "Ball Possession",
            "value": f"{team1_poss:.0f}% / {team2_poss:.0f}%",
            "delta": "White / Red",
            "accent": "linear-gradient(90deg,#3B82F6,#EF4444)",
            "delta_color": "#60A5FA",
        },
    ])

    # ── Possession Bar ───────────────────────────────────────────────────────
    st.markdown("<hr>", unsafe_allow_html=True)
    section_header("Ball Possession Overview", "⬜")
    possession_bar(team1_poss, team2_poss)

    # ── System Info ──────────────────────────────────────────────────────────
    st.markdown("<hr>", unsafe_allow_html=True)
    section_header("System Architecture", "🔧")

    col1, col2, col3 = st.columns(3)
    cards = [
        ("👁️ Detection", "Member 1", "YOLOv8 custom trained on football dataset. Player, referee, and ball detection across all frames."),
        ("🔗 Tracking", "Member 1", "ByteTrack multi-object tracker for consistent player IDs across frames with re-identification."),
        ("📊 Analytics", "Member 2", "Distance, speed, possession, heatmaps and spatial analysis per player and team."),
    ]
    for col, (title, member, desc) in zip([col1, col2, col3], cards):
        with col:
            st.markdown(
                f"""
                <div class="glass-card">
                    <div style="font-size:1.8rem; margin-bottom:0.5rem;">{title.split()[0]}</div>
                    <div style="font-family:'Rajdhani',sans-serif; font-size:1rem;
                                font-weight:700; color:#CBD5E1; margin-bottom:4px;">
                        {title.split(' ',1)[1]}
                    </div>
                    <div style="display:inline-block; background:rgba(59,130,246,0.15);
                                border:1px solid rgba(59,130,246,0.3); border-radius:4px;
                                padding:2px 8px; font-size:0.7rem; color:#60A5FA;
                                margin-bottom:8px;">{member}</div>
                    <div style="color:#64748B; font-size:0.8rem; line-height:1.5;">{desc}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # ── Project Description ──────────────────────────────────────────────────
    st.markdown("<hr>", unsafe_allow_html=True)
    with st.expander("📄 Project Description & Hackathon Details", expanded=False):
        st.markdown(
            """
            ### FootballIQ Analytics — Hackathon Project

            This dashboard is the **visualization and presentation layer** of a full-stack
            computer vision football analytics pipeline built during a **24-hour hackathon**.

            #### Pipeline Overview
            ```
            [Raw Match Video]
                    ↓
            [Member 1] YOLOv8 Detection + ByteTrack → tracked_video.mp4 + detections.json
                    ↓
            [Member 2] Analytics Engine → analytics.json + player_heatmap.png + ball_heatmap.png
                    ↓
            [Member 3] Streamlit Dashboard
            ```

            #### Key Technical Highlights
            - **Detection**: Custom YOLOv8 model fine-tuned on football datasets
            - **Tracking**: ByteTrack algorithm for robust multi-object tracking
            - **Analytics**: Per-player distance, speed, possession, and spatial metrics
            - **Visualization**: Professional Streamlit dashboard with Plotly charts

            #### Member Responsibilities
            | Member | Role | Deliverables |
            |--------|------|-------------|
            | Member 1 | Detection & Tracking | tracked_video.mp4, detections.json |
            | Member 2 | Analytics & Heatmaps | analytics.json, player_heatmap.png, ball_heatmap.png |
            | Member 3 | Dashboard (This App) | Streamlit visualization platform |
            """,
            unsafe_allow_html=True,
        )
