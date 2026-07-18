"""
dashboard/about.py
------------------
About Project page: team info, tech stack, architecture diagram, and credits.
"""

import streamlit as st
from components.metrics import section_header


def render_about() -> None:
    """Render the About Project page."""
    st.markdown(
        """
        <div class="page-header">
            <div class="page-title">ℹ️ About Project</div>
            <div class="page-subtitle">Team overview, technology stack, and architecture</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Hero ─────────────────────────────────────────────────────────────────
    st.markdown(
        """
        <div class="hero-banner" style="padding:2rem 2rem; margin-bottom:1.5rem;">
            <div style="font-size:3rem; margin-bottom:0.5rem;">⚽</div>
            <div class="hero-title" style="font-size:2.2rem;">FootballIQ Analytics</div>
            <div class="hero-subtitle">
                An end-to-end computer vision pipeline for professional football match analysis
            </div>
            <div style="margin-top:1rem;">
                <span class="hackathon-badge">🏆 24-Hour CV Hackathon Submission</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Team Cards ────────────────────────────────────────────────────────────
    section_header("Team Members", "👥")
    members = [
        {
            "id": "01",
            "role": "Detection & Tracking",
            "desc": "Custom YOLOv8 training on football datasets. ByteTrack integration for consistent player IDs across all frames. Produces tracked_video.mp4 and detections.json.",
            "tech": ["YOLOv8", "ByteTrack", "OpenCV", "PyTorch"],
            "color": "#3B82F6",
        },
        {
            "id": "02",
            "role": "Analytics Engine",
            "desc": "Computes per-player distance, speed, possession, and spatial statistics. Generates KDE heatmaps for players and the ball. Produces analytics.json and heatmap images.",
            "tech": ["NumPy", "OpenCV", "Pandas", "Python", "JSON"],
            "color": "#8B5CF6",
        },
        {
            "id": "03",
            "role": "Visualization & Dashboard",
            "desc": "Professional Streamlit dashboard consuming outputs from Members 1 & 2. Interactive charts, heatmaps, player profiles, and match insights presentation.",
            "tech": ["Streamlit", "Plotly", "Pandas", "OpenCV", "Pillow"],
            "color": "#00FF87",
        },
    ]

    cols = st.columns(3)
    for col, m in zip(cols, members):
        with col:
            r, g, b_ = int(m["color"][1:3], 16), int(m["color"][3:5], 16), int(m["color"][5:7], 16)
            tags = "".join([
                f'<span style="background:rgba({r},{g},{b_},0.15); border:1px solid rgba({r},{g},{b_},0.3); '
                f'border-radius:4px; padding:2px 8px; font-size:0.7rem; color:{m["color"]}; margin-right:4px;">'
                f'{t}</span>'
                for t in m["tech"]
            ])
            st.markdown(
                f"""
                <div class="glass-card" style="border-top:3px solid {m["color"]}; height:100%;">
                    <div style="font-family:'Rajdhani',sans-serif; font-size:2rem;
                                font-weight:900; color:{m["color"]}; margin-bottom:0.25rem;">
                        MEMBER {m["id"]}
                    </div>
                    <div style="font-weight:700; color:#CBD5E1; margin-bottom:0.75rem;
                                font-size:0.95rem;">{m["role"]}</div>
                    <div style="color:#64748B; font-size:0.82rem; line-height:1.6;
                                margin-bottom:1rem;">{m["desc"]}</div>
                    <div style="display:flex; flex-wrap:wrap; gap:4px;">{tags}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # ── Architecture ──────────────────────────────────────────────────────────
    st.markdown("<hr>", unsafe_allow_html=True)
    section_header("System Architecture", "🏗️")

    st.markdown(
        """
        <div class="glass-card" style="font-family:monospace; font-size:0.85rem;
                                       color:#94A3B8; line-height:2;">
            <div style="color:#00FF87; font-weight:700; margin-bottom:0.5rem;">
                [Raw Match Video .mp4]
            </div>
            <div style="padding-left:2rem;">
                ↓
                <span style="background:rgba(59,130,246,0.15); border:1px solid rgba(59,130,246,0.3);
                             border-radius:4px; padding:2px 10px; color:#60A5FA;">
                  Member 1
                </span>
                YOLOv8 Detection + ByteTrack
            </div>
            <div style="padding-left:4rem; color:#3B82F6;">
                → tracked_video.mp4 &nbsp; → &nbsp; detections.json
            </div>
            <div style="padding-left:2rem; margin-top:0.5rem;">
                ↓
                <span style="background:rgba(139,92,246,0.15); border:1px solid rgba(139,92,246,0.3);
                             border-radius:4px; padding:2px 10px; color:#A78BFA;">
                  Member 2
                </span>
                Analytics Engine (KPIs + Heatmaps)
            </div>
            <div style="padding-left:4rem; color:#8B5CF6;">
                → analytics.json &nbsp; → &nbsp; player_heatmap.png &nbsp; → &nbsp; ball_heatmap.png
            </div>
            <div style="padding-left:2rem; margin-top:0.5rem;">
                ↓
                <span style="background:rgba(0,255,135,0.15); border:1px solid rgba(0,255,135,0.3);
                             border-radius:4px; padding:2px 10px; color:#00FF87;">
                  Member 3
                </span>
                Streamlit Analytics Dashboard
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Tech Stack ────────────────────────────────────────────────────────────
    st.markdown("<hr>", unsafe_allow_html=True)
    section_header("Dashboard Tech Stack", "🛠️")

    stack = [
        ("🐍", "Python 3.11+", "Core language"),
        ("🌐", "Streamlit", "Dashboard framework"),
        ("📊", "Plotly", "Interactive charts"),
        ("🐼", "Pandas", "Data manipulation"),
        ("🔢", "NumPy", "Numerical computing"),
        ("📷", "OpenCV", "Video frame extraction"),
        ("🖼️", "Pillow", "Image loading & processing"),
    ]

    cols = st.columns(4)
    for i, (icon, name, desc) in enumerate(stack):
        with cols[i % 4]:
            st.markdown(
                f"""
                <div class="glass-card" style="text-align:center; padding:1rem;">
                    <div style="font-size:1.8rem; margin-bottom:0.4rem;">{icon}</div>
                    <div style="font-weight:700; color:#CBD5E1; font-size:0.9rem;">{name}</div>
                    <div style="color:#475569; font-size:0.72rem;">{desc}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # ── Version info ─────────────────────────────────────────────────────────
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(
        """
        <div style="text-align:center; color:#334155; font-size:0.78rem; padding:1rem;">
            FootballIQ Analytics v1.0.0 &nbsp;·&nbsp;
            24-Hour Computer Vision Hackathon
        </div>
        """,
        unsafe_allow_html=True,
    )
