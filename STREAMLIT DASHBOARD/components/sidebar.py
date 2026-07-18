"""
components/sidebar.py
---------------------
Renders the styled sidebar navigation for the dashboard.
Returns a tuple of (currently selected page name, is_live_mode boolean).
"""

import streamlit as st
from config import APP_TITLE, APP_ICON, APP_VERSION
from typing import Tuple

# Navigation items: (icon, label)
NAV_ITEMS = [
    ("🏠", "Home"),
    ("📡", "Live Match"),
    ("📊", "Team Analysis"),
    ("👤", "Player Analysis"),
    ("🌡️", "Heatmaps"),
    ("💡", "Match Insights"),
    ("⬇️", "Downloads"),
    ("ℹ️", "About Project"),
]


def render_sidebar() -> Tuple[str, bool]:
    """
    Render the navigation sidebar.
    Returns:
        (str, bool): The label of the selected page, and True if Live Mode is enabled.
    """
    with st.sidebar:
        # ── Logo / Brand ────────────────────────────────────
        st.markdown(
            f"""
            <div class="logo-bar">
                <span style="font-size:2rem;">{APP_ICON}</span>
                <div>
                    <div class="logo-title">{APP_TITLE}</div>
                    <span class="logo-version">v{APP_VERSION}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ── Data Mode Toggle ────────────────────────────────
        is_live = st.toggle("🔴 Enable Live Mode", value=False, help="Toggle between Demo data and real Hackathon outputs.")
        if is_live:
            st.markdown("<div style='color:#00FF87; font-size:0.75rem; margin-top:-10px; margin-bottom:15px;'>Reading from real output files...</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div style='color:#60A5FA; font-size:0.75rem; margin-top:-10px; margin-bottom:15px;'>Reading from generated dummy data...</div>", unsafe_allow_html=True)

        # ── Navigation ──────────────────────────────────────
        st.markdown(
            '<div class="section-title" style="font-size:0.7rem;margin-bottom:0.5rem;">NAVIGATION</div>',
            unsafe_allow_html=True,
        )

        labels = [f"{icon}  {label}" for icon, label in NAV_ITEMS]
        selected_full = st.radio(
            label="navigation",
            options=labels,
            label_visibility="collapsed",
        )

        # Extract label without icon prefix
        selected = selected_full.split("  ", 1)[1] if "  " in selected_full else selected_full

        # ── Match Status Widget ─────────────────────────────
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown(
            """
            <div style="text-align:center; padding: 0.75rem 0;">
                <div class="live-label"><span class="live-dot"></span> ANALYSIS ACTIVE</div>
                <div style="color:#475569; font-size:0.72rem; margin-top:0.5rem;">
                    Model: YOLOv8 + ByteTrack
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ── Footer ──────────────────────────────────────────
        st.markdown(
            """
            <div style="padding-top:1rem; border-top:1px solid #1E293B; text-align:center;">
                <div style="color:#334155; font-size:0.7rem; line-height:1.7;">
                    24-Hour CV Hackathon<br>
                    Member 3 · Visualization &amp; Dashboard<br>
                    <span style="color:#1E3A5F;">© 2025 FootballIQ</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    return selected, is_live
