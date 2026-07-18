"""
components/metrics.py
---------------------
Reusable metric card builders rendered as raw HTML for full style control.
"""

from typing import Optional
import streamlit as st


def metric_card(
    icon: str,
    label: str,
    value: str,
    delta: Optional[str] = None,
    accent: str = "linear-gradient(90deg, #00FF87, #3B82F6)",
    delta_color: str = "#00FF87",
) -> None:
    """Render a single premium metric card."""
    delta_html = (
        f'<div class="metric-delta" style="color:{delta_color};">{delta}</div>'
        if delta
        else ""
    )
    st.markdown(
        f"""
        <div class="metric-card" style="--accent:{accent};">
            <div class="metric-icon">{icon}</div>
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            {delta_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def metric_row(items: list) -> None:
    """
    Render multiple metric cards in equal columns.
    Each item is a dict with keys: icon, label, value, delta (optional).
    """
    cols = st.columns(len(items))
    for col, item in zip(cols, items):
        with col:
            metric_card(
                icon=item["icon"],
                label=item["label"],
                value=item["value"],
                delta=item.get("delta"),
                accent=item.get("accent", "linear-gradient(90deg, #00FF87, #3B82F6)"),
                delta_color=item.get("delta_color", "#00FF87"),
            )


def possession_bar(team1_pct: float, team2_pct: float) -> None:
    """Render a styled horizontal possession bar."""
    st.markdown(
        f"""
        <div style="margin:1rem 0;">
            <div style="display:flex; justify-content:space-between; margin-bottom:4px;">
                <span style="color:#60A5FA; font-size:0.8rem; font-weight:600;">⬜ Team 1</span>
                <span style="color:#F87171; font-size:0.8rem; font-weight:600;">Team 2 🔴</span>
            </div>
            <div class="possession-bar">
                <div class="poss-2" style="width:{team1_pct:.1f}%;">{team1_pct:.0f}%</div>
                <div class="poss-1"  style="width:{team2_pct:.1f}%;">{team2_pct:.0f}%</div>
            </div>
            <div style="display:flex; justify-content:space-between; margin-top:4px;">
                <span style="color:#475569; font-size:0.7rem;">Ball Possession</span>
                <span style="color:#475569; font-size:0.7rem;">Ball Possession</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def stat_row_pair(
    label: str,
    blue_val: str,
    red_val: str,
    blue_raw: float = 0,
    red_raw: float = 0,
) -> None:
    """Side-by-side team stat with progress bars."""
    total = (blue_raw + red_raw) or 1
    blue_w = (blue_raw / total) * 100
    red_w  = (red_raw / total) * 100

    st.markdown(
        f"""
        <div style="margin-bottom:1rem;">
            <div style="display:flex; justify-content:space-between; margin-bottom:4px;">
                <span style="color:#60A5FA; font-weight:600;">{blue_val}</span>
                <span style="color:#64748B; font-size:0.78rem;">{label}</span>
                <span style="color:#F87171; font-weight:600;">{red_val}</span>
            </div>
            <div style="display:flex; gap:4px; height:6px; border-radius:999px; overflow:hidden;">
                <div style="width:{blue_w:.1f}%; background:linear-gradient(90deg,#1D4ED8,#3B82F6); border-radius:999px 0 0 999px;"></div>
                <div style="width:{red_w:.1f}%; background:linear-gradient(90deg,#EF4444,#DC2626); border-radius:0 999px 999px 0;"></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_header(title: str, icon: str = "") -> None:
    """Render a styled section separator header."""
    st.markdown(
        f'<div class="section-title">{icon} {title}</div>',
        unsafe_allow_html=True,
    )


def insight_card(text: str, variant: str = "default") -> None:
    """Render a highlighted insight card."""
    css_class = f"insight-card {variant}" if variant != "default" else "insight-card"
    st.markdown(
        f'<div class="{css_class}">{text}</div>',
        unsafe_allow_html=True,
    )
