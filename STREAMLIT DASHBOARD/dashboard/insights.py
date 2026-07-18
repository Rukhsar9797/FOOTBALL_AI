"""
dashboard/insights.py
---------------------
Match Insights page: key findings, awards, top performer breakdowns,
and match narrative derived from the real analytics.json schema.

analytics.json actual schema (as of current Member 2 output):
    {
        "possession":           {"Red Team": float, "White Team": float},
        "top_distance_players": [{"player_id": int, "distance_meters": float}, ...],
        "highest_speed":        {"player_id": int, "speed": float},
        "most_active_player":   {"player_id": int, "activity_score": float},
        "momentum":             {"dominant_team": str}
    }

NOTE: The box-plot section ("📈 Box Plots" tab) has been intentionally removed
per the Change 2 requirement. _box_plot() is still defined in components/charts.py
if needed in future; it is simply no longer called from this page.
"""

from typing import Dict, List, Optional

import pandas as pd
import streamlit as st

from utils import (
    load_analytics,
    load_detections,
    fmt_distance,
    fmt_speed,
    fmt_pct,
    fmt_time,
)
from components.metrics import section_header, metric_row, insight_card

# Team display colours — detections.json teams are "White Team" / "Red Team"
_TEAM_COLORS: Dict[str, str] = {
    "White Team": "#3B82F6",
    "Red Team":   "#EF4444",
}
_DEFAULT_COLOR = "#94A3B8"


# ─────────────────────────────────────────────────────────────────────────────
# Page renderer
# ─────────────────────────────────────────────────────────────────────────────

def render_insights() -> None:
    """Render the Match Insights page."""
    st.markdown(
        """
        <div class="page-header">
            <div class="page-title">💡 Match Insights</div>
            <div class="page-subtitle">AI-powered analysis — key findings, awards, and match narrative</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    analytics = load_analytics()
    if not analytics:
        st.markdown(
            '<div class="warning-card"><span class="warning-icon">⚠️</span>'
            '<div><strong>analytics.json not found.</strong><br>'
            '<small>Expected at <code>data/analytics.json</code>.</small></div></div>',
            unsafe_allow_html=True,
        )
        return

    # ── Parse analytics.json with the ACTUAL schema ───────────────────────────
    possession: Dict[str, float]  = analytics.get("possession", {})
    top_dist_list: List[Dict]     = analytics.get("top_distance_players", [])
    highest_speed_info: Dict      = analytics.get("highest_speed", {})
    most_active_info: Dict        = analytics.get("most_active_player", {})
    momentum_team: str            = analytics.get("momentum", {}).get("dominant_team", "—")

    white_poss = float(possession.get("White Team", 50.0))
    red_poss   = float(possession.get("Red Team",   50.0))

    # Normalise possession to 100 % (defensive guard)
    total_poss = white_poss + red_poss
    if total_poss > 0:
        white_poss = round(white_poss / total_poss * 100, 1)
        red_poss   = round(100.0 - white_poss, 1)

    poss_winner       = "White Team" if white_poss >= red_poss else "Red Team"
    poss_winner_val   = max(white_poss, red_poss)
    winner_color      = _TEAM_COLORS.get(poss_winner, _DEFAULT_COLOR)
    momentum_color    = _TEAM_COLORS.get(momentum_team, _DEFAULT_COLOR)

    # Highlights from analytics
    hs_pid    = highest_speed_info.get("player_id", "—")
    hs_speed  = float(highest_speed_info.get("speed", 0))
    ma_pid    = most_active_info.get("player_id", "—")
    ma_score  = float(most_active_info.get("activity_score", 0))

    # Top distance players (for awards / table)
    top_dist_map: Dict[int, float] = {
        int(e["player_id"]): float(e["distance_meters"])
        for e in top_dist_list
        if "player_id" in e and "distance_meters" in e
    }

    # ── Match Summary Card ────────────────────────────────────────────────────
    section_header("Match Summary", "📋")
    st.markdown(
        f"""
        <div class="glass-card" style="border-top: 3px solid {winner_color}; margin-bottom:1.5rem;">
            <div style="text-align:center; padding:0.5rem 0 1rem 0;">
                <div style="color:#64748B; font-size:0.78rem; margin-bottom:0.5rem;">
                    MATCH CONTROL WINNER
                </div>
                <div style="font-family:'Rajdhani',sans-serif; font-size:2.5rem;
                            font-weight:900; color:{winner_color};">
                    {poss_winner} 🏆
                </div>
                <div style="color:#94A3B8; font-size:0.9rem; margin-top:4px;">
                    Possession: <strong>{white_poss:.1f}%</strong> (White)
                    vs <strong>{red_poss:.1f}%</strong> (Red)
                </div>
            </div>
            <div style="display:grid; grid-template-columns:1fr 1fr; gap:1rem; padding-top:1rem;
                        border-top:1px solid #1E293B;">
                <div style="text-align:center;">
                    <div style="color:#64748B; font-size:0.72rem; text-transform:uppercase;">
                        Dominant Team
                    </div>
                    <div style="font-family:'Rajdhani',sans-serif; font-size:1.5rem;
                                font-weight:700; color:{momentum_color};">
                        {momentum_team}
                    </div>
                </div>
                <div style="text-align:center;">
                    <div style="color:#64748B; font-size:0.72rem; text-transform:uppercase;">
                        Most Active
                    </div>
                    <div style="font-family:'Rajdhani',sans-serif; font-size:1.5rem;
                                font-weight:700; color:#00FF87;">
                        Player {ma_pid}
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Player Awards ─────────────────────────────────────────────────────────
    section_header("Player Awards", "🏅")
    col1, col2, col3 = st.columns(3)

    with col1:
        _award_card_static(
            icon="🏃",
            award="Most Active Player",
            player_id=ma_pid,
            label="Activity Score",
            value=f"{ma_score:.1f}",
            color="#00FF87",
        )

    with col2:
        _award_card_static(
            icon="⚡",
            award="Highest Speed",
            player_id=hs_pid,
            label="Top Speed",
            value=fmt_speed(hs_speed) if hs_speed else "N/A",
            color="#F59E0B",
        )

    with col3:
        # Top distance from top_distance_players[0]
        if top_dist_list:
            top1 = top_dist_list[0]
            td_pid  = top1.get("player_id", "—")
            td_dist = float(top1.get("distance_meters", 0))
        else:
            td_pid  = "—"
            td_dist = 0.0

        _award_card_static(
            icon="📏",
            award="Longest Distance",
            player_id=td_pid,
            label="Distance",
            value=fmt_distance(td_dist) if td_dist else "N/A",
            color="#8B5CF6",
        )

    # ── Key Insights ──────────────────────────────────────────────────────────
    st.markdown("<hr>", unsafe_allow_html=True)
    section_header("Key Insights", "🔍")

    _auto_insights(
        poss_winner=poss_winner,
        poss_winner_val=poss_winner_val,
        momentum_team=momentum_team,
        hs_pid=hs_pid,
        hs_speed=hs_speed,
        ma_pid=ma_pid,
        ma_score=ma_score,
        top_dist_list=top_dist_list,
    )

    # ── Top Distance Players Table ────────────────────────────────────────────
    if top_dist_list:
        st.markdown("<hr>", unsafe_allow_html=True)
        section_header("Top 5 — Distance Covered", "🏃")

        # NOTE: The box plot that was previously here has been removed (Change 2).
        # _box_plot() still exists in components/charts.py but is not called.

        top_df = pd.DataFrame(top_dist_list[:5])
        top_df["Rank"]     = range(1, len(top_df) + 1)
        top_df["Distance"] = top_df["distance_meters"].apply(fmt_distance)
        top_df = top_df.rename(columns={"player_id": "Player ID"})

        st.dataframe(
            top_df[["Rank", "Player ID", "Distance"]],
            use_container_width=True,
            hide_index=True,
        )

    # ── Possession Summary Cards ──────────────────────────────────────────────
    st.markdown("<hr>", unsafe_allow_html=True)
    section_header("Possession Breakdown", "🎯")

    metric_row([
        {
            "icon": "⬜",
            "label": "White Team Possession",
            "value": fmt_pct(white_poss),
            "accent": "linear-gradient(90deg,#1D4ED8,#3B82F6)",
            "delta_color": "#60A5FA",
        },
        {
            "icon": "🔴",
            "label": "Red Team Possession",
            "value": fmt_pct(red_poss),
            "accent": "linear-gradient(90deg,#DC2626,#EF4444)",
            "delta_color": "#F87171",
        },
    ])


# ─────────────────────────────────────────────────────────────────────────────
# Sub-components
# ─────────────────────────────────────────────────────────────────────────────

def _award_card_static(
    icon: str,
    award: str,
    player_id,
    label: str,
    value: str,
    color: str,
) -> None:
    """Render a static award card (no DataFrame lookup needed)."""
    r_, g_, b__ = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
    st.markdown(
        f"""
        <div class="glass-card" style="border-top:3px solid {color}; text-align:center;">
            <div style="font-size:2.2rem; margin-bottom:0.4rem;">{icon}</div>
            <div style="color:#64748B; font-size:0.7rem; text-transform:uppercase;
                        letter-spacing:0.08em;">{award}</div>
            <div style="font-family:'Rajdhani',sans-serif; font-size:2rem;
                        font-weight:800; color:{color};">P{player_id}</div>
            <div style="background:rgba({r_},{g_},{b__},0.15); border:1px solid rgba({r_},{g_},{b__},0.3);
                        border-radius:8px; padding:4px 12px; display:inline-block;
                        font-weight:700; color:{color}; font-size:1rem; margin-top:4px;">
                {label}: {value}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _auto_insights(
    poss_winner: str,
    poss_winner_val: float,
    momentum_team: str,
    hs_pid,
    hs_speed: float,
    ma_pid,
    ma_score: float,
    top_dist_list: List[Dict],
) -> None:
    """Generate insight cards automatically from the parsed analytics data."""

    # Possession dominance
    insight_card(
        f"🎯 <strong>{poss_winner}</strong> dominated ball possession with "
        f"<strong>{poss_winner_val:.1f}%</strong> — indicating superior control in build-up play.",
        "default",
    )

    # Momentum
    if momentum_team and momentum_team != "—":
        insight_card(
            f"⚡ <strong>{momentum_team}</strong> held match momentum, suggesting sustained "
            "pressure and better transitional play throughout the match.",
            "info",
        )

    # Speed
    if hs_speed and hs_speed > 0:
        insight_card(
            f"⚡ Player <strong>P{hs_pid}</strong> recorded the highest sprint speed of "
            f"<strong>{fmt_speed(hs_speed)}</strong> — standout pace performance.",
            "info",
        )
    else:
        insight_card(
            f"⚡ Player <strong>P{hs_pid}</strong> was flagged as the highest-speed detection "
            "— exact speed data pending from the analytics pipeline.",
            "warning",
        )

    # Most active
    insight_card(
        f"🏃 Player <strong>P{ma_pid}</strong> was the most active player on the pitch "
        f"with an activity score of <strong>{ma_score:.1f}</strong> — highest workrate recorded.",
        "default",
    )

    # Distance
    if top_dist_list:
        top1 = top_dist_list[0]
        td_pid  = top1.get("player_id", "—")
        td_dist = float(top1.get("distance_meters", 0))
        insight_card(
            f"📏 Player <strong>P{td_pid}</strong> covered the most ground at "
            f"<strong>{fmt_distance(td_dist)}</strong> — exceptional engine performance.",
            "default",
        )
