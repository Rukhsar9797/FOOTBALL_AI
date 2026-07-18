"""
dashboard/team_analysis.py
--------------------------
Team Analysis page: side-by-side statistics, possession donut,
comparison bar charts, and top-player rankings.

analytics.json actual schema (as of current Member 2 output):
    {
        "possession":           {"Red Team": float, "White Team": float},
        "top_distance_players": [{"player_id": int, "distance_meters": float}, ...],
        "highest_speed":        {"player_id": int, "speed": float},
        "most_active_player":   {"player_id": int, "activity_score": float},
        "momentum":             {"dominant_team": str}
    }

NOTE: The old keys team_stats / players / match_info no longer exist in
analytics.json. All per-player stats are derived live from detections.json.
"""

from typing import Dict, List, Optional

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils import load_analytics, load_detections, fmt_distance, fmt_speed, fmt_pct
from components.charts import possession_donut, team_comparison_bar, _empty_fig
from components.metrics import section_header, metric_row, possession_bar
from config import PLOTLY_TEMPLATE, CHART_BG, CHART_PAPER_BG

# Team name mapping: analytics.json uses these exact strings
TEAM_WHITE = "White Team"
TEAM_RED   = "Red Team"

# Highly visible dark colors optimized for a light dashboard theme
_TEAM_COLORS: Dict[str, str] = {
    TEAM_WHITE: "#0F172A",  # Dark Navy Slate instead of light white
    TEAM_RED:   "#B91C1C",  # Deep Crimson Red instead of bright neon red
}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers to derive per-team stats from detections.json
# ─────────────────────────────────────────────────────────────────────────────

def _detections_to_player_df(detections: List[Dict]) -> pd.DataFrame:
    """
    Aggregate per-player metrics from flat detections list.
    Each detection row has: player_id, team, current_speed, x_pitch, y_pitch.
    Returns one row per player_id with: player_id, team, max_speed, avg_speed,
    distance_approx (count-based proxy), frame_count.
    """
    if not detections:
        return pd.DataFrame()

    df = pd.DataFrame(detections)
    # Keep only player detections
    if "class" in df.columns:
        df = df[df["class"] == "player"]

    if df.empty or "player_id" not in df.columns:
        return pd.DataFrame()

    # Ensure numeric speed
    if "current_speed" in df.columns:
        df["current_speed"] = pd.to_numeric(df["current_speed"], errors="coerce").fillna(0)
    else:
        df["current_speed"] = 0.0

    agg = (
        df.groupby(["player_id", "team"], as_index=False)
        .agg(
            max_speed=("current_speed", "max"),
            avg_speed=("current_speed", "mean"),
            frame_count=("frame", "count") if "frame" in df.columns else ("current_speed", "count"),
        )
    )
    agg["max_speed"]   = agg["max_speed"].round(2)
    agg["avg_speed"]   = agg["avg_speed"].round(2)
    return agg


def _team_summary(player_df: pd.DataFrame, team_name: str) -> Dict:
    """Return aggregate stats for a single team from the player DataFrame."""
    sub = player_df[player_df["team"] == team_name] if not player_df.empty else pd.DataFrame()
    if sub.empty:
        return {"player_count": 0, "max_speed": 0.0, "avg_speed": 0.0, "frame_count": 0}
    return {
        "player_count": len(sub),
        "max_speed":    round(float(sub["max_speed"].max()), 2),
        "avg_speed":    round(float(sub["avg_speed"].mean()), 2),
        "frame_count":  int(sub["frame_count"].sum()),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Page renderer
# ─────────────────────────────────────────────────────────────────────────────

def render_team_analysis() -> None:
    """Render the Team Analysis page."""
    st.markdown(
        """
        <div class="page-header">
            <div class="page-title">📊 Team Analysis</div>
            <div class="page-subtitle">Comparative statistics and performance breakdown by team</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    analytics  = load_analytics()
    detections = load_detections() or []

    # ── Guard: analytics.json must be present ────────────────────────────────
    if not analytics:
        st.markdown(
            '<div class="warning-card"><span class="warning-icon">⚠️</span>'
            '<div><strong>analytics.json not found.</strong><br>'
            '<small>Place the file in the <code>data/</code> directory.</small></div></div>',
            unsafe_allow_html=True,
        )
        return

    # ── Parse analytics.json with the ACTUAL schema ───────────────────────
    possession: Dict[str, float] = analytics.get("possession", {})
    white_poss = float(possession.get(TEAM_WHITE, 50.0))
    red_poss   = float(possession.get(TEAM_RED, 50.0))

    # Normalise in case they don't sum to 100 (defensive)
    total_poss = white_poss + red_poss
    if total_poss > 0:
        white_poss = round(white_poss / total_poss * 100, 1)
        red_poss   = round(100.0 - white_poss, 1)

    momentum_team: str = analytics.get("momentum", {}).get("dominant_team", "—")
    momentum_color = _TEAM_COLORS.get(momentum_team, "#475569")

    top_dist_list: List[Dict] = analytics.get("top_distance_players", [])
    highest_speed_info: Dict  = analytics.get("highest_speed", {})
    most_active_info: Dict    = analytics.get("most_active_player", {})

    # ── Derive live per-team stats from detections.json ───────────────────
    player_df   = _detections_to_player_df(detections)
    white_stats = _team_summary(player_df, TEAM_WHITE)
    red_stats   = _team_summary(player_df, TEAM_RED)

    # ── Possession Overview ───────────────────────────────────────────────
    section_header("Ball Possession", "🎯")

    col_donut, col_bar = st.columns([1, 2])

    with col_donut:
        fig = possession_donut(white_poss, red_poss)
        st.plotly_chart(fig, use_container_width=True)

    with col_bar:
        st.markdown("<div style='height:1rem;'></div>", unsafe_allow_html=True)
        possession_bar(white_poss, red_poss)

        metric_row([
            {
                "icon": "⬛",
                "label": f"{TEAM_WHITE} Possession",
                "value": fmt_pct(white_poss),
                "accent": "linear-gradient(90deg,#475569,#0F172A)",
                "delta_color": "#0F172A",
            },
            {
                "icon": "🔴",
                "label": f"{TEAM_RED} Possession",
                "value": fmt_pct(red_poss),
                "accent": "linear-gradient(90deg,#DC2626,#B91C1C)",
                "delta_color": "#B91C1C",
            },
        ])

    # ── Match Momentum & Winner Banner ────────────────────────────────────
    st.markdown("<hr>", unsafe_allow_html=True)
    section_header("Match Momentum", "⚡")

    st.markdown(
        f"""
        <div class="glass-card" style="border-top: 3px solid {momentum_color}; text-align:center;
                                        padding:1rem 0 1.2rem 0; margin-bottom:1.5rem; background: rgba(0,0,0,0.02);">
            <div style="color:#64748B; font-size:0.78rem; text-transform:uppercase;
                        letter-spacing:0.08em; margin-bottom:0.5rem;">Dominant Team</div>
            <div style="font-family:'Rajdhani',sans-serif; font-size:2.2rem;
                        font-weight:900; color:{momentum_color};">
                {momentum_team} 🏆
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Side-by-side team cards ───────────────────────────────────────────
    st.markdown("<hr>", unsafe_allow_html=True)
    section_header("Team Performance Cards", "📋")
    col_w, col_r = st.columns(2)

    _team_card(col_w, TEAM_WHITE, white_poss, white_stats, "#FFFFFF", _TEAM_COLORS[TEAM_WHITE])
    _team_card(col_r, TEAM_RED,   red_poss,   red_stats,   "#EF4444", _TEAM_COLORS[TEAM_RED])

    # ── Head-to-Head comparison bar ───────────────────────────────────────
    st.markdown("<hr>", unsafe_allow_html=True)
    section_header("Head-to-Head Comparison", "⚖️")

    cats   = ["Possession (%)", "Player Count", "Max Speed (km/h)", "Avg Speed (km/h)"]
    w_vals = [
        white_poss,
        float(white_stats["player_count"]),
        white_stats["max_speed"],
        white_stats["avg_speed"],
    ]
    r_vals = [
        red_poss,
        float(red_stats["player_count"]),
        red_stats["max_speed"],
        red_stats["avg_speed"],
    ]

    fig_bar = team_comparison_bar(cats, w_vals, r_vals, title="Team Metrics Comparison")
    st.plotly_chart(fig_bar, use_container_width=True)

    # ── Top Distance Players ──────────────────────────────────────────────
    if top_dist_list:
        st.markdown("<hr>", unsafe_allow_html=True)
        section_header("Top Distance Players", "📏")

        top_df = pd.DataFrame(top_dist_list)
        if not player_df.empty and "player_id" in player_df.columns:
            team_lookup = player_df.drop_duplicates("player_id")[["player_id", "team"]]
            top_df = top_df.merge(team_lookup, on="player_id", how="left")
            top_df["team"] = top_df["team"].fillna("Unknown")
        else:
            top_df["team"] = "—"

        top_df["Distance"] = top_df["distance_meters"].apply(fmt_distance)
        top_df["Rank"]     = range(1, len(top_df) + 1)
        top_df = top_df.rename(columns={"player_id": "Player ID"})
        display_cols = ["Rank", "Player ID", "team", "Distance"]
        if "team" in top_df.columns:
            top_df = top_df.rename(columns={"team": "Team"})
            display_cols = ["Rank", "Player ID", "Team", "Distance"]

        st.dataframe(
            top_df[display_cols],
            use_container_width=True,
            hide_index=True,
        )

    # ── Key Award Highlights ──────────────────────────────────────────────
    st.markdown("<hr>", unsafe_allow_html=True)
    section_header("Match Highlights", "🏅")

    col1, col2 = st.columns(2)

    with col1:
        hs_pid   = highest_speed_info.get("player_id", "—")
        hs_speed = float(highest_speed_info.get("speed", 0))
        st.markdown(
            f"""
            <div class="glass-card" style="border-top:3px solid #F59E0B; text-align:center;">
                <div style="font-size:2rem; margin-bottom:0.4rem;">⚡</div>
                <div style="color:#64748B; font-size:0.7rem; text-transform:uppercase;
                            letter-spacing:0.08em;">Highest Speed Recorded</div>
                <div style="font-family:'Rajdhani',sans-serif; font-size:2rem;
                            font-weight:800; color:#F59E0B;">P{hs_pid}</div>
                <div style="background:rgba(245,158,11,0.15); border:1px solid rgba(245,158,11,0.3);
                            border-radius:8px; padding:4px 12px; display:inline-block;
                            font-weight:700; color:#F59E0B; font-size:1rem; margin-top:4px;">
                    {fmt_speed(hs_speed) if hs_speed else "N/A"}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        ma_pid   = most_active_info.get("player_id", "—")
        ma_score = float(most_active_info.get("activity_score", 0))
        st.markdown(
            f"""
            <div class="glass-card" style="border-top:3px solid #00FF87; text-align:center;">
                <div style="font-size:2rem; margin-bottom:0.4rem;">🏃</div>
                <div style="color:#64748B; font-size:0.7rem; text-transform:uppercase;
                            letter-spacing:0.08em;">Most Active Player</div>
                <div style="font-family:'Rajdhani',sans-serif; font-size:2rem;
                            font-weight:800; color:#00FF87;">P{ma_pid}</div>
                <div style="background:rgba(0,255,135,0.15); border:1px solid rgba(0,255,135,0.3);
                            border-radius:8px; padding:4px 12px; display:inline-block;
                            font-weight:700; color:#00FF87; font-size:1rem; margin-top:4px;">
                    Activity Score: {ma_score:.1f}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Sub-component: single team card
# ─────────────────────────────────────────────────────────────────────────────

def _team_card(
    col,
    team_name: str,
    possession: float,
    stats: Dict,
    badge_color: str,
    text_color: str,
) -> None:
    """Render a single team stat card in the given Streamlit column."""
    # Handle subtle visual markers cleanly on light canvas backgrounds
    shadow_color = "148,163,184" if badge_color.upper() == "#FFFFFF" else "185,28,28"
    
    count   = stats.get("player_count", 0)
    max_spd = stats.get("max_speed", 0.0)
    avg_spd = stats.get("avg_speed", 0.0)

    with col:
        st.markdown(
            f"""
            <div class="glass-card" style="border-top: 3px solid {text_color}; background: #F8FAFC;">
                <div style="display:flex; align-items:center; gap:10px; margin-bottom:1rem;">
                    <div style="width:16px; height:16px; border-radius:50%; border: 1px solid #64748B;
                                background:{badge_color}; box-shadow:0 0 10px rgba({shadow_color},0.6);">
                    </div>
                    <div style="font-family:'Rajdhani',sans-serif; font-size:1.3rem;
                                font-weight:700; color:{text_color};">{team_name}</div>
                </div>
                <div class="stat-row">
                    <span class="stat-key" style="color:#475569;">👥 Player Count</span>
                    <span class="stat-value" style="color:#0F172A; font-weight:600;">{count}</span>
                </div>
                <div class="stat-row">
                    <span class="stat-key" style="color:#475569;">💨 Average Speed</span>
                    <span class="stat-value" style="color:#0F172A; font-weight:600;">{fmt_speed(avg_spd)}</span>
                </div>
                <div class="stat-row">
                    <span class="stat-key" style="color:#475569;">⚡ Max Speed</span>
                    <span class="stat-value" style="color:#0F172A; font-weight:600;">{fmt_speed(max_spd)}</span>
                </div>
                <div class="stat-row">
                    <span class="stat-key" style="color:#475569;">🎯 Possession</span>
                    <span class="stat-value" style="color:{text_color}; font-weight:700;">{fmt_pct(possession)}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )