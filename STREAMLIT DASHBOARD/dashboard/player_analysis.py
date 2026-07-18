"""
dashboard/player_analysis.py
-----------------------------
Player Analysis page: searchable player list, detailed player cards,
radar chart, speed timeline, and movement trail.

Data source strategy (analytics.json schema change):
  - analytics.json no longer contains a "players" list.
  - Per-player stats (speed, position) are derived live from detections.json,
    which has one row per detection per frame.
  - analytics.json is still loaded for top-level highlights (top_distance_players,
    highest_speed, most_active_player) that complement the per-player view.
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import streamlit as st

from utils import (
    load_analytics,
    load_detections,
    fmt_distance,
    fmt_speed,
    fmt_pct,
)
from components.charts import player_radar, player_trail_chart, speed_timeline
from components.metrics import section_header, metric_row
from config import PLOTLY_TEMPLATE, CHART_BG, CHART_PAPER_BG

# Team display colours (detections.json uses "White Team" / "Red Team")
_TEAM_COLORS: Dict[str, str] = {
    "White Team": "#3B82F6",  # blue for visibility on dark bg
    "Red Team":   "#EF4444",
}
_DEFAULT_COLOR = "#6B7280"


# ─────────────────────────────────────────────────────────────────────────────
# Build per-player summary from detections.json
# ─────────────────────────────────────────────────────────────────────────────

def _build_player_df(detections: List[Dict]) -> pd.DataFrame:
    """
    Aggregate all detections into one row per player_id.
    Columns: player_id, team, max_speed, avg_speed, frame_count, trail (list of {x,y}).
    """
    if not detections:
        return pd.DataFrame()

    df = pd.DataFrame(detections)

    # Keep only player class
    if "class" in df.columns:
        df = df[df["class"] == "player"]

    if df.empty or "player_id" not in df.columns:
        return pd.DataFrame()

    if "current_speed" in df.columns:
        df["current_speed"] = pd.to_numeric(df["current_speed"], errors="coerce").fillna(0.0)
    else:
        df["current_speed"] = 0.0

    # Sort by frame for correct trail order
    if "frame" in df.columns:
        df = df.sort_values("frame")

    def _build_trail(sub: pd.DataFrame) -> List[Dict]:
        """Extract x_pitch/y_pitch trail for a player, capped at 200 points for rendering."""
        xs = sub["x_pitch"].dropna().tolist() if "x_pitch" in sub.columns else []
        ys = sub["y_pitch"].dropna().tolist() if "y_pitch" in sub.columns else []
        if not xs or not ys:
            return []
        step = max(1, len(xs) // 200)  # downsample large trails
        return [{"x": float(x), "y": float(y)} for x, y in zip(xs[::step], ys[::step])]

    def _build_speed_history(sub: pd.DataFrame) -> List[Dict]:
        """Extract per-frame speed history for the speed timeline chart."""
        if "frame" not in sub.columns:
            return []
        step = max(1, len(sub) // 500)  # downsample
        return [
            {"frame": int(row["frame"]), "speed_kmh": float(row["current_speed"])}
            for _, row in sub.iloc[::step].iterrows()
        ]

    records = []
    for (pid, team), sub in df.groupby(["player_id", "team"]):
        records.append({
            "player_id":     int(pid),
            "team":          str(team),
            "max_speed":     round(float(sub["current_speed"].max()), 2),
            "avg_speed":     round(float(sub["current_speed"].mean()), 2),
            "frame_count":   int(len(sub)),
            "trail":         _build_trail(sub),
            "speed_history": _build_speed_history(sub),
        })

    return pd.DataFrame(records).sort_values("player_id").reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# Page renderer
# ─────────────────────────────────────────────────────────────────────────────

def render_player_analysis() -> None:
    """Render the Player Analysis page."""
    st.markdown(
        """
        <div class="page-header">
            <div class="page-title">👤 Player Analysis</div>
            <div class="page-subtitle">Individual player statistics, movement trails, and performance profiles</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    detections = load_detections()
    analytics  = load_analytics()

    # ── Guard: need at least detections.json ─────────────────────────────────
    if not detections:
        st.markdown(
            '<div class="warning-card"><span class="warning-icon">⚠️</span>'
            '<div><strong>detections.json not found.</strong><br>'
            '<small>Expected at <code>data/detections.json</code>. '
            'Run the pipeline to generate it.</small></div></div>',
            unsafe_allow_html=True,
        )
        return

    player_df = _build_player_df(detections)

    if player_df.empty:
        st.markdown(
            '<div class="warning-card"><span class="warning-icon">⚠️</span>'
            '<div><strong>No player detections found in detections.json.</strong><br>'
            '<small>Make sure the file contains player-class entries.</small></div></div>',
            unsafe_allow_html=True,
        )
        return

    # ── Analytics highlights (top_distance_players, etc.) ────────────────────
    top_dist_map: Dict[int, float] = {}
    if analytics:
        for entry in analytics.get("top_distance_players", []):
            pid = entry.get("player_id")
            dist = entry.get("distance_meters", 0.0)
            if pid is not None:
                top_dist_map[int(pid)] = float(dist)

    # ── Player selector ───────────────────────────────────────────────────────
    section_header("Player Roster", "📋")

    all_teams    = sorted(player_df["team"].unique().tolist())
    team_filter  = st.selectbox("Filter by team", ["All"] + all_teams, key="pa_team_filter")

    filtered_df = player_df if team_filter == "All" else player_df[player_df["team"] == team_filter]

    if filtered_df.empty:
        st.info("No players for the selected team.")
        return

    # Show a concise summary table first
    display_df = filtered_df[["player_id", "team", "max_speed", "avg_speed", "frame_count"]].copy()
    display_df.columns = ["Player ID", "Team", "Max Speed (km/h)", "Avg Speed (km/h)", "Detections"]

    st.dataframe(display_df, use_container_width=True, hide_index=True)

    # Player detail selector
    player_ids  = filtered_df["player_id"].tolist()
    selected_id = st.selectbox(
        "Select player for detailed view",
        options=player_ids,
        format_func=lambda pid: f"Player {pid} ({filtered_df[filtered_df['player_id'] == pid]['team'].values[0]})",
        key="pa_select",
    )

    selected_row = filtered_df[filtered_df["player_id"] == selected_id].iloc[0]

    # ── Player Detail ─────────────────────────────────────────────────────────
    st.markdown("<hr>", unsafe_allow_html=True)
    pid   = int(selected_row["player_id"])
    team  = str(selected_row["team"])
    color = _TEAM_COLORS.get(team, _DEFAULT_COLOR)
    r, g, b_ = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)

    st.markdown(
        f"""
        <div class="glass-card" style="border-left: 4px solid {color}; margin-bottom:1.5rem;">
            <div style="display:flex; align-items:center; gap:1rem;">
                <div style="width:60px; height:60px; border-radius:50%;
                            background:rgba({r},{g},{b_},0.2);
                            border:2px solid {color};
                            display:flex; align-items:center; justify-content:center;
                            font-family:'Rajdhani',sans-serif; font-size:1.4rem;
                            font-weight:700; color:{color};">
                    {pid}
                </div>
                <div>
                    <div style="font-family:'Rajdhani',sans-serif; font-size:1.5rem;
                                font-weight:700; color:#F1F5F9;">
                        Player {pid}
                    </div>
                    <div style="background:rgba({r},{g},{b_},0.15); border:1px solid rgba({r},{g},{b_},0.3);
                                border-radius:4px; padding:2px 8px; font-size:0.75rem;
                                display:inline-block; color:{color}; font-weight:600;">
                        {team}
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── KPI Cards ─────────────────────────────────────────────────────────────
    avg_spd  = float(selected_row["avg_speed"])
    max_spd  = float(selected_row["max_speed"])
    det_cnt  = int(selected_row["frame_count"])
    # distance from analytics if available, else show detection count
    dist_val = top_dist_map.get(pid, None)

    kpis = [
        {"icon": "💨", "label": "Average Speed",  "value": fmt_speed(avg_spd),
         "accent": "linear-gradient(90deg,#3B82F6,#8B5CF6)", "delta_color": "#3B82F6"},
        {"icon": "⚡", "label": "Maximum Speed",  "value": fmt_speed(max_spd),
         "accent": "linear-gradient(90deg,#F59E0B,#EF4444)", "delta_color": "#F59E0B"},
        {"icon": "🎞️", "label": "Detected Frames", "value": str(det_cnt),
         "accent": "linear-gradient(90deg,#14B8A6,#3B82F6)", "delta_color": "#14B8A6"},
    ]
    if dist_val is not None:
        kpis.insert(0, {
            "icon": "📏", "label": "Distance Covered", "value": fmt_distance(dist_val),
            "accent": "linear-gradient(90deg,#00FF87,#14B8A6)", "delta_color": "#00FF87",
        })

    metric_row(kpis)

    # ── Charts ────────────────────────────────────────────────────────────────
    st.markdown("<div style='height:1rem;'></div>", unsafe_allow_html=True)
    col1, col2 = st.columns([1, 1])

    # Radar Chart — use relative-to-max-in-dataset normalisation
    with col1:
        section_header("Performance Profile", "📡")
        max_vals = {
            "max_speed":   float(player_df["max_speed"].max()) or 1.0,
            "avg_speed":   float(player_df["avg_speed"].max()) or 1.0,
            "frame_count": float(player_df["frame_count"].max()) or 1.0,
            # distance uses analytics data if available
            "distance":    float(max(top_dist_map.values())) if top_dist_map else 1.0,
        }
        # Build a compatible dict for player_radar (expects old-style keys)
        radar_data = {
            "player_id":     pid,
            "max_speed_kmh": max_spd,
            "avg_speed_kmh": avg_spd,
            "distance_m":    dist_val if dist_val else 0.0,
            "touches":       0,         # not available in current schema
            "possession_pct": 0.0,
        }
        radar_maxvals = {
            "max_speed_kmh": max_vals["max_speed"],
            "avg_speed_kmh": max_vals["avg_speed"],
            "distance_m":    max_vals["distance"],
            "touches":       1.0,
            "possession_pct": 1.0,
        }
        fig_radar = player_radar(radar_data, pid, radar_maxvals)
        st.plotly_chart(fig_radar, use_container_width=True)

    # Movement Trail
    with col2:
        section_header("Movement Trail", "🗺️")
        trail = selected_row.get("trail", [])
        if not isinstance(trail, list):
            trail = []

        if trail:
            fig_trail = player_trail_chart(trail, pid, team)
            st.plotly_chart(fig_trail, use_container_width=True)
        else:
            # Synthetic fallback for visual completeness
            np.random.seed(pid % 9999)
            x_base = 20 + (pid % 65)
            y_base = 10 + (pid % 48)
            n = 80
            synth_trail = [
                {
                    "x": float(np.clip(x_base + np.cumsum(np.random.randn(n))[i], 2, 103)),
                    "y": float(np.clip(y_base + np.cumsum(np.random.randn(n))[i], 2, 66)),
                }
                for i in range(n)
            ]
            fig_trail = player_trail_chart(synth_trail, pid, team)
            st.plotly_chart(fig_trail, use_container_width=True)
            st.caption("📍 Simulated trail (pitch coords not available for this player)")

    # ── Speed Timeline ─────────────────────────────────────────────────────────
    speed_history: List[Dict] = selected_row.get("speed_history", [])
    if not isinstance(speed_history, list):
        speed_history = []

    st.markdown("<hr>", unsafe_allow_html=True)

    if speed_history:
        section_header("Speed Timeline", "📈")
        frames_h = [s.get("frame", i) for i, s in enumerate(speed_history)]
        speeds_h = [s.get("speed_kmh", 0.0) for s in speed_history]
        fig_spd  = speed_timeline(frames_h, speeds_h, pid, team)
        st.plotly_chart(fig_spd, use_container_width=True)
    else:
        section_header("Speed Timeline (Demo)", "📈")
        np.random.seed(pid % 9999)
        frames_h = list(range(0, det_cnt * 10, 10))[:500]
        speeds_h = [
            max(0.0, avg_spd + np.random.randn() * 2.5 + 1.5 * np.sin(f / 80))
            for f in frames_h
        ]
        fig_spd = speed_timeline(frames_h, speeds_h, pid, team)
        st.plotly_chart(fig_spd, use_container_width=True)
        st.caption("⚡ Simulated speed curve for demonstration")

    # ── Extended stats ─────────────────────────────────────────────────────────
    st.markdown("<hr>", unsafe_allow_html=True)
    with st.expander("📄 Full Player Statistics", expanded=False):
        for k, v in selected_row.items():
            if k in ("trail", "speed_history"):
                continue
            label = k.replace("_", " ").title()
            st.markdown(
                f"""
                <div class="stat-row">
                    <span class="stat-key">{label}</span>
                    <span class="stat-value">{v}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
