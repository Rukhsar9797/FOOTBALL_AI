"""
components/charts.py
--------------------
Plotly-based chart builders:
 - Donut / pie charts
 - Bar charts (team comparison, player ranking)
 - Scatter / trail charts
 - Speed timeline charts
All charts use the dark custom theme from config.py.
"""

from typing import Dict, List, Optional
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

from config import (
    COLORS,
    TEAM_COLORS,
    PLOTLY_TEMPLATE,
    CHART_BG,
    CHART_PAPER_BG,
    CHART_FONT_COLOR,
    CHART_GRID_COLOR,
    PITCH_LENGTH,
    PITCH_WIDTH,
)

# ─────────────────────────────────────────────────────────────────────────────
# SHARED LAYOUT DEFAULTS
# ─────────────────────────────────────────────────────────────────────────────

_LAYOUT = dict(
    template=PLOTLY_TEMPLATE,
    paper_bgcolor=CHART_PAPER_BG,
    plot_bgcolor=CHART_BG,
    font=dict(color=CHART_FONT_COLOR, family="Inter, sans-serif"),
    margin=dict(l=16, r=16, t=40, b=16),
    legend=dict(
        bgcolor="rgba(26,35,50,0.7)",
        bordercolor="#1E293B",
        borderwidth=1,
        font=dict(size=11),
    ),
)


def _apply_grid(fig: go.Figure) -> go.Figure:
    fig.update_xaxes(gridcolor=CHART_GRID_COLOR, zerolinecolor=CHART_GRID_COLOR)
    fig.update_yaxes(gridcolor=CHART_GRID_COLOR, zerolinecolor=CHART_GRID_COLOR)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# DONUT CHART — Possession
# ─────────────────────────────────────────────────────────────────────────────

def possession_donut(team1_pct: float, team2_pct: float) -> go.Figure:
    """Animated possession donut chart."""
    fig = go.Figure(
        go.Pie(
            labels=["Team 1", "Team 2"],
            values=[team1_pct, team2_pct],
            hole=0.65,
            marker=dict(
                colors=["#FFFFFF", "#EF4444"],  # Team 1: White, Team 2: Red
                line=dict(color="#0A0E1A", width=3),
            ),
            textinfo="percent",
            textfont=dict(size=13, color="white", family="Rajdhani, sans-serif"),
            hovertemplate="<b>%{label}</b><br>Possession: %{value:.1f}%<extra></extra>",
            pull=[0.04, 0.04],
        )
    )
    fig.update_layout(
        **_LAYOUT,
        title=dict(text="Ball Possession", font=dict(size=14, color=CHART_FONT_COLOR)),
        annotations=[
            dict(
                text=f"<b>{max(team1_pct, team2_pct):.0f}%</b>",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=26, color="#F1F5F9", family="Rajdhani, sans-serif"),
            )
        ],
        height=280,
        showlegend=True,
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# TEAM COMPARISON BAR CHART
# ─────────────────────────────────────────────────────────────────────────────

def team_comparison_bar(
    categories: List[str],
    blue_values: List[float],
    red_values: List[float],
    title: str = "Team Statistics",
    y_label: str = "",
) -> go.Figure:
    """Grouped horizontal bar chart comparing two teams."""
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Team 1",
        y=categories,
        x=blue_values,
        orientation="h",
        marker=dict(
            color="#FFFFFF",  # Team 1: White
            opacity=0.85,
            line=dict(color="#CBD5E1", width=1),
        ),
        hovertemplate="<b>Team 1</b><br>%{y}: %{x:.1f}<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        name="Team 2",
        y=categories,
        x=red_values,
        orientation="h",
        marker=dict(
            color="#EF4444",  # Team 2: Red
            opacity=0.85,
            line=dict(color="#DC2626", width=1),
        ),
        hovertemplate="<b>Team 2</b><br>%{y}: %{x:.1f}<extra></extra>",
    ))
    fig.update_layout(
        **_LAYOUT,
        barmode="group",
        title=dict(text=title, font=dict(size=14, color=CHART_FONT_COLOR)),
        xaxis_title=y_label,
        height=340,
    )
    return _apply_grid(fig)


# ─────────────────────────────────────────────────────────────────────────────
# PLAYER RANKING BAR
# ─────────────────────────────────────────────────────────────────────────────

def player_ranking_bar(
    df: pd.DataFrame,
    value_col: str,
    title: str,
    color_col: str = "team",
    top_n: int = 10,
    unit: str = "",
) -> go.Figure:
    """Horizontal bar chart ranking top N players."""
    df = df.nlargest(top_n, value_col).copy()
    df["color"] = df[color_col].map(TEAM_COLORS).fillna(COLORS["accent_teal"])
    df["label"] = df.apply(
        lambda r: f"P{r['player_id']} ({r.get('team','?')[:3].upper()})", axis=1
    )

    fig = go.Figure(
        go.Bar(
            y=df["label"],
            x=df[value_col],
            orientation="h",
            marker=dict(color=df["color"], opacity=0.9),
            text=df[value_col].apply(lambda v: f"{v:.1f}{unit}"),
            textposition="outside",
            textfont=dict(color="#94A3B8", size=11),
            hovertemplate="<b>%{y}</b><br>" + title + ": %{x:.2f}" + unit + "<extra></extra>",
        )
    )
    fig.update_layout(
        **_LAYOUT,
        title=dict(text=title, font=dict(size=14, color=CHART_FONT_COLOR)),
        height=max(300, top_n * 32),
        yaxis=dict(autorange="reversed"),
    )
    return _apply_grid(fig)


# ─────────────────────────────────────────────────────────────────────────────
# PLAYER MOVEMENT TRAIL
# ─────────────────────────────────────────────────────────────────────────────

def player_trail_chart(trail_data: List[Dict], player_id: int, team: str) -> go.Figure:
    """
    Scatter-line chart of a player's movement trail on a pitch outline.
    trail_data: list of {x, y} pixel coords.
    """
    if not trail_data:
        return _empty_fig("No trail data available")

    xs = [d["x"] for d in trail_data]
    ys = [d["y"] for d in trail_data]
    color = TEAM_COLORS.get(team, COLORS["accent_teal"])

    fig = go.Figure()

    # Pitch rectangle outline
    fig.add_shape(type="rect", x0=0, y0=0, x1=PITCH_LENGTH, y1=PITCH_WIDTH,
                  line=dict(color="#334155", width=2), fillcolor="rgba(30,41,59,0.3)")
    # Centre line
    fig.add_shape(type="line", x0=PITCH_LENGTH/2, y0=0, x1=PITCH_LENGTH/2, y1=PITCH_WIDTH,
                  line=dict(color="#334155", width=1, dash="dash"))
    # Centre circle (approximate)
    theta = np.linspace(0, 2*np.pi, 60)
    cx, cy = PITCH_LENGTH/2, PITCH_WIDTH/2
    r = 9.15
    fig.add_trace(go.Scatter(
        x=cx + r * np.cos(theta), y=cy + r * np.sin(theta),
        mode="lines", line=dict(color="#334155", width=1), showlegend=False, hoverinfo="skip",
    ))

    # Trail gradient line
    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode="lines+markers",
        line=dict(color=color, width=2.5, shape="spline"),
        marker=dict(
            size=[4] * (len(xs) - 1) + [12],
            color=[f"rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},{i/len(xs):.2f})"
                   for i in range(1, len(xs)+1)],
            symbol=["circle"] * (len(xs) - 1) + ["star"],
        ),
        name=f"Player {player_id}",
        hovertemplate="X: %{x:.1f}m<br>Y: %{y:.1f}m<extra></extra>",
    ))

    fig.update_layout(
        **_LAYOUT,
        title=dict(text=f"Movement Trail — Player {player_id}", font=dict(size=14)),
        xaxis=dict(range=[0, PITCH_LENGTH], title="Length (m)", showgrid=False),
        yaxis=dict(range=[0, PITCH_WIDTH], title="Width (m)", showgrid=False, scaleanchor="x"),
        height=380,
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# SPEED TIMELINE
# ─────────────────────────────────────────────────────────────────────────────

def speed_timeline(
    frames: List[int],
    speeds: List[float],
    player_id: int,
    team: str,
) -> go.Figure:
    """Line chart of speed over frames."""
    color = TEAM_COLORS.get(team, COLORS["accent_teal"])
    fig = go.Figure(
        go.Scatter(
            x=frames,
            y=speeds,
            mode="lines",
            line=dict(color=color, width=2, shape="spline"),
            fill="tozeroy",
            fillcolor=f"rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.15)",
            name=f"Player {player_id}",
            hovertemplate="Frame %{x}<br>Speed: %{y:.1f} km/h<extra></extra>",
        )
    )
    fig.update_layout(
        **_LAYOUT,
        title=dict(text=f"Speed Over Time — Player {player_id}", font=dict(size=14)),
        xaxis_title="Frame",
        yaxis_title="Speed (km/h)",
        height=260,
    )
    return _apply_grid(fig)


# ─────────────────────────────────────────────────────────────────────────────
# PLAYER SCATTER ON PITCH
# ─────────────────────────────────────────────────────────────────────────────

def player_positions_scatter(frame_detections: List[Dict]) -> go.Figure:
    """Scatter of all player positions in a given frame on a pitch."""
    if not frame_detections:
        return _empty_fig("No detections in this frame")

    fig = go.Figure()
    # Pitch background
    fig.add_shape(type="rect", x0=0, y0=0, x1=PITCH_LENGTH, y1=PITCH_WIDTH,
                  line=dict(color="#334155", width=2), fillcolor="rgba(15,30,15,0.6)")
    fig.add_shape(type="line", x0=PITCH_LENGTH/2, y0=0, x1=PITCH_LENGTH/2, y1=PITCH_WIDTH,
                  line=dict(color="#334155", width=1, dash="dash"))
    theta = np.linspace(0, 2*np.pi, 60)
    cx, cy = PITCH_LENGTH/2, PITCH_WIDTH/2
    fig.add_trace(go.Scatter(
        x=cx + 9.15 * np.cos(theta), y=cy + 9.15 * np.sin(theta),
        mode="lines", line=dict(color="#334155", width=1),
        showlegend=False, hoverinfo="skip",
    ))

    # Plot per team
    teams_done = set()
    for det in frame_detections:
        team = det.get("team", "Unknown")
        cls  = det.get("class", "player")
        x    = det.get("x_pitch", det.get("x", 0))
        y    = det.get("y_pitch", det.get("y", 0))
        pid  = det.get("player_id", "?")
        color = TEAM_COLORS.get(team, COLORS["accent_teal"])
        sym = "circle" if cls == "player" else ("diamond" if cls == "referee" else "circle-open")

        fig.add_trace(go.Scatter(
            x=[x], y=[y],
            mode="markers+text",
            marker=dict(size=14 if cls=="ball" else 12, color=color, symbol=sym,
                        line=dict(width=2, color="#0A0E1A")),
            text=[str(pid)],
            textfont=dict(size=8, color="white"),
            textposition="top center",
            name=team if team not in teams_done else None,
            showlegend=team not in teams_done,
            hovertemplate=f"<b>{'Ball' if cls=='ball' else f'Player {pid}'}</b><br>Team: {team}<br>X: {x:.1f}m  Y: {y:.1f}m<extra></extra>",
        ))
        teams_done.add(team)

    fig.update_layout(
        **_LAYOUT,
        xaxis=dict(range=[-2, PITCH_LENGTH+2], title="Length (m)", showgrid=False),
        yaxis=dict(range=[-2, PITCH_WIDTH+2], title="Width (m)", showgrid=False, scaleanchor="x"),
        title=dict(text="Player Positions", font=dict(size=14)),
        height=400,
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# RADAR CHART — player comparison
# ─────────────────────────────────────────────────────────────────────────────

def player_radar(stats: Dict, player_id: int, max_vals: Dict) -> go.Figure:
    """Radar chart for a single player's normalised stats."""
    cats = ["Distance", "Avg Speed", "Max Speed", "Touches", "Possession"]
    keys = ["distance_m", "avg_speed_kmh", "max_speed_kmh", "touches", "possession_pct"]
    vals = []
    for k in keys:
        v = stats.get(k, 0)
        m = max_vals.get(k, 1) or 1
        vals.append(min(v / m, 1.0) * 100)

    team = stats.get("team", "Unknown")
    color = TEAM_COLORS.get(team, COLORS["accent_teal"])

    fig = go.Figure(go.Scatterpolar(
        r=vals + [vals[0]],
        theta=cats + [cats[0]],
        fill="toself",
        line=dict(color=color, width=2),
        fillcolor=f"rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.2)",
        name=f"Player {player_id}",
    ))
    fig.update_layout(
        **_LAYOUT,
        polar=dict(
            bgcolor="rgba(17,24,39,0)",
            radialaxis=dict(visible=True, range=[0,100], color="#334155", gridcolor="#1E293B"),
            angularaxis=dict(color="#64748B", gridcolor="#1E293B"),
        ),
        title=dict(text=f"Player {player_id} — Performance Profile", font=dict(size=14)),
        height=360,
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# HELPER
# ─────────────────────────────────────────────────────────────────────────────

def _empty_fig(msg: str) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        **_LAYOUT,
        annotations=[dict(
            text=msg, x=0.5, y=0.5, showarrow=False,
            font=dict(color="#475569", size=14),
        )],
        height=300,
    )
    return fig