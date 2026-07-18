"""
dashboard/heatmaps.py
---------------------
Heatmap page: displays team heatmaps (custom-uploaded or live-computed) and
the ball heatmap, with zoom controls using Plotly image/histogram traces.
"""
 
from pathlib import Path
from typing import List, Dict, Optional
 
import numpy as np
import streamlit as st
from PIL import Image
import plotly.graph_objects as go
 
from config import BLUE_TEAM_HEATMAP, RED_TEAM_HEATMAP, BALL_HEATMAP, PITCH_LENGTH, PITCH_WIDTH
from utils import load_image, check_file, load_detections, save_uploaded_file
from components.metrics import section_header
 
 
def _add_pitch_markings(fig: go.Figure) -> None:
    """
    Draw standard pitch lines (outer rectangle, halfway line, centre circle,
    goal boxes) onto an existing Plotly figure. Shared by both the static
    image-based heatmaps and the live histogram-based team heatmaps so all
    heatmaps look visually consistent.
    """
    fig.add_shape(type="rect", x0=0, y0=0, x1=PITCH_LENGTH, y1=PITCH_WIDTH,
                  line=dict(color="rgba(255,255,255,0.5)", width=2))
    fig.add_shape(type="line", x0=PITCH_LENGTH/2, y0=0, x1=PITCH_LENGTH/2, y1=PITCH_WIDTH,
                  line=dict(color="rgba(255,255,255,0.4)", width=1, dash="dash"))
 
    theta = np.linspace(0, 2*np.pi, 60)
    cx, cy = PITCH_LENGTH/2, PITCH_WIDTH/2
    fig.add_trace(go.Scatter(
        x=cx + 9.15 * np.cos(theta),
        y=cy + 9.15 * np.sin(theta),
        mode="lines",
        line=dict(color="rgba(255,255,255,0.35)", width=1),
        showlegend=False, hoverinfo="skip",
    ))
 
    for gx in [0, PITCH_LENGTH]:
        fig.add_shape(type="rect",
                      x0=gx - 5.5 if gx > 0 else gx,
                      y0=(PITCH_WIDTH - 18.32) / 2,
                      x1=gx + 5.5 if gx == 0 else gx,
                      y1=(PITCH_WIDTH + 18.32) / 2,
                      line=dict(color="rgba(255,255,255,0.3)", width=1))
 
 
def _base_pitch_layout(fig: go.Figure, title: str, height: int) -> None:
    """Apply the shared dark-theme pitch layout settings to a figure."""
    fig.update_xaxes(
        range=[0, PITCH_LENGTH], showgrid=False, zeroline=False,
        title="Length (m)", color="#64748B",
    )
    fig.update_yaxes(
        range=[0, PITCH_WIDTH], showgrid=False, zeroline=False,
        title="Width (m)", color="#64748B", scaleanchor="x",
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(17,24,39,0)",
        plot_bgcolor="rgba(13,19,33,0.9)",
        font=dict(color="#94A3B8", family="Inter, sans-serif"),
        margin=dict(l=16, r=16, t=40, b=32),
        title=dict(text=title, font=dict(size=15, color="#CBD5E1")),
        height=height,
        dragmode="zoom",
    )
 
 
def _heatmap_fig(img: Image.Image, title: str, height: int = 560) -> go.Figure:
    """Wrap a PIL heatmap image inside a zoomable/pannable Plotly figure."""
    fig = go.Figure()
    fig.add_layout_image(
        dict(
            source=img,
            xref="x", yref="y",
            x=0, y=PITCH_WIDTH,
            sizex=PITCH_LENGTH, sizey=PITCH_WIDTH,
            sizing="stretch",
            layer="below",
        )
    )
    _add_pitch_markings(fig)
    _base_pitch_layout(fig, title, height)
    return fig
 
 
def _team_heatmap_fig(
    detections: List[Dict],
    team: str,
    colorscale: str,
    title: str,
    height: int = 560,
) -> go.Figure:
    """
    Build a live positional-density heatmap for one team, computed directly
    from detections.json's x_pitch/y_pitch coordinates.
    """
    xs = [d.get("x_pitch") for d in detections
          if d.get("team") == team and d.get("class") == "player" and d.get("x_pitch") is not None]
    ys = [d.get("y_pitch") for d in detections
          if d.get("team") == team and d.get("class") == "player" and d.get("y_pitch") is not None]
 
    fig = go.Figure()
    if xs and ys:
        fig.add_trace(go.Histogram2d(
            x=xs, y=ys,
            colorscale=colorscale,
            nbinsx=40, nbinsy=26,
            opacity=0.85,
            showscale=False,
            zsmooth="best",
        ))
 
    _add_pitch_markings(fig)
    _base_pitch_layout(fig, title, height)
    return fig
 
 
def _team_avg_position_insight(detections: List[Dict]) -> str:
    """Short, safe comparison insight between the two teams' average pitch position."""
    blue_xs = [d.get("x_pitch") for d in detections
               if d.get("team") == "White Team" and d.get("class") == "player" and d.get("x_pitch") is not None]
    red_xs = [d.get("x_pitch") for d in detections
              if d.get("team") == "Red Team" and d.get("class") == "player" and d.get("x_pitch") is not None]
 
    if not blue_xs or not red_xs:
        return "Not enough data yet to compare team positioning."
 
    blue_avg = sum(blue_xs) / len(blue_xs)
    red_avg = sum(red_xs) / len(red_xs)
    diff = abs(blue_avg - red_avg)
    leading_team = "White Team" if blue_avg > red_avg else "Red Team"
 
    return (
        f"💡 <strong>Insight:</strong> On average, <strong>{leading_team}</strong> maintained a more "
        f"advanced pitch position (avg. length difference of <strong>{diff:.1f}m</strong>), "
        f"suggesting a more attacking or higher defensive line during the match."
    )
 
 
def _render_team_heatmap_column(
    team_label: str,
    team_key: str,
    custom_path: Path,
    colorscale: str,
    detections: List[Dict],
    height: int,
) -> None:
    """
    Render one team's heatmap column: a custom-uploaded PNG if present,
    otherwise the live-computed histogram from detections.json. Includes
    an uploader and a "revert to live" button.
    """
    st.markdown(
        f'<div class="section-title" style="font-size:0.85rem;">{team_label}</div>',
        unsafe_allow_html=True,
    )
 
    custom_img = load_image(custom_path)
 
    if custom_img:
        st.caption("📌 Custom uploaded heatmap")
        fig = _heatmap_fig(custom_img, f"{team_label} — Custom Heatmap", height=height)
        st.plotly_chart(fig, use_container_width=True)
        if st.button(f"↩ Revert to live-computed heatmap", key=f"revert_{team_key}"):
            custom_path.unlink(missing_ok=True)
            st.rerun()
    else:
        st.caption("⚡ Live-computed from detections.json")
        fig = _team_heatmap_fig(detections, team_label, colorscale,
                                 f"{team_label} — Positional Density", height=height)
        st.plotly_chart(fig, use_container_width=True)
 
    uploaded = st.file_uploader(
        f"Upload custom {team_label} heatmap PNG",
        type=["png", "jpg", "jpeg"],
        key=f"upload_{team_key}",
    )
    if uploaded is not None:
        save_uploaded_file(uploaded, custom_path)
        st.success(f"✅ Custom {team_label} heatmap saved.")
        st.rerun()
 
 
def render_heatmaps() -> None:
    """Render the Heatmaps page."""
    st.markdown(
        """
        <div class="page-header">
            <div class="page-title">🌡️ Heatmaps</div>
            <div class="page-subtitle">
                Positional density heatmaps for both teams and the ball across the match.
                Use scroll to zoom · click &amp; drag to pan. Upload your own heatmap image
                per team to override the live-computed one.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
 
    detections = load_detections() or []
    ball_img = load_image(BALL_HEATMAP)
 
    tab_teams, tab_ball = st.tabs(["⬜🔴 Team Heatmaps", "⚽ Ball Heatmap"])
 
    # ── Team Heatmaps Tab ────────────────────────────────────────────────────
    with tab_teams:
        section_header("White vs Red — Positional Density", "⬜🔴")
 
        if not detections:
            st.markdown(
                '<p style="color:#64748B; font-size:0.83rem; margin-bottom:0.5rem;">'
                'No detections.json found yet — live-computed heatmaps need this file. '
                'You can still upload a custom PNG heatmap for each team below.</p>',
                unsafe_allow_html=True,
            )
 
        height_t = st.select_slider(
            "Chart height",
            options=[400, 480, 560, 640],
            value=560,
            key="heatmap_team_height",
        )
 
        col_blue, col_red = st.columns(2)
        with col_blue:
            _render_team_heatmap_column(
                "White Team", "white", BLUE_TEAM_HEATMAP, "Blues", detections, height_t,
            )
        with col_red:
            _render_team_heatmap_column(
                "Red Team", "red", RED_TEAM_HEATMAP, "Reds", detections, height_t,
            )
 
        if detections:
            st.markdown(
                f"""
                <div class="insight-card info" style="margin-top:1rem;">
                    {_team_avg_position_insight(detections)}
                </div>
                """,
                unsafe_allow_html=True,
            )
 
    # ── Ball Heatmap Tab ─────────────────────────────────────────────────────
    with tab_ball:
        section_header("Ball Position Density", "⚽")
        if ball_img:
            st.markdown(
                '<p style="color:#64748B; font-size:0.83rem; margin-bottom:0.5rem;">'
                'Brighter zones indicate where the ball spent the most time. Generated by Member 2.</p>',
                unsafe_allow_html=True,
            )
            col_opt2, _ = st.columns([2, 3])
            with col_opt2:
                height_b = st.select_slider(
                    "Chart height",
                    options=[400, 480, 560, 640],
                    value=560,
                    key="heatmap_b_height",
                )
            fig_b = _heatmap_fig(ball_img, "Ball Position Heatmap", height=height_b)
            st.plotly_chart(fig_b, use_container_width=True)
 
            with st.expander("🖼️ View Raw Image"):
                st.image(str(BALL_HEATMAP), use_container_width=True,
                         caption="ball_heatmap.png (raw)")
        else:
            check_file(BALL_HEATMAP, "Ball Heatmap")
 
