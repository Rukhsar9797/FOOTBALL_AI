"""
components/video_player.py
--------------------------
Video playback and frame inspection component.
Reads the tracked_video.mp4 and overlays detection metadata.
"""

import io
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import streamlit as st
from PIL import Image

from config import TRACKED_VIDEO, VIDEO_FPS, TEAM_COLORS
from utils import check_file
from components.video_compare import generate_comparison_video

from config import (
    ORIGINAL_VIDEO,
    TRACKED_VIDEO,
    COMPARISON_VIDEO,
)

def render_video_player():

    if not check_file(ORIGINAL_VIDEO, "Original Video"):
        return

    if not check_file(TRACKED_VIDEO, "Tracked Video"):
        return

    if (
        not COMPARISON_VIDEO.exists()
        or COMPARISON_VIDEO.stat().st_mtime
        < TRACKED_VIDEO.stat().st_mtime
    ):
        with st.spinner("Generating comparison video..."):
            generate_comparison_video()

    st.video("outputs/comparison.mp4")


def render_frame_inspector(detections: Optional[List[Dict]]) -> None:
    """
    Frame-by-frame inspector using OpenCV.
    Displays the selected frame with detection bounding boxes overlaid.
    """
    if not check_file(TRACKED_VIDEO, "Tracked Match Video"):
        return

    cap = cv2.VideoCapture(str(TRACKED_VIDEO))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or VIDEO_FPS

    if total_frames <= 0:
        st.warning("Could not read video properties.")
        cap.release()
        return

    st.markdown("#### 🎞️ Frame Inspector")

    frame_idx = st.slider(
        "Seek Frame",
        min_value=0,
        max_value=total_frames - 1,
        value=0,
        step=1,
        key="frame_slider",
    )

    timestamp = frame_idx / fps
    mm = int(timestamp // 60)
    ss = int(timestamp % 60)
    st.markdown(
        f"<div style='color:#64748B; font-size:0.8rem; margin-bottom:0.5rem;'>"
        f"⏱ {mm:02d}:{ss:02d} &nbsp;|&nbsp; Frame {frame_idx} / {total_frames}</div>",
        unsafe_allow_html=True,
    )

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        st.error("Failed to read frame.")
        return

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Overlay bounding boxes from detections
    if detections:
        frame_dets = [d for d in detections if d.get("frame") == frame_idx]
        for det in frame_dets:
            bbox = det.get("bbox", [])
            if len(bbox) == 4:
                x1, y1, x2, y2 = [int(v) for v in bbox]
                team  = det.get("team", "Unknown")
                pid   = det.get("player_id", "?")
                cls   = det.get("class", "player")
                color_hex = TEAM_COLORS.get(team, "#6B7280")
                r, g, b = (
                    int(color_hex[1:3], 16),
                    int(color_hex[3:5], 16),
                    int(color_hex[5:7], 16),
                )
                cv2.rectangle(frame, (x1, y1), (x2, y2), (r, g, b), 2)
                label = "BALL" if cls == "ball" else f"P{pid}"
                cv2.putText(
                    frame, label, (x1, max(y1 - 6, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (r, g, b), 2,
                )

    pil_img = Image.fromarray(frame)
    st.image(pil_img, use_container_width=True)

    return frame_idx


def get_frame_detections(detections: Optional[List[Dict]], frame_idx: int) -> List[Dict]:
    """Filter detections to a specific frame."""
    if not detections:
        return []
    return [d for d in detections if d.get("frame") == frame_idx]
