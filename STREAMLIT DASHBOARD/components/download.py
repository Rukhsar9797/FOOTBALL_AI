"""
components/download.py
----------------------
Download buttons for tracked video, analytics JSON, and detections JSON.
"""

import json
from pathlib import Path
from typing import Optional, Dict, List

import streamlit as st

from config import TRACKED_VIDEO, ANALYTICS_JSON, DETECTIONS_JSON


def _read_bytes(path: Path) -> Optional[bytes]:
    if path.exists():
        return path.read_bytes()
    return None


def render_download_card(
    icon: str,
    title: str,
    description: str,
    file_path: Path,
    file_name: str,
    mime: str,
) -> None:
    """Render a styled download card with a download button."""
    exists = file_path.exists()
    status_color = "#00FF87" if exists else "#EF4444"
    status_label = "Available" if exists else "File Not Found"

    st.markdown(
        f"""
        <div class="glass-card" style="margin-bottom:0;">
            <div style="display:flex; align-items:flex-start; gap:1rem;">
                <span style="font-size:2.5rem;">{icon}</span>
                <div style="flex:1;">
                    <div style="font-family:'Rajdhani',sans-serif; font-size:1.1rem;
                                font-weight:700; color:#F1F5F9; margin-bottom:4px;">
                        {title}
                    </div>
                    <div style="color:#64748B; font-size:0.83rem; margin-bottom:8px;">
                        {description}
                    </div>
                    <div style="display:flex; align-items:center; gap:6px; font-size:0.75rem;">
                        <span style="width:8px; height:8px; border-radius:50%;
                                     background:{status_color}; display:inline-block;"></span>
                        <span style="color:{status_color};">{status_label}</span>
                        {"<span style='color:#334155; font-size:0.7rem;'>· " + str(file_path.name) + "</span>"
                          if exists else ""}
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if exists:
        data = _read_bytes(file_path)
        st.download_button(
            label=f"⬇️ Download {title}",
            data=data,
            file_name=file_name,
            mime=mime,
            use_container_width=True,
        )
    else:
        st.button(
            f"⚠️ {title} — Unavailable",
            disabled=True,
            use_container_width=True,
        )

    st.markdown("<div style='height:0.5rem;'></div>", unsafe_allow_html=True)
