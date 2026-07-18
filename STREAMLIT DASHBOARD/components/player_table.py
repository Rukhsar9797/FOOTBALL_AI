"""
components/player_table.py
--------------------------
Reusable searchable, sortable player statistics table component.
"""

from typing import Optional
import pandas as pd
import streamlit as st

from config import TEAM_COLORS, COLORS
from utils import fmt_speed, fmt_distance, fmt_pct


COLUMN_LABELS = {
    "player_id":             "ID",
    "team":                  "Team",
    "distance_m":            "Distance",
    "avg_speed_kmh":         "Avg Speed",
    "max_speed_kmh":         "Max Speed",
    "touches":               "Touches",
    "possession_pct":        "Possession %",
    "attack_momentum_score": "Att. Momentum",
}


def render_player_table(
    df: pd.DataFrame,
    search_key: str = "player_search",
    sort_key: str = "player_sort",
    filter_key: str = "player_filter",
) -> Optional[pd.Series]:
    """
    Renders an interactive player table with search, sort, filter, and row selection.
    Returns the selected player's row as a Series, or None.
    """
    if df.empty:
        st.markdown(
            '<div class="warning-card"><span class="warning-icon">📭</span>'
            '<div>No player data available.</div></div>',
            unsafe_allow_html=True,
        )
        return None

    col1, col2, col3 = st.columns([2, 2, 2])
    with col1:
        query = st.text_input(
            "🔍 Search Player",
            placeholder="Player ID or team...",
            key=search_key,
        )
    with col2:
        teams = ["All Teams"] + sorted(list(df["team"].unique()))
        selected_team = st.selectbox(
            "Filter by Team",
            options=teams,
            key=filter_key,
        )
    with col3:
        # Include attack_momentum_score if it exists in the dataframe, otherwise omit it from sort options to prevent errors
        sort_options = ["distance_m", "avg_speed_kmh", "max_speed_kmh", "touches", "possession_pct"]
        if "attack_momentum_score" in df.columns:
            sort_options.append("attack_momentum_score")
            
        sort_col = st.selectbox(
            "Sort by",
            options=sort_options,
            format_func=lambda c: COLUMN_LABELS.get(c, c),
            key=sort_key,
        )

    # Filter by team
    filtered = df.copy()
    if selected_team != "All Teams":
        filtered = filtered[filtered["team"] == selected_team]

    # Filter by query
    if query:
        q = query.lower()
        mask = (
            filtered["player_id"].astype(str).str.contains(q) |
            filtered["team"].str.lower().str.contains(q)
        )
        filtered = filtered[mask]

    # Sort
    if sort_col in filtered.columns:
        filtered = filtered.sort_values(sort_col, ascending=False)

    # Format display copy
    display = filtered.copy()
    if "distance_m" in display.columns:
        display["distance_m"] = display["distance_m"].apply(fmt_distance)
    if "avg_speed_kmh" in display.columns:
        display["avg_speed_kmh"] = display["avg_speed_kmh"].apply(fmt_speed)
    if "max_speed_kmh" in display.columns:
        display["max_speed_kmh"] = display["max_speed_kmh"].apply(fmt_speed)
    if "possession_pct" in display.columns:
        display["possession_pct"] = display["possession_pct"].apply(fmt_pct)
    if "attack_momentum_score" in display.columns:
        # Assuming attack momentum is a float, let's round it for display
        display["attack_momentum_score"] = display["attack_momentum_score"].apply(lambda x: f"{x:.2f}" if pd.notnull(x) else "N/A")

    # Rename columns using our dictionary
    display = display.rename(columns=COLUMN_LABELS)

    st.dataframe(
        display,
        use_container_width=True,
        hide_index=True,
        height=min(40 + len(display) * 35, 500),
    )

    st.markdown(
        f"<div style='color:#475569; font-size:0.75rem; margin-top:4px;'>"
        f"Showing {len(filtered)} player(s)</div>",
        unsafe_allow_html=True,
    )

    # Player selector
    if not filtered.empty:
        ids = filtered["player_id"].tolist()
        selected_id = st.selectbox(
            "Select player for detailed view",
            options=ids,
            format_func=lambda i: f"Player {i} — {filtered[filtered['player_id']==i]['team'].values[0]}",
        )
        row = filtered[filtered["player_id"] == selected_id].iloc[0]
        return row

    return None
