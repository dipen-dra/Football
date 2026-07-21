import streamlit as st
import pandas as pd
import sys
import os

# Add workspace to path
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from src.app_helper import init_app, render_segmented_tabs
from src.viz import plot_regains_and_shots, plot_pressing_intensity_timeline, CARD_BG, TEXT_COLOR
from src.ingestion import get_match_events

def main():
    success = init_app("Match Detail", "Deconstruct a single match: visualize tactical timelines, possession regains, and pressing-to-shot maps.")
    if not success or 'matches_features' not in st.session_state:
        st.warning("Please load dataset in sidebar first.")
        return
        
    events_df = st.session_state['events_features']
    matches_df = st.session_state['matches_features']
    matches_meta = st.session_state['matches_df']
    
    # 1. Match Selection
    col_sel1, col_sel2 = st.columns(2)
    
    with col_sel1:
        all_teams = sorted(matches_df['team'].unique())
        selected_team = st.selectbox("Select Team", all_teams, index=0)
        
    with col_sel2:
        # Get matches for this team
        team_matches = matches_df[matches_df['team'] == selected_team].copy()
        team_matches = team_matches.merge(
            matches_meta[['match_id', 'match_date', 'home_team', 'away_team']], 
            on='match_id', 
            how='left'
        ).sort_values(by='match_date')
        
        match_labels = team_matches.apply(
            lambda r: f"{r['match_date']}: {r['home_team']} vs {r['away_team']}", axis=1
        ).tolist()
        
        if not match_labels:
            st.error("No matches found for this team.")
            return
            
        selected_match_label = st.selectbox("Select Match", match_labels, index=0)
        
    # Get selected match ID
    match_idx = match_labels.index(selected_match_label)
    selected_match_row = team_matches.iloc[match_idx]
    match_id = int(selected_match_row['match_id'])
    
    # Load raw events for this single match (to build timeline)
    with st.spinner("Loading match events..."):
        try:
            match_events = get_match_events(match_id)
        except Exception as e:
            st.error(f"Failed to load raw events for match: {e}")
            return
            
    # Get home and away team names
    meta_row = matches_meta[matches_meta['match_id'] == match_id].iloc[0]
    home_team = meta_row['home_team']
    away_team = meta_row['away_team']
    home_score = meta_row['home_score']
    away_score = meta_row['away_score']
    
    # Load features for this match (both home and away)
    match_feats = matches_df[matches_df['match_id'] == match_id]
    home_feats = match_feats[match_feats['team'] == home_team].iloc[0]
    away_feats = match_feats[match_feats['team'] == away_team].iloc[0]
    
    # Display match scorecard
    st.markdown(f"### 🏟️ {home_team}  {home_score} - {away_score}  {away_team}")
    st.markdown(f"<p style='color: #a0aec0;'>Date: {meta_row['match_date']}</p>", unsafe_allow_html=True)
    
    # 2. Side-by-side stats comparison
    st.markdown("#### 📊 Match Team Stats")
    
    comp_col1, comp_col2, comp_col3 = st.columns([2, 1, 2])
    
    def render_stat_row(label, home_val, away_val, format_str="{:.1f}", invert=False):
        # Determine who did better (PPDA lower is better, others higher is better)
        home_num = float(home_val)
        away_num = float(away_val)
        
        is_home_better = (home_num < away_num) if invert else (home_num > away_num)
        is_away_better = (away_num < home_num) if invert else (away_num > home_num)
        
        home_style = "color: #38a169; font-weight: bold;" if is_home_better else ""
        away_style = "color: #38a169; font-weight: bold;" if is_away_better else ""
        
        st.markdown(f"""
        <div style="display: flex; justify-content: space-between; padding: 6px 12px; border-bottom: 1px solid rgba(255,255,255,0.05);">
            <div style="width: 35%; text-align: left; {home_style}">{format_str.format(home_val)}</div>
            <div style="width: 30%; text-align: center; color: #a0aec0; font-size: 13px;">{label}</div>
            <div style="width: 35%; text-align: right; {away_style}">{format_str.format(away_val)}</div>
        </div>
        """, unsafe_allow_html=True)
        
    with comp_col1:
        st.markdown(f"<h4 style='text-align: left; color: #3182ce;'>{home_team} (Home)</h4>", unsafe_allow_html=True)
    with comp_col2:
        st.markdown("<h4 style='text-align: center; color: #fafafa;'>VS</h4>", unsafe_allow_html=True)
    with comp_col3:
        st.markdown(f"<h4 style='text-align: right; color: #e53e3e;'>{away_team} (Away)</h4>", unsafe_allow_html=True)
        
    render_stat_row("PPDA", home_feats['ppda'], away_feats['ppda'], "{:.2f}", invert=True)
    render_stat_row("Possession %", home_feats['possession_pct'], away_feats['possession_pct'], "{:.1f}%")
    render_stat_row("Pressures Applied", home_feats['pressures_count'], away_feats['pressures_count'], "{:.0f}")
    render_stat_row("Press Regain Rate", home_feats['pressing_success_rate'], away_feats['pressing_success_rate'], "{:.1f}%")
    render_stat_row("Counter-pressures", home_feats['counter_pressures_count'], away_feats['counter_pressures_count'], "{:.0f}")
    render_stat_row("Dangerous Regains", home_feats['dangerous_regains_count'], away_feats['dangerous_regains_count'], "{:.0f}")
    render_stat_row("Shots Created", home_feats['shots_created'], away_feats['shots_created'], "{:.0f}")
    render_stat_row("Expected Goals (xG)", home_feats['xg_scored'], away_feats['xg_scored'], "{:.2f}")

    st.markdown("---")
    
    # 3. Timeline & Maps
    tab_options = ["timeline", "regains"]
    tab_labels = {
        "timeline": "📈 Pressing Timeline",
        "regains": "🗺️ Regains & Shot Maps"
    }
    active_tab = render_segmented_tabs(tab_options, labels=tab_labels, query_param_name="tab")
    st.markdown("---")
    
    if active_tab == "timeline":
        st.markdown("### 📈 Pressing Intensity Over 90 Minutes")
        st.markdown("Visualizes rolling 5-minute pressure event frequencies to illustrate game momentum and tactical adaptations.")
        fig_timeline = plot_pressing_intensity_timeline(match_events, home_team, away_team)
        col_l, col_c, col_r = st.columns([1, 5, 1])
        with col_c:
            st.pyplot(fig_timeline)
        
    elif active_tab == "regains":
        st.markdown("### 🗺️ Regains & Resulting Shots Map")
        st.markdown("Select a team to plot where they won the ball back and which regains led to subsequent shots (within 15 seconds).")
        
        map_team = st.radio("Select Team to Map", [home_team, away_team])
        
        # Filter pressures for this match and team
        match_pressures = events_df[
            (events_df['match_id'] == match_id) & 
            (events_df['team'] == map_team)
        ]
        
        if len(match_pressures) > 0:
            fig_map = plot_regains_and_shots(match_pressures, map_team)
            col_l, col_c, col_r = st.columns([1, 5, 1])
            with col_c:
                st.pyplot(fig_map)
        else:
            st.info(f"No pressure events recorded for {map_team} in this match.")

if __name__ == "__main__":
    main()
