import streamlit as st
import pandas as pd
import sys
import os

# Add workspace to path
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from src.app_helper import init_app, render_dark_table, render_segmented_tabs
from src.viz import plot_correlation_scatter

def main():
    success = init_app("League Explorer", "Analyze team-level pressing profiles and correlations across the entire league.")
    if not success or 'matches_features' not in st.session_state:
        st.warning("Please load dataset in sidebar first.")
        return
        
    matches_df = st.session_state['matches_features']
    
    # Aggregate data by team (calculate averages per match)
    # We want averages for most variables except points/wins which we can sum or average
    team_metrics = matches_df.groupby('team').agg(
        matches_played=('match_id', 'count'),
        avg_ppda=('ppda', 'mean'),
        avg_pressures=('pressures_count', 'mean'),
        avg_pressures_def=('pressures_def_third', 'mean'),
        avg_pressures_mid=('pressures_mid_third', 'mean'),
        avg_pressures_att=('pressures_att_third', 'mean'),
        avg_counter_pressures=('counter_pressures_count', 'mean'),
        pressing_success_rate=('pressing_success_rate', 'mean'),
        avg_dangerous_regains=('dangerous_regains_count', 'mean'),
        avg_possession=('possession_pct', 'mean'),
        avg_xg_scored=('xg_scored', 'mean'),
        avg_xg_conceded=('xg_conceded', 'mean'),
        avg_goals_scored=('goals_scored', 'mean'),
        avg_goals_conceded=('goals_conceded', 'mean'),
        points_per_match=('points', 'mean'),
        total_points=('points', 'sum')
    ).reset_index()
    
    # Calculate Attacking Third Pressing Share
    team_metrics['att_third_press_share'] = (
        team_metrics['avg_pressures_att'] / team_metrics['avg_pressures'] * 100.0
    ).fillna(0.0)
    
    # Segmented custom tabs with routes
    tab_options = ["profiles", "correlations"]
    tab_labels = {
        "profiles": "📋 Team Pressing Profiles",
        "correlations": "📈 Pressing Correlations"
    }
    active_tab = render_segmented_tabs(tab_options, labels=tab_labels, query_param_name="tab")
    st.markdown("---")
    
    if active_tab == "profiles":
        st.markdown('<div class="section-header">📊 Team Pressing Statistics Table</div>', unsafe_allow_html=True)
        st.markdown("Overview of seasonal pressing metrics averaged per match.")
        
        # Display table
        display_cols = {
            'team': 'Team',
            'matches_played': 'MP',
            'avg_ppda': 'PPDA',
            'avg_pressures': 'Pressures/90',
            'att_third_press_share': 'Att Third Press %',
            'avg_counter_pressures': 'Counter-press/90',
            'pressing_success_rate': 'Press Success %',
            'avg_possession': 'Possession %',
            'points_per_match': 'Points/Match'
        }
        
        table_df = team_metrics[list(display_cols.keys())].rename(columns=display_cols)
        table_df = table_df.sort_values(by='PPDA') # Sort by most intense pressing teams by default
        
        # Format values for display
        for col, fmt in [('PPDA', '{:.2f}'), ('Pressures/90', '{:.1f}'),
                         ('Att Third Press %', '{:.1f}%'), ('Counter-press/90', '{:.1f}'),
                         ('Press Success %', '{:.1f}%'), ('Possession %', '{:.1f}%'),
                         ('Points/Match', '{:.2f}')]:
            if col in table_df.columns:
                table_df[col] = table_df[col].apply(lambda v: fmt.format(v))
        
        render_dark_table(table_df.reset_index(drop=True), caption="All Teams — Sorted by Pressing Intensity (PPDA)")
        
    elif active_tab == "correlations":
        st.markdown('<div class="section-header">🔗 Pressing &amp; Match Outcomes Relationship</div>', unsafe_allow_html=True)
        st.markdown("Explore how pressing variables correlate with attacking, defensive, and match success outcomes.")
        
        col1, col2 = st.columns(2)
        
        # Variable options mapping
        var_opts = {
            'avg_ppda': 'Passes Allowed per Defensive Action (PPDA)',
            'avg_pressures': 'Total Pressures per Match',
            'att_third_press_share': 'Attacking Third Pressing Share (%)',
            'avg_counter_pressures': 'Counter-Pressures per Match',
            'pressing_success_rate': 'Pressing Success Rate (%)',
            'avg_dangerous_regains': 'Dangerous Regains per Match'
        }
        
        outcome_opts = {
            'points_per_match': 'Points per Match',
            'avg_goals_conceded': 'Goals Conceded per Match',
            'avg_xg_conceded': 'xG Conceded per Match',
            'avg_goals_scored': 'Goals Scored per Match',
            'avg_xg_scored': 'xG Scored per Match',
            'avg_possession': 'Average Possession %'
        }
        
        with col1:
            x_select = st.selectbox(
                "Select Pressing Feature (X-Axis)",
                options=list(var_opts.keys()),
                format_func=lambda x: var_opts[x],
                index=0
            )
            
        with col2:
            y_select = st.selectbox(
                "Select Match Outcome Feature (Y-Axis)",
                options=list(outcome_opts.keys()),
                format_func=lambda x: outcome_opts[x],
                index=0
            )
            
        # Draw plot
        fig = plot_correlation_scatter(
            team_metrics,
            x_col=x_select,
            y_col=y_select,
            x_label=var_opts[x_select],
            y_label=outcome_opts[y_select],
            color_col='avg_possession'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Explain the correlation
        st.markdown("""
        > [!NOTE]
        > **Analytical Interpretation:**
        > - A negative correlation with **PPDA** means that *lower* PPDA values (higher pressing intensity) are associated with higher values of the Y-axis variable (e.g. higher Points/Match or higher goals conceded if team quality varies).
        > - Pay attention to **p-value**: A p-value < 0.05 indicates statistical significance at the 95% confidence level.
        """)

if __name__ == "__main__":
    main()
