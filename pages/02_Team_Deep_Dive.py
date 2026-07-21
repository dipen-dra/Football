import streamlit as st
import pandas as pd
import sys
import os
import plotly.express as px

# Add workspace to path
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from src.app_helper import init_app, render_segmented_tabs
from src.viz import plot_pressure_heatmap, ACCENT_BLUE, ACCENT_GOLD, CARD_BG, TEXT_COLOR

def main():
    success = init_app("Team Deep Dive", "Detailed investigation of a single team's pressing intensity, spatial tactics, and triggering patterns.")
    if not success or 'matches_features' not in st.session_state:
        st.warning("Please load dataset in sidebar first.")
        return
        
    events_df = st.session_state['events_features']
    matches_df = st.session_state['matches_features']
    matches_meta = st.session_state['matches_df']
    
    # Team Selector
    all_teams = sorted(matches_df['team'].unique())
    selected_team = st.selectbox("Select Team", all_teams, index=0)
    
    # 1. Load team data
    team_matches = matches_df[matches_df['team'] == selected_team].sort_values(by='match_id')
    team_pressures = events_df[events_df['team'] == selected_team]
    
    # Get match metadata to sort matches chronologically
    team_matches = team_matches.merge(
        matches_meta[['match_id', 'match_date', 'home_team', 'away_team']], 
        on='match_id', 
        how='left'
    ).sort_values(by='match_date')
    
    # 2. Benchmarks
    league_ppda = matches_df['ppda'].mean()
    league_pressures = matches_df['pressures_count'].mean()
    league_success = matches_df['pressing_success_rate'].mean()
    league_cp = matches_df['counter_pressures_count'].mean()
    
    team_avg_ppda = team_matches['ppda'].mean()
    team_avg_pressures = team_matches['pressures_count'].mean()
    team_avg_success = team_matches['pressing_success_rate'].mean()
    team_avg_cp = team_matches['counter_pressures_count'].mean()
    
    # Display comparison cards
    col1, col2, col3, col4 = st.columns(4)
    
    def display_benchmark_card(title, value, league_val, higher_is_better=True):
        diff = value - league_val
        is_better = (diff >= 0) if higher_is_better else (diff <= 0)
        
        # Color coding delta
        delta_class = "delta-positive" if is_better else "delta-negative"
        sign = "+" if diff > 0 else ""
        
        # PPDA lower is better, so flip delta logic for display
        if not higher_is_better:
            delta_text = f"{sign}{diff:.2f} vs League ({league_val:.2f})"
            val_str = f"{value:.2f}"
        else:
            delta_text = f"{sign}{diff:.1f}% vs League ({league_val:.1f}%)" if "%" in title or "Rate" in title else f"{sign}{diff:.1f} vs League ({league_val:.1f})"
            val_str = f"{value:.1f}"
            
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">{title}</div>
            <div class="metric-value">{val_str}</div>
            <div class="metric-delta {delta_class}">{delta_text}</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col1:
        display_benchmark_card("Average PPDA", team_avg_ppda, league_ppda, higher_is_better=False)
    with col2:
        display_benchmark_card("Avg Pressures/Match", team_avg_pressures, league_pressures)
    with col3:
        display_benchmark_card("Press Success Rate", team_avg_success, league_success)
    with col4:
        display_benchmark_card("Avg Counter-pressures", team_avg_cp, league_cp)
        
    st.markdown("---")
    
    # Segmented custom tabs with routes
    tab_options = ["spatial", "trend", "triggers"]
    tab_labels = {
        "spatial": "🗺️ Spatial Heatmap",
        "trend": "📈 Seasonal PPDA Trend",
        "triggers": "⚡ Pressing Triggers"
    }
    active_tab = render_segmented_tabs(tab_options, labels=tab_labels, query_param_name="tab")
    st.markdown("---")
    
    if active_tab == "spatial":
        st.markdown("### 🗺️ Pressure Event Heatmap")
        st.markdown("This map shows where the team applies pressure. Red/hot zones represent higher pressure concentration.")
        
        col_heat, col_desc = st.columns([1.3, 1.0])
        
        with col_heat:
            fig_heatmap = plot_pressure_heatmap(events_df, selected_team)
            st.pyplot(fig_heatmap)
            
        with col_desc:
            st.markdown("#### 🔍 Tactical Insights")
            
            # Attattacking / defensive breakdown
            p_def = len(team_pressures[team_pressures['zone'] == 'Defensive Third'])
            p_mid = len(team_pressures[team_pressures['zone'] == 'Middle Third'])
            p_att = len(team_pressures[team_pressures['zone'] == 'Attacking Third'])
            total_p = len(team_pressures)
            
            if total_p > 0:
                p_def_pct = p_def / total_p * 100
                p_mid_pct = p_mid / total_p * 100
                p_att_pct = p_att / total_p * 100
            else:
                p_def_pct, p_mid_pct, p_att_pct = 0, 0, 0
                
            st.write(f"**Zone Breakdown:**")
            st.write(f"- 🔴 **Attacking Third:** {p_att_pct:.1f}%")
            st.write(f"- 🟡 **Middle Third:** {p_mid_pct:.1f}%")
            st.write(f"- 🟢 **Defensive Third:** {p_def_pct:.1f}%")
            
            st.markdown("---")
            if p_att_pct > 30:
                st.info(f"💡 **High Pressing Signature:** {selected_team} exhibits an intense high pressing system, applying over 30% of their pressures in the attacking third. This indicates a tactic focused on winning the ball close to the opponent's goal.")
            elif p_mid_pct > 50:
                st.info(f"💡 **Mid-Block Signature:** {selected_team} focuses pressing actions heavily in the middle third, indicating a structured mid-block designed to choke opponent progression passing lanes.")
            else:
                st.info(f"💡 **Low-Block / Restrictive Signature:** {selected_team} records a high proportion of pressures in their own defensive third, indicating a deep defensive structure that prioritizes penalty box protection over high-up regains.")
                
    elif active_tab == "trend":
        st.markdown("### 📈 Seasonal PPDA Trend")
        st.markdown("Track the match-by-match variation of PPDA over the course of the season. Lower values represent higher pressing intensity.")
        
        if len(team_matches) > 1:
            team_matches['match_num'] = range(1, len(team_matches) + 1)
            # Add opponent names to label
            team_matches['Opponent_Label'] = team_matches.apply(
                lambda r: f"Vs {r['opponent']} ({'H' if r['is_home'] else 'A'})", axis=1
            )
            
            # Simple line chart using plotly
            fig_trend = px.line(
                team_matches,
                x='match_num',
                y='ppda',
                hover_data=['Opponent_Label', 'match_result'],
                title=f"PPDA Match-by-Match (Season Average: {team_avg_ppda:.2f})",
                labels={'match_num': 'Match Number', 'ppda': 'PPDA'}
            )
            
            # Add a rolling average line
            team_matches['ppda_rolling'] = team_matches['ppda'].rolling(window=3, min_periods=1).mean()
            fig_trend.add_scatter(
                x=team_matches['match_num'],
                y=team_matches['ppda_rolling'],
                mode='lines',
                line=dict(color=ACCENT_GOLD, width=2.5),
                name='3-Match Rolling Avg'
            )
            
            # Style plot
            fig_trend.update_layout(
                paper_bgcolor=CARD_BG,
                plot_bgcolor=CARD_BG,
                font=dict(color=TEXT_COLOR),
                xaxis=dict(gridcolor='#2d3748', showgrid=True),
                yaxis=dict(gridcolor='#2d3748', showgrid=True),
                margin=dict(l=40, r=40, t=50, b=40)
            )
            
            st.plotly_chart(fig_trend, use_container_width=True)
        else:
            st.info("Insufficient matches to plot a trend line.")
            
    elif active_tab == "triggers":
        st.markdown("### ⚡ Pressing Triggers Analysis")
        st.markdown("Shows what events immediately preceded the team's pressures. Understanding triggers is crucial for tactical analysis.")
        
        trigger_counts = team_pressures['trigger'].value_counts().reset_index()
        trigger_counts.columns = ['Trigger', 'Count']
        trigger_counts['Share (%)'] = trigger_counts['Count'] / trigger_counts['Count'].sum() * 100.0
        
        fig_trigger = px.bar(
            trigger_counts,
            y='Trigger',
            x='Share (%)',
            orientation='h',
            title=f"Pressing Triggers Share: {selected_team}",
            color='Share (%)',
            color_continuous_scale='Plasma',
            text=trigger_counts['Share (%)'].apply(lambda s: f"{s:.1f}%")
        )
        
        fig_trigger.update_layout(
            paper_bgcolor=CARD_BG,
            plot_bgcolor=CARD_BG,
            font=dict(color=TEXT_COLOR),
            xaxis=dict(gridcolor='#2d3748', showgrid=True),
            yaxis=dict(gridcolor='#2d3748', showgrid=False),
            margin=dict(l=40, r=40, t=50, b=40)
        )
        
        st.plotly_chart(fig_trigger, use_container_width=True)
        
        st.markdown("""
        > [!TIP]
        > **Thesis Explanation of Triggers:**
        > - **Poor Touch / Miscontrol:** Opponent player makes a bad touch, triggering defenders to rush in and contest.
        > - **Backward Pass:** Opponent passes backward, which acts as a trigger to push the defensive block higher.
        > - **Pass Reception / Build-up:** Defenders press as the ball travels to a recipient to disrupt control.
        > - **Carrying / Dribbling:** Defender presses a ball carrier to stop progression.
        """)

if __name__ == "__main__":
    main()
