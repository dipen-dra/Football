import streamlit as st
import pandas as pd
import numpy as np
import sys
import os

# Add workspace to path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.app_helper import init_app, render_dark_table

def main():
    # Initialize page
    success = init_app("Overview", "Thesis Research Dashboard on Pressing Intensity & Its Impact on Defensive Structure and Match Outcomes.")
    if not success:
        st.warning("Please configure the dataset in the sidebar to get started.")
        st.info("💡 Select a competition (e.g. Women's World Cup 2019) and click 'Download & Build Pipeline' to start.")
        return

    # Check if data is loaded
    if 'events_features' not in st.session_state or 'matches_features' not in st.session_state:
        st.info("👋 Welcome! Use the sidebar on the left to select a competition and click **Download & Build Pipeline** to load the dataset and train models.")
        
        # Display project brief in card format
        st.markdown("""
        <div class="metric-card">
            <h3>🔬 Research Project: Pressing Intensity & Match Outcomes</h3>
            <p>This interactive analytics dashboard is a final-year bachelor's thesis deliverable exploring the relationships between football pressing intensity, defensive structures, and match outcomes.</p>
            <h4>Key Objectives:</h4>
            <ul>
                <li><b>Metric Computation:</b> Calculate Passes Allowed per Defensive Action (PPDA) and analyze spatial pressure event distributions.</li>
                <li><b>Press Success Classifier:</b> A machine learning model (Logistic Regression & XGBoost) predicting whether a press results in a possession regain within 5 seconds.</li>
                <li><b>Outcome Predictor:</b> A team-level model predicting match outcomes (Win/Draw/Loss) using aggregate pressing and possession metrics.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        return

    # Data is loaded, render the dashboard
    comp_name = st.session_state['comp_name']
    season_name = st.session_state['season_name']
    events_df = st.session_state['events_features']
    matches_df = st.session_state['matches_features']

    # 1. League-wide KPIs
    st.markdown("### 🏆 League-wide Pressing KPIs")
    
    # Calculations
    avg_ppda = matches_df['ppda'].mean()
    avg_pressures = matches_df['pressures_count'].mean()
    avg_success_rate = matches_df['pressing_success_rate'].mean()
    avg_counter_press = matches_df['counter_pressures_count'].mean()
    
    # Benchmark calculations (e.g. standard PPDA in top leagues is around 10-12, lower is more intense)
    kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
    
    with kpi_col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Avg PPDA</div>
            <div class="metric-value">{avg_ppda:.2f}</div>
            <div class="metric-delta">Lower indicates higher pressing intensity</div>
        </div>
        """, unsafe_allow_html=True)
        
    with kpi_col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Avg Pressures / Team-Match</div>
            <div class="metric-value">{avg_pressures:.1f}</div>
            <div class="metric-delta">Total pressures applied per 90 mins</div>
        </div>
        """, unsafe_allow_html=True)
        
    with kpi_col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Press Regain Success Rate</div>
            <div class="metric-value">{avg_success_rate:.1f}%</div>
            <div class="metric-delta">Possession regained within 5 seconds</div>
        </div>
        """, unsafe_allow_html=True)
        
    with kpi_col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Avg Counter-pressures / Match</div>
            <div class="metric-value">{avg_counter_press:.1f}</div>
            <div class="metric-delta">Pressures within 5s of losing ball</div>
        </div>
        """, unsafe_allow_html=True)

    # 2. Main Content Layout
    col_left, col_right = st.columns([3, 2])
    
    with col_left:
        st.markdown("#### 📖 Project Background")
        st.markdown("""
        In modern football, pressing is not merely a defensive duty but a primary playmaker tool. Teams use organized pressing structures to disrupt opponent build-ups, force turnovers near the opponent's goal, and prevent counterattacks.
        
        This thesis project explores:
        1. **PPDA (Passes Allowed per Defensive Action)**: The standard quantitative metric measuring how many passes a team allows their opponent in the opponent's defensive 60% of the pitch before making a defensive action. A lower PPDA indicates a more intense high press.
        2. **Pressure Events & Triggers**: Analyzing the contextual cues (like backward passes, poor ball control) that cause defensive players to initiate a pressure event.
        3. **Predictive Analytics**: Using Machine Learning to classify pressure success and predict match outcomes from defensive intensity signatures.
        """)
        
        # Team PPDA Ranking
        st.markdown("#### ⚡ Top 5 Pressing Teams (Lowest Season-Avg PPDA)")
        team_ppda = matches_df.groupby('team')['ppda'].mean().reset_index()
        top_pressing = team_ppda.sort_values(by='ppda').head(5)
        top_pressing.columns = ['Team', 'Average PPDA']
        
        # Display as a styled dark HTML table
        render_dark_table(top_pressing.reset_index(drop=True), caption="Top 5 Pressing Teams — Lowest Season-Avg PPDA")

    with col_right:
        st.markdown("#### 📂 Dataset Summary")
        st.markdown(f"""
        - **Source:** StatsBomb Open Data
        - **Competition:** {comp_name}
        - **Season:** {season_name}
        - **Number of Matches:** {len(matches_df) // 2}
        - **Number of Pressure Events:** {len(events_df):,}
        """)
        
        st.info("💡 **Navigation Guide:** Use the sidebar or pages menu to navigate through sections:\n"
                "- **League Explorer:** View team comparisons and correlations.\n"
                "- **Team Deep Dive:** Spatial pressure maps and seasonal trends.\n"
                "- **Match Detail:** Press timeline and regains map for a single match.\n"
                "- **Pressing Success Model:** Interactive classifier & feature explainability.\n"
                "- **Match Outcome Model:** Match predictor using pressing metrics.\n"
                "- **Methodology:** Formulas and assumptions details.")

if __name__ == "__main__":
    main()
