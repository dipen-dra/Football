import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
import joblib
import plotly.express as px
import plotly.graph_objects as go
import warnings

# Add workspace to path
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from src.app_helper import init_app, render_dark_table, render_segmented_tabs
from src.viz import DARK_BG, TEXT_COLOR, CARD_BG, ACCENT_BLUE, ACCENT_GOLD, ACCENT_RED

warnings.filterwarnings("ignore")

def main():
    success = init_app("Match Outcome Model", "A machine learning model predicting match outcomes (Win / Draw / Loss) based on team pressing profiles, possession%, and xG metrics.")
    if not success:
        return
        
    comp_id = st.session_state.get('comp_id')
    season_id = st.session_state.get('season_id')
    model_path = os.path.join("models", f"outcome_model_{comp_id}_{season_id}.joblib")
    
    if not os.path.exists(model_path) or st.session_state.get('outcome_model_data') is None:
        st.warning("⚠️ Model file not found. Please click 'Download & Build Pipeline' in the sidebar to train the models.")
        return
        
    model_data = st.session_state['outcome_model_data']
    
    rf_metrics = model_data['rf_metrics']
    lr_metrics = model_data['lr_metrics']
    feature_names = model_data['feature_names']
    rf_model = model_data['rf_model']
    importances = model_data['feature_importances']
    
    # 1. Performance Overview
    st.markdown('<div class="section-header">📊 Model Performance Comparison</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Random Forest Classifier (Best Model)")
        sub_col1, sub_col2 = st.columns(2)
        with sub_col1:
            st.metric("Test Accuracy", f"{rf_metrics['accuracy']*100:.1f}%")
        with sub_col2:
            st.metric("Test Log-Loss", f"{rf_metrics['log_loss']:.3f}")
            
        # Display confusion matrix as dark HTML table
        cm = rf_metrics['conf_matrix']
        render_dark_table(
            pd.DataFrame(cm, 
                index=['Actual Loss', 'Actual Draw', 'Actual Win'], 
                columns=['Predicted Loss', 'Predicted Draw', 'Predicted Win']
            ),
            caption="Confusion Matrix (Random Forest)"
        )
        
    with col2:
        st.markdown("#### Logistic Regression (Baseline)")
        sub_col1, sub_col2 = st.columns(2)
        with sub_col1:
            st.metric("Test Accuracy", f"{lr_metrics['accuracy']*100:.1f}%")
        with sub_col2:
            st.metric("Test Log-Loss", f"{lr_metrics['log_loss']:.3f}")
            
        # Display confusion matrix as dark HTML table
        cm_lr = lr_metrics['conf_matrix']
        render_dark_table(
            pd.DataFrame(cm_lr, 
                index=['Actual Loss', 'Actual Draw', 'Actual Win'], 
                columns=['Predicted Loss', 'Predicted Draw', 'Predicted Win']
            ),
            caption="Confusion Matrix (Logistic Regression)"
        )
        
    st.markdown("---")
    
    # Segmented custom tabs with routes
    tab_options = ["explain", "sim"]
    tab_labels = {
        "explain": "💡 Feature Importance",
        "sim": "🎮 Interactive Match Simulator"
    }
    active_tab = render_segmented_tabs(tab_options, labels=tab_labels, query_param_name="tab")
    st.markdown("---")
    
    if active_tab == "explain":
        st.markdown('<div class="section-header">💡 What Drives Match Outcomes?</div>', unsafe_allow_html=True)
        st.markdown("Feature importance scores from the Random Forest model show which inputs have the greatest predictive power.")
        
        # Plot feature importances
        df_imp = pd.DataFrame({
            'Feature': [f.replace('_', ' ').title() for f in feature_names],
            'Importance': importances
        }).sort_values(by='Importance', ascending=True)
        
        fig_imp = px.bar(
            df_imp,
            y='Feature',
            x='Importance',
            orientation='h',
            title='Random Forest Feature Importance',
            color='Importance',
            color_continuous_scale='Viridis'
        )
        
        fig_imp.update_layout(
            paper_bgcolor=CARD_BG,
            plot_bgcolor=CARD_BG,
            font=dict(color=TEXT_COLOR),
            xaxis=dict(gridcolor='#1E293B', showgrid=True),
            yaxis=dict(gridcolor='#1E293B', showgrid=False),
            margin=dict(l=40, r=40, t=50, b=40)
        )
        
        st.plotly_chart(fig_imp, use_container_width=True)
        
        st.markdown("""
        > [!IMPORTANT]
        > **Thesis Analytical Caveat (Correlation vs Causal Confounders):**
        > - Expected Goals (**xG Scored / Conceded**) and **Possession%** are highly predictive, acting as strong control variables.
        > - Pressing metrics like **PPDA** and **Pressing Success Rate** add incremental predictive accuracy, representing *tactical efficiency*.
        > - Remember: These correlations do not imply causality. Score effects (e.g. teams drop their press when winning to protect a lead) act as significant confounders.
        """)
        
    elif active_tab == "sim":
        st.markdown('<div class="section-header">🎮 Team Profile Simulator</div>', unsafe_allow_html=True)
        st.markdown("Adjust the sliders below to simulate a team's statistical match profile and view the model's predicted outcome probabilities.")
        
        sim_col1, sim_col2 = st.columns([1, 1])
        
        with sim_col1:
            st.markdown("#### ⚙️ Team Stats Simulator")
            
            sim_ppda = st.slider("PPDA (Passes Allowed per Def. Action)", min_value=4.0, max_value=25.0, value=11.5, step=0.1)
            sim_pressures = st.slider("Total Pressures Applied", min_value=30, max_value=300, value=140, step=5)
            
            col_z1, col_z2 = st.columns(2)
            with col_z1:
                sim_att = st.slider("Pressures in Attacking Third", min_value=5, max_value=120, value=40, step=1)
            with col_z2:
                sim_mid = st.slider("Pressures in Middle Third", min_value=10, max_value=180, value=70, step=1)
                
            sim_cp = st.slider("Counter-pressures Count", min_value=0, max_value=80, value=25, step=1)
            sim_succ_rate = st.slider("Pressing Success Rate (%)", min_value=10.0, max_value=50.0, value=28.0, step=0.5)
            
            sim_poss = st.slider("Possession Percentage (%)", min_value=20.0, max_value=80.0, value=50.0, step=0.5)
            
            col_xg1, col_xg2 = st.columns(2)
            with col_xg1:
                sim_xg_scored = st.slider("Expected Goals Created (xG)", min_value=0.0, max_value=4.0, value=1.4, step=0.1)
            with col_xg2:
                sim_xg_conceded = st.slider("Expected Goals Conceded (xG)", min_value=0.0, max_value=4.0, value=1.0, step=0.1)
                
        with sim_col2:
            st.markdown("#### 🔮 Predicted Outcome Probabilities")
            
            # Feature ordering:
            # ['ppda', 'pressures_count', 'pressures_att_third', 'pressures_mid_third',
            #  'counter_pressures_count', 'pressing_success_rate', 'dangerous_regains_count',
            #  'possession_pct', 'xg_scored', 'xg_conceded']
            
            # Map inputs. Let's proxy dangerous regains as pressures * success_rate * 0.1 (a sensible heuristic for what-if)
            sim_dang = int(sim_pressures * (sim_succ_rate / 100.0) * 0.15)
            
            input_dict = {
                'ppda': sim_ppda,
                'pressures_count': sim_pressures,
                'pressures_att_third': sim_att,
                'pressures_mid_third': sim_mid,
                'counter_pressures_count': sim_cp,
                'pressing_success_rate': sim_succ_rate,
                'dangerous_regains_count': sim_dang,
                'possession_pct': sim_poss,
                'xg_scored': sim_xg_scored,
                'xg_conceded': sim_xg_conceded
            }
            
            input_df = pd.DataFrame([input_dict])
            
            # Model output classes: 0 (Loss), 1 (Draw), 2 (Win)
            probs = rf_model.predict_proba(input_df)[0]
            
            # Map predictions dynamically to available classes
            prob_dict = {0: 0.0, 1: 0.0, 2: 0.0}
            for cls_idx, class_label in enumerate(rf_model.classes_):
                prob_dict[class_label] = probs[cls_idx]
                
            loss_prob = prob_dict[0] * 100
            draw_prob = prob_dict[1] * 100
            win_prob = prob_dict[2] * 100
            
            # Predict outcome class
            pred_class = rf_model.predict(input_df)[0]
            class_labels = {0: 'LOSS', 1: 'DRAW', 2: 'WIN'}
            class_colors = {0: ACCENT_RED, 1: ACCENT_GOLD, 2: ACCENT_BLUE}
            
            st.markdown(f"""
            <div class="metric-card" style="text-align: center; border-left: 5px solid {class_colors[pred_class]};">
                <div class="metric-title" style="font-size: 16px;">Predicted Outcome</div>
                <div class="metric-value" style="color: {class_colors[pred_class]}; font-size: 32px; margin: 10px 0;">{class_labels[pred_class]} PREDICTED</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Draw a Plotly pie/donut chart
            fig_pie = go.Figure(data=[go.Pie(
                labels=['Loss', 'Draw', 'Win'],
                values=[loss_prob, draw_prob, win_prob],
                hole=.4,
                marker=dict(colors=[ACCENT_RED, ACCENT_GOLD, ACCENT_BLUE]),
                textinfo='label+percent',
                hoverinfo='percent'
            )])
            
            fig_pie.update_layout(
                paper_bgcolor=DARK_BG,
                plot_bgcolor=DARK_BG,
                font=dict(color=TEXT_COLOR),
                margin=dict(l=20, r=20, t=20, b=20),
                height=250,
                showlegend=False
            )
            
            st.plotly_chart(fig_pie, use_container_width=True)
            
            st.caption("Distribution of predicted outcomes based on simulated match statistics.")

if __name__ == "__main__":
    main()
