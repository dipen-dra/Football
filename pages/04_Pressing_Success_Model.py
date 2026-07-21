import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
import joblib
import matplotlib.pyplot as plt
import shap
import warnings

# Add workspace to path
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from src.app_helper import init_app, render_dark_table, render_segmented_tabs
from src.modeling import TRIGGER_MAPPING
from src.viz import DARK_BG, TEXT_COLOR, CARD_BG, ACCENT_EMERALD, ACCENT_RED

warnings.filterwarnings("ignore")

def main():
    success = init_app("Pressing Success Model", "A machine learning model predicting whether an individual pressing action results in a ball regain within 5 seconds.")
    if not success:
        return
        
    comp_id = st.session_state.get('comp_id')
    season_id = st.session_state.get('season_id')
    model_path = os.path.join("models", f"success_model_{comp_id}_{season_id}.joblib")
    
    if not os.path.exists(model_path) or st.session_state.get('success_model_data') is None:
        st.warning("⚠️ Model file not found. Please click 'Download & Build Pipeline' in the sidebar to train the models.")
        return
        
    model_data = st.session_state['success_model_data']
    
    xgb_metrics = model_data['xgb_metrics']
    lr_metrics = model_data['lr_metrics']
    feature_names = model_data['feature_names']
    xgb_model = model_data['xgb_model']
    X_test = model_data['X_test']
    shap_values = model_data['shap_values']
    
    # 1. Performance Overview
    st.markdown('<div class="section-header">📊 Model Performance Comparison</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### XGBoost Classifier (Best Model)")
        sub_col1, sub_col2, sub_col3 = st.columns(3)
        with sub_col1:
            st.metric("Test Accuracy", f"{xgb_metrics['accuracy']*100:.1f}%")
        with sub_col2:
            st.metric("Test ROC-AUC", f"{xgb_metrics['auc']:.3f}")
        with sub_col3:
            st.metric("Test Recall", f"{xgb_metrics['recall']*100:.1f}%")
            
        # Display confusion matrix as dark HTML table
        cm = xgb_metrics['conf_matrix']
        render_dark_table(
            pd.DataFrame(cm, 
                index=['Actual Fail', 'Actual Success'], 
                columns=['Predicted Fail', 'Predicted Success']
            ),
            caption="Confusion Matrix (XGBoost)"
        )
        
    with col2:
        st.markdown("#### Logistic Regression (Baseline)")
        sub_col1, sub_col2, sub_col3 = st.columns(3)
        with sub_col1:
            st.metric("Test Accuracy", f"{lr_metrics['accuracy']*100:.1f}%")
        with sub_col2:
            st.metric("Test ROC-AUC", f"{lr_metrics['auc']:.3f}")
        with sub_col3:
            st.metric("Test Recall", f"{lr_metrics['recall']*100:.1f}%")
            
        # Display confusion matrix as dark HTML table
        cm_lr = lr_metrics['conf_matrix']
        render_dark_table(
            pd.DataFrame(cm_lr, 
                index=['Actual Fail', 'Actual Success'], 
                columns=['Predicted Fail', 'Predicted Success']
            ),
            caption="Confusion Matrix (Logistic Regression)"
        )
        
    st.markdown("---")
    
    # Segmented custom tabs with routes
    tab_options = ["explain", "sim"]
    tab_labels = {
        "explain": "💡 Feature Explainability (SHAP)",
        "sim": "🎮 Interactive What-If Simulation"
    }
    active_tab = render_segmented_tabs(tab_options, labels=tab_labels, query_param_name="tab")
    st.markdown("---")
    
    if active_tab == "explain":
        st.markdown('<div class="section-header">💡 Explaining XGBoost Decisions with SHAP Values</div>', unsafe_allow_html=True)
        st.markdown("SHAP (SHapley Additive exPlanations) values decompose the model's predictions to show the contribution of each feature to the final success probability.")
        
        if shap_values is not None:
            # Generate SHAP summary plot
            fig, ax = plt.subplots(figsize=(10, 5))
            fig.patch.set_facecolor(DARK_BG)
            ax.set_facecolor(DARK_BG)
            
            # Map encoded features to readable names for plotting if desired, but shap summary plot is fine
            # We must set text color for SHAP labels
            plt.rcParams['text.color'] = TEXT_COLOR
            plt.rcParams['axes.labelcolor'] = TEXT_COLOR
            
            # Plot
            shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False)
            
            # Adjust plot aesthetics
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
            
            st.markdown("""
            > [!NOTE]
            > **How to interpret the SHAP plot:**
            > - **Feature Value:** Red represents high values of a feature, blue represents low values.
            > - **SHAP Value (X-Axis):** Points to the right of 0 represent features that increase the probability of a successful press. Points to the left decrease it.
            > - **Key Findings:** Typically, pressing closer to the opponent's goal (**higher X / lower distance_to_goal**) and **counter-pressing** (is_counter_press = 1) have strong positive impacts on regain success.
            """)
        else:
            st.info("SHAP values could not be computed (requires more events to calculate TreeExplainer).")
            
    elif active_tab == "sim":
        st.markdown('<div class="section-header">🎮 Pressing Success What-If Simulation</div>', unsafe_allow_html=True)
        st.markdown("Tweak tactical parameters to predict the probability of a successful press in real-time.")
        
        sim_col1, sim_col2 = st.columns([1, 1])
        
        with sim_col1:
            st.markdown("#### 🛠️ Tactical Scenario Editor")
            
            # Sliders
            x_coord = st.slider("Press Start Coordinate X (Goal to Goal)", min_value=0.0, max_value=120.0, value=80.0, step=1.0, help="0 is defending goal, 120 is attacking goal")
            y_coord = st.slider("Press Start Coordinate Y (Width)", min_value=0.0, max_value=80.0, value=40.0, step=1.0, help="0 is left touchline, 80 is right touchline, 40 is center")
            
            # Recompute distance to goal
            dist = np.sqrt((120.0 - x_coord)**2 + (40.0 - y_coord)**2)
            st.caption(f"Calculated distance to opponent goal: **{dist:.1f} yards**")
            
            trigger_sel = st.selectbox(
                "Trigger Event (What preceded pressure?)",
                options=list(TRIGGER_MAPPING.keys()),
                index=1 # Poor Touch
            )
            
            is_cp = st.checkbox("Is Counter-Press? (Within 5s of losing ball)", value=True)
            under_pres = st.checkbox("Pressed player already under pressure?", value=False)
            
            match_min = st.slider("Match Minute", min_value=0, max_value=95, value=45, step=1)
            score_diff = st.slider("Score Differential (Pressing Team - Pressed Team)", min_value=-3, max_value=3, value=0, step=1)
            
        with sim_col2:
            st.markdown("#### 🔮 Model Prediction Outcome")
            
            # Construct input vector matching feature_names:
            # ['x', 'y', 'distance_to_goal', 'is_counter_press', 'score_diff', 'minute', 'trigger_encoded', 'under_pressure']
            input_dict = {
                'x': x_coord,
                'y': y_coord,
                'distance_to_goal': dist,
                'is_counter_press': int(is_cp),
                'score_diff': score_diff,
                'minute': match_min,
                'trigger_encoded': TRIGGER_MAPPING[trigger_sel],
                'under_pressure': int(under_pres)
            }
            
            input_df = pd.DataFrame([input_dict])
            
            # Predict
            prob_success = xgb_model.predict_proba(input_df)[0, 1]
            pred_class = xgb_model.predict(input_df)[0]
            
            # Display results with premium gauge/metrics
            prob_pct = prob_success * 100
            
            if pred_class == 1:
                result_text = "SUCCESSFUL REGGAIN PREDICTED"
                result_color = ACCENT_EMERALD
            else:
                result_text = "FAILED REGGAIN PREDICTED"
                result_color = ACCENT_RED
                
            st.markdown(f"""
            <div class="metric-card" style="text-align: center; border-left: 5px solid {result_color};">
                <div class="metric-title" style="font-size: 16px;">Predicted Class</div>
                <div class="metric-value" style="color: {result_color}; font-size: 32px; margin: 10px 0;">{result_text}</div>
                <hr style="margin: 15px 0; border-color: rgba(255,255,255,0.05);">
                <div class="metric-title">Success Probability</div>
                <div class="metric-value" style="font-size: 48px; color: #fafafa;">{prob_pct:.1f}%</div>
                <div class="metric-delta">Model predicts a {prob_pct:.1f}% chance of regaining possession within 5s of pressure initiation.</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Visualizing the pitch point
            st.markdown("##### 📍 Location on Pitch")
            # Drawing a small mock pitch in SVG for high speed rendering!
            svg_pitch = f"""
            <svg width="100%" height="150" viewBox="0 0 120 80" style="background:#1a1c23; border:1px solid #2d3748; border-radius:8px;">
                <!-- Pitch lines -->
                <line x1="60" y1="0" x2="60" y2="80" stroke="#4a5568" stroke-width="1"/>
                <circle cx="60" cy="40" r="10" fill="none" stroke="#4a5568" stroke-width="1"/>
                <rect x="0" y="18" width="18" height="44" fill="none" stroke="#4a5568" stroke-width="1"/>
                <rect x="102" y="18" width="18" height="44" fill="none" stroke="#4a5568" stroke-width="1"/>
                <!-- Press point -->
                <circle cx="{x_coord}" cy="{y_coord}" r="3" fill="{result_color}" stroke="#fafafa" stroke-width="0.5"/>
            </svg>
            """
            st.markdown(svg_pitch, unsafe_allow_html=True)
            st.caption("Point shows the simulated pressing location (left-to-right attack).")

if __name__ == "__main__":
    main()
