import os
import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    roc_auc_score, confusion_matrix, log_loss
)
import warnings
from typing import Dict, Any, Tuple

warnings.filterwarnings("ignore")

# Define features and mapping for the Success Classifier
TRIGGER_MAPPING = {
    'Pass Reception / Build-up': 0,
    'Poor Touch / Miscontrol': 1,
    'Backward Pass': 2,
    'Carrying / Dribbling': 3,
    'Throw-in': 4,
    'Ball Recovery / Defensive Action': 5
}

def preprocess_success_data(events_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Filter, clean, and encode event data for Pressing Success model.
    """
    df = events_df.copy()
    
    # Drop rows with missing crucial features
    df = df.dropna(subset=['x', 'y', 'distance_to_goal', 'success'])
    
    # Encode trigger
    df['trigger_encoded'] = df['trigger'].map(TRIGGER_MAPPING).fillna(0).astype(int)
    
    # Feature columns
    feature_cols = [
        'x', 'y', 'distance_to_goal', 'is_counter_press', 
        'score_diff', 'minute', 'trigger_encoded', 'under_pressure'
    ]
    
    X = df[feature_cols]
    y = df['success']
    
    return X, y

def train_success_model(events_df: pd.DataFrame, models_dir: str = "models") -> Dict[str, Any]:
    """
    Train Pressing Success Classifier models (Logistic Regression and XGBoost).
    Evaluate, compute SHAP values, and save the models.
    """
    os.makedirs(models_dir, exist_ok=True)
    
    X, y = preprocess_success_data(events_df)
    
    # Handle small datasets gracefully
    if len(X) < 10:
        raise ValueError(f"Insufficient pressure event data ({len(X)} samples) to train model.")
        
    # Split by match or simple stratified split
    # Since we want to prevent leakage, split stratified on target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    
    # Model 1: Logistic Regression (interpretable baseline)
    lr_model = LogisticRegression(max_iter=1000, random_state=42)
    lr_model.fit(X_train, y_train)
    
    # Model 2: XGBoost (high performance)
    # Scale pos weight to handle class imbalance if any
    pos_count = y_train.sum()
    neg_count = len(y_train) - pos_count
    scale_weight = neg_count / pos_count if pos_count > 0 else 1.0
    
    xgb_model = XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.05,
        scale_pos_weight=scale_weight,
        random_state=42,
        eval_metric='logloss'
    )
    xgb_model.fit(X_train, y_train)
    
    # Evaluate Logistic Regression
    lr_preds = lr_model.predict(X_test)
    lr_probs = lr_model.predict_proba(X_test)[:, 1]
    lr_metrics = {
        'accuracy': accuracy_score(y_test, lr_preds),
        'precision': precision_score(y_test, lr_preds, zero_division=0),
        'recall': recall_score(y_test, lr_preds, zero_division=0),
        'auc': roc_auc_score(y_test, lr_probs),
        'conf_matrix': confusion_matrix(y_test, lr_preds, labels=[0, 1]).tolist()
    }
    
    # Evaluate XGBoost
    xgb_preds = xgb_model.predict(X_test)
    xgb_probs = xgb_model.predict_proba(X_test)[:, 1]
    xgb_metrics = {
        'accuracy': accuracy_score(y_test, xgb_preds),
        'precision': precision_score(y_test, xgb_preds, zero_division=0),
        'recall': recall_score(y_test, xgb_preds, zero_division=0),
        'auc': roc_auc_score(y_test, xgb_probs),
        'conf_matrix': confusion_matrix(y_test, xgb_preds, labels=[0, 1]).tolist()
    }
    
    # Calculate SHAP values for XGBoost on test set
    try:
        explainer = shap.TreeExplainer(xgb_model)
        shap_values = explainer.shap_values(X_test)
    except Exception as e:
        explainer = None
        shap_values = None
        print(f"SHAP explanation failed: {e}")
        
    # Save everything
    model_data = {
        'lr_model': lr_model,
        'xgb_model': xgb_model,
        'feature_names': list(X.columns),
        'lr_metrics': lr_metrics,
        'xgb_metrics': xgb_metrics,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'shap_values': shap_values,
        'explainer': explainer
    }
    
    joblib.dump(model_data, os.path.join(models_dir, "pressing_success_model.joblib"))
    return model_data

def preprocess_outcome_data(matches_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Clean and format match outcome dataset.
    Target is match_result: Win (2), Draw (1), Loss (0)
    """
    df = matches_df.copy()
    
    # Target mapping
    outcome_mapping = {'Loss': 0, 'Draw': 1, 'Win': 2}
    y = df['match_result'].map(outcome_mapping)
    
    # Features
    feature_cols = [
        'ppda', 'pressures_count', 'pressures_att_third', 'pressures_mid_third',
        'counter_pressures_count', 'pressing_success_rate', 'dangerous_regains_count',
        'possession_pct', 'xg_scored', 'xg_conceded'
    ]
    
    X = df[feature_cols].copy()
    
    # Clean up NaNs
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0.0)
        
    return X, y

def train_outcome_model(matches_df: pd.DataFrame, models_dir: str = "models") -> Dict[str, Any]:
    """
    Train Match Outcome Predictor models (Logistic Regression vs Random Forest / XGBoost).
    """
    os.makedirs(models_dir, exist_ok=True)
    
    X, y = preprocess_outcome_data(matches_df)
    
    if len(X) < 10:
        raise ValueError(f"Insufficient match data ({len(X)} samples) to train outcome model.")
        
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    
    # Model 1: Multiclass Logistic Regression
    lr_model = LogisticRegression(multi_class='multinomial', max_iter=1000, random_state=42)
    lr_model.fit(X_train, y_train)
    
    # Model 2: Random Forest
    rf_model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    rf_model.fit(X_train, y_train)
    
    # Evaluate Logistic Regression
    lr_preds = lr_model.predict(X_test)
    lr_probs = lr_model.predict_proba(X_test)
    lr_metrics = {
        'accuracy': accuracy_score(y_test, lr_preds),
        'log_loss': log_loss(y_test, lr_probs),
        'conf_matrix': confusion_matrix(y_test, lr_preds, labels=[0, 1, 2]).tolist()
    }
    
    # Evaluate Random Forest
    rf_preds = rf_model.predict(X_test)
    rf_probs = rf_model.predict_proba(X_test)
    rf_metrics = {
        'accuracy': accuracy_score(y_test, rf_preds),
        'log_loss': log_loss(y_test, rf_probs),
        'conf_matrix': confusion_matrix(y_test, rf_preds, labels=[0, 1, 2]).tolist()
    }
    
    # Save everything
    model_data = {
        'lr_model': lr_model,
        'rf_model': rf_model,
        'feature_names': list(X.columns),
        'lr_metrics': lr_metrics,
        'rf_metrics': rf_metrics,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'feature_importances': rf_model.feature_importances_
    }
    
    joblib.dump(model_data, os.path.join(models_dir, "match_outcome_model.joblib"))
    return model_data
