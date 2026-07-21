import os
import sys
import pandas as pd
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Add src to python path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.ingestion import get_competitions, get_matches, load_all_events_for_matches
from src.features import build_features_pipeline
from src.modeling import train_success_model, train_outcome_model

def run_verification():
    print("=== STARTING PIPELINE VERIFICATION & BOOTSTRAPPING ===")
    
    # 1. Fetch competitions
    print("\n[Step 1] Fetching competitions...")
    try:
        comps = get_competitions()
        print(f"Success! Fetched {len(comps)} competitions.")
        print(comps[['competition_id', 'season_id', 'competition_name', 'season_name']].head(10))
    except Exception as e:
        print(f"Error fetching competitions: {e}")
        return False

    # 2. Pick a competition (FIFA Women's World Cup 2019: comp_id=72, season_id=30)
    # If not present, we will pick the first available competition that has matches
    comp_id = 72
    season_id = 30
    
    match_comp = comps[(comps['competition_id'] == comp_id) & (comps['season_id'] == season_id)]
    if len(match_comp) == 0:
        print(f"\nWarning: Women's World Cup 2019 (72, 30) not found in competitions. Picking first available...")
        comp_id = comps.iloc[0]['competition_id']
        season_id = comps.iloc[0]['season_id']
        comp_name = comps.iloc[0]['competition_name']
        season_name = comps.iloc[0]['season_name']
    else:
        comp_name = match_comp.iloc[0]['competition_name']
        season_name = match_comp.iloc[0]['season_name']
        
    print(f"\nSelected competition: {comp_name} ({season_name}) - ID: {comp_id}, Season: {season_id}")
    
    # 3. Fetch matches
    print("\n[Step 2] Fetching matches...")
    try:
        matches = get_matches(comp_id, season_id)
        print(f"Success! Found {len(matches)} matches.")
        print(matches[['match_id', 'match_date', 'home_team', 'away_team', 'home_score', 'away_score']].head(5))
    except Exception as e:
        print(f"Error fetching matches: {e}")
        return False
        
    if len(matches) == 0:
        print("Error: No matches found for this competition.")
        return False
        
    # We will fetch a subset (e.g. 5 matches) for rapid verification/bootstrapping
    sample_size = min(5, len(matches))
    sample_matches = matches.head(sample_size)
    print(f"\nRunning pipeline on a sample of {sample_size} matches...")
    
    # 4. Load events
    print("\n[Step 3] Loading match events (caching to disk)...")
    try:
        events = load_all_events_for_matches(sample_matches, cache_dir="data/raw/events")
        print(f"Success! Loaded {len(events)} events.")
    except Exception as e:
        print(f"Error loading events: {e}")
        return False
        
    # 5. Run feature pipeline
    print("\n[Step 4] Running feature extraction pipeline...")
    try:
        events_features, matches_features = build_features_pipeline(events, sample_matches)
        
        # Save processed features
        os.makedirs("data/processed", exist_ok=True)
        events_features.to_parquet("data/processed/events_features.parquet", index=False)
        matches_features.to_parquet("data/processed/matches_features.parquet", index=False)
        
        print("Success! Features extracted and saved to data/processed/")
        print(f"Pressures extracted (event-level): {len(events_features)} rows.")
        print(f"Match aggregations (match-level): {len(matches_features)} rows.")
        
        if len(events_features) > 0:
            print("\nSample pressures features:")
            print(events_features[['team', 'x', 'y', 'zone', 'trigger', 'is_counter_press', 'success', 'danger']].head(5))
        if len(matches_features) > 0:
            print("\nSample match features:")
            print(matches_features[['team', 'opponent', 'ppda', 'pressures_count', 'pressing_success_rate', 'match_result']].head(5))
            
    except Exception as e:
        print(f"Error in feature pipeline: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    # 6. Train models
    print("\n[Step 5] Training models on extracted features...")
    try:
        print("Training Pressing Success Classifier...")
        success_model_data = train_success_model(events_features, models_dir="models")
        print("Success! Success Classifier trained.")
        print(f"XGBoost Test Accuracy: {success_model_data['xgb_metrics']['accuracy']:.4f}")
        print(f"XGBoost Test ROC-AUC: {success_model_data['xgb_metrics']['auc']:.4f}")
        
        print("\nTraining Match Outcome Predictor...")
        outcome_model_data = train_outcome_model(matches_features, models_dir="models")
        print("Success! Match Outcome Predictor trained.")
        print(f"Random Forest Test Accuracy: {outcome_model_data['rf_metrics']['accuracy']:.4f}")
    except Exception as e:
        print(f"Error training models: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    print("\n=== PIPELINE VERIFICATION COMPLETED SUCCESSFULLY ===")
    return True

if __name__ == "__main__":
    success = run_verification()
    sys.exit(0 if success else 1)
