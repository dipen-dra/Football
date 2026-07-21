import numpy as np
import pandas as pd
import warnings
from typing import Tuple, Dict, Any

# Suppress warnings
warnings.filterwarnings("ignore")

def parse_locations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract x and y coordinates from StatsBomb 'location' columns.
    """
    df = df.copy()
    
    def get_coord(l, idx):
        if isinstance(l, (list, tuple, np.ndarray)) and len(l) > idx:
            return float(l[idx])
        return np.nan

    if 'location' in df.columns:
        df['x'] = df['location'].apply(lambda l: get_coord(l, 0))
        df['y'] = df['location'].apply(lambda l: get_coord(l, 1))
    else:
        df['x'] = np.nan
        df['y'] = np.nan
        
    if 'pass_end_location' in df.columns:
        df['pass_end_x'] = df['pass_end_location'].apply(lambda l: get_coord(l, 0))
        df['pass_end_y'] = df['pass_end_location'].apply(lambda l: get_coord(l, 1))
    else:
        df['pass_end_x'] = np.nan
        df['pass_end_y'] = np.nan
        
    return df

def timestamp_to_seconds(ts_str: str) -> float:
    """
    Convert StatsBomb HH:MM:SS.mmm timestamp to seconds.
    """
    if pd.isna(ts_str) or not isinstance(ts_str, str):
        return 0.0
    try:
        parts = ts_str.split(':')
        if len(parts) == 3:
            h, m, s = parts
            return float(h) * 3600.0 + float(m) * 60.0 + float(s)
    except ValueError:
        pass
    return 0.0

def calculate_match_ppda(match_events: pd.DataFrame, home_team: str, away_team: str) -> Tuple[float, float, Dict[str, Any]]:
    """
    Calculate PPDA for home and away teams in a single match.
    Formula: Opponent Completed Passes in Opponent's Defensive 60% / (Team's Defensive Actions in Opponent's Defensive 60%)
    Opponent's Defensive 60% of pitch:
      - For passing team: x <= 72 (since they attack from 0 to 120)
      - For pressing team: x >= 48 (since they attack from 0 to 120, opponent's defensive 60% is their attacking 60%)
    """
    events = parse_locations(match_events)
    
    # 1. Passes
    passes = events[events['type'] == 'Pass'].copy()
    # Completed passes have NaN in pass_outcome
    completed_passes = passes[passes['pass_outcome'].isna()]
    
    # 2. Defensive Actions
    # Standard PPDA defensive actions: Tackles, Interceptions, Challenges, Fouls
    # In StatsBomb:
    # - Duels with duel_type == 'Tackle' (includes successful/unsuccessful tackles and challenges)
    # - Interceptions
    # - Fouls Committed
    def_actions = events[
        (events['type'] == 'Interception') |
        (events['type'] == 'Foul Committed') |
        ((events['type'] == 'Duel') & (events['duel_type'] == 'Tackle'))
    ].copy()
    
    # Calculate PPDA components for Home Team (opponent is Away Team)
    # Home PPDA = Away Completed Passes in Away's Defensive 60% (x <= 72) / Home Defensive Actions in Away's Defensive 60% (x >= 48)
    away_passes_in_zone = completed_passes[(completed_passes['team'] == away_team) & (completed_passes['x'] <= 72)]
    home_def_actions_in_zone = def_actions[(def_actions['team'] == home_team) & (def_actions['x'] >= 48)]
    
    # Calculate PPDA components for Away Team (opponent is Home Team)
    home_passes_in_zone = completed_passes[(completed_passes['team'] == home_team) & (completed_passes['x'] <= 72)]
    away_def_actions_in_zone = def_actions[(def_actions['team'] == away_team) & (def_actions['x'] >= 48)]
    
    home_passes_count = len(away_passes_in_zone)
    home_def_actions_count = len(home_def_actions_in_zone)
    
    away_passes_count = len(home_passes_in_zone)
    away_def_actions_count = len(away_def_actions_in_zone)
    
    home_ppda = home_passes_count / home_def_actions_count if home_def_actions_count > 0 else 99.0
    away_ppda = away_passes_count / away_def_actions_count if away_def_actions_count > 0 else 99.0
    
    debug_info = {
        'home_passes_allowed': home_passes_count,
        'home_def_actions': home_def_actions_count,
        'away_passes_allowed': away_passes_count,
        'away_def_actions': away_def_actions_count
    }
    
    return home_ppda, away_ppda, debug_info

def extract_pressing_events(match_events: pd.DataFrame, home_team: str, away_team: str) -> pd.DataFrame:
    """
    Extract all pressure events in a match, tag counter-presses, identify triggers, and link outcomes.
    """
    events = parse_locations(match_events).copy()
    
    # Sort events to ensure correct temporal sequence
    events['time_sec'] = events['period'] * 3000.0 + events['timestamp'].apply(timestamp_to_seconds)
    events = events.sort_values(by=['period', 'time_sec', 'index']).reset_index(drop=True)
    
    # Calculate score at each event
    # Find goals
    goals = events[(events['type'] == 'Shot') & (events['shot_outcome'] == 'Goal')].copy()
    # Note: Statsbombpy events has a 'possession_team' or 'team'.
    # We will accumulate score
    home_score_seq = np.zeros(len(events))
    away_score_seq = np.zeros(len(events))
    
    home_goals = 0
    away_goals = 0
    
    for idx, row in goals.iterrows():
        if row['team'] == home_team:
            home_goals += 1
        else:
            away_goals += 1
        # Set scores from this event onwards
        # In a real match, we set the score *after* the goal
        # Since we sorted events, we can just track it
        
    # Let's do a running score tracking
    current_home_score = 0
    current_away_score = 0
    home_scores = []
    away_scores = []
    
    for idx, row in events.iterrows():
        if row['type'] == 'Shot' and row['shot_outcome'] == 'Goal':
            if row['team'] == home_team:
                current_home_score += 1
            else:
                current_away_score += 1
        home_scores.append(current_home_score)
        away_scores.append(current_away_score)
        
    events['home_score_state'] = home_scores
    events['away_score_state'] = away_scores
    
    # Identify possession transition timestamps
    # In StatsBomb, a change in 'possession_team' marks a transition.
    events['possession_changed'] = events['possession_team'] != events['possession_team'].shift(1)
    
    # Record transition times
    transition_times = {} # maps possession_id/index to transition time
    current_possession_team = None
    last_transition_time = 0.0
    
    possession_start_times = {} # maps possession number to start time
    for idx, row in events.iterrows():
        pos_num = row['possession']
        if pos_num not in possession_start_times:
            possession_start_times[pos_num] = row['time_sec']
            
    # Add possession start time to all events
    events['possession_start_time'] = events['possession'].map(possession_start_times)
    
    # Keep track of when a team lost possession
    # A team lost possession when the possession team changed.
    # We find all events where possession changed
    possession_changes = events[events['possession_changed'] == True].copy()
    
    # Map from possession number to the time when the *previous* possession ended
    possession_end_times = events.groupby('possession')['time_sec'].max().to_dict()
    
    # Find pressure events
    pressures = events[events['type'] == 'Pressure'].copy()
    if len(pressures) == 0:
        return pd.DataFrame()
        
    pressure_features = []
    
    for idx, press in pressures.iterrows():
        press_team = press['team']
        opp_team = away_team if press_team == home_team else home_team
        press_time = press['time_sec']
        pos_num = press['possession']
        period = press['period']
        
        # 1. Counter-press flag: occurred within 5 seconds of losing possession.
        # Find the last possession of the pressing team before this possession.
        prev_possessions = events[
            (events['period'] == period) & 
            (events['possession'] < pos_num) & 
            (events['possession_team'] == press_team)
        ]
        
        is_counter_press = False
        if len(prev_possessions) > 0:
            last_prev_poss_id = prev_possessions['possession'].max()
            last_prev_poss_end = possession_end_times[last_prev_poss_id]
            time_since_loss = press_time - last_prev_poss_end
            if 0 <= time_since_loss <= 5.0:
                is_counter_press = True
                
        # 2. Extract pressing trigger from events preceding this pressure event
        # Look at the previous 1-3 events within the same possession and same period
        preceding_events = events[
            (events['period'] == period) &
            (events['possession'] == pos_num) &
            (events['index'] < press['index'])
        ].sort_values(by='index', ascending=False)
        
        trigger = "Pass Reception / Build-up" # default
        if len(preceding_events) > 0:
            last_event = preceding_events.iloc[0]
            
            # Check for bad touch/miscontrol
            # Find any bad touch or dispossessed in the last 2 events
            bad_touch_events = preceding_events.iloc[:2]
            bad_touch_events = bad_touch_events[
                (bad_touch_events['type'] == 'Bad Touch') | 
                (bad_touch_events['type'] == 'Dispossessed')
            ]
            
            if len(bad_touch_events) > 0:
                trigger = "Poor Touch / Miscontrol"
            elif last_event['type'] == 'Pass':
                # Check for throw-in
                if last_event.get('pass_type') == 'Throw-in':
                    trigger = "Throw-in"
                # Check for backward pass (end_x < start_x)
                elif pd.notna(last_event['pass_end_x']) and pd.notna(last_event['x']) and last_event['pass_end_x'] < last_event['x']:
                    trigger = "Backward Pass"
                else:
                    trigger = "Pass Reception / Build-up"
            elif last_event['type'] == 'Carry':
                trigger = "Carrying / Dribbling"
            elif last_event['type'] == 'Ball Recovery' or last_event['type'] == 'Interception':
                trigger = "Ball Recovery / Defensive Action"
                
        # 3. Outcome linking: Possession regained within 5 seconds?
        # Look ahead for events where pressing team gains possession or makes a successful recovery
        subsequent_events = events[
            (events['period'] == period) &
            (events['time_sec'] > press_time)
        ].sort_values(by='time_sec').iloc[:30] # look at next 30 events
        
        regain = 0
        regain_time = None
        led_to_shot = 0
        led_to_goal = 0
        
        for s_idx, seq_ev in subsequent_events.iterrows():
            # Check time elapsed since pressure
            time_elapsed = seq_ev['time_sec'] - press_time
            if time_elapsed > 5.0:
                # Regain must happen within 5 seconds
                break
                
            # Regain is identified if:
            # - Possession changes to pressing team (possession_team == press_team)
            # - Pressing team makes a ball recovery, successful tackle, or interception
            if seq_ev['possession_team'] == press_team:
                regain = 1
                regain_time = seq_ev['time_sec']
                break
            elif seq_ev['team'] == press_team and seq_ev['type'] in ['Ball Recovery', 'Interception']:
                regain = 1
                regain_time = seq_ev['time_sec']
                break
            elif seq_ev['team'] == press_team and seq_ev['type'] == 'Duel' and seq_ev.get('duel_outcome') in ['Won', 'Success']:
                regain = 1
                regain_time = seq_ev['time_sec']
                break
                
        # 4. If regained, did it lead to a shot/goal within 15 seconds of regain?
        if regain == 1 and regain_time is not None:
            shot_events = events[
                (events['period'] == period) &
                (events['time_sec'] >= regain_time) &
                (events['time_sec'] <= regain_time + 15.0) &
                (events['team'] == press_team) &
                (events['type'] == 'Shot')
            ]
            if len(shot_events) > 0:
                led_to_shot = 1
                goal_events = shot_events[shot_events['shot_outcome'] == 'Goal']
                if len(goal_events) > 0:
                    led_to_goal = 1
                    
        # 5. Context features
        x = press['x']
        y = press['y']
        
        # Thirds
        if pd.isna(x):
            zone = "Middle Third"
        elif x < 40.0:
            zone = "Defensive Third"
        elif x < 80.0:
            zone = "Middle Third"
        else:
            zone = "Attacking Third"
            
        distance_to_goal = np.sqrt((120.0 - x)**2 + (40.0 - y)**2) if (pd.notna(x) and pd.notna(y)) else np.nan
        
        # Score state from perspective of pressing team
        if press_team == home_team:
            score_diff = press['home_score_state'] - press['away_score_state']
        else:
            score_diff = press['away_score_state'] - press['home_score_state']
            
        pressure_features.append({
            'match_id': press['match_id'],
            'index': press['index'],
            'period': period,
            'minute': press['minute'],
            'second': press['second'],
            'team': press_team,
            'opponent': opp_team,
            'player': press['player'],
            'x': x,
            'y': y,
            'zone': zone,
            'distance_to_goal': distance_to_goal,
            'trigger': trigger,
            'is_counter_press': int(is_counter_press),
            'score_diff': score_diff,
            'success': regain,
            'danger': led_to_shot,
            'goal': led_to_goal,
            'under_pressure': int(press.get('under_pressure', 0) == True or press.get('under_pressure', 0) == 1)
        })
        
    return pd.DataFrame(pressure_features)

def build_features_pipeline(all_events_df: pd.DataFrame, matches_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Processes the raw events and matches data to generate the event-level and match-level feature tables.
    """
    all_pressures = []
    match_aggregations = []
    
    # Group events by match_id
    grouped = all_events_df.groupby('match_id')
    
    for match_id, match_events in grouped:
        # Get match info
        match_info = matches_df[matches_df['match_id'] == match_id]
        if len(match_info) == 0:
            continue
        match_row = match_info.iloc[0]
        
        home_team = match_row['home_team']
        away_team = match_row['away_team']
        home_score = match_row['home_score']
        away_score = match_row['away_score']
        
        # Calculate PPDA
        home_ppda, away_ppda, ppda_debug = calculate_match_ppda(match_events, home_team, away_team)
        
        # Extract pressures
        pressures_df = extract_pressing_events(match_events, home_team, away_team)
        
        if len(pressures_df) > 0:
            all_pressures.append(pressures_df)
            
        # Calculate Team possession percentage
        # StatsBomb events have duration. Let's sum duration of possessions
        possession_durations = match_events.groupby(['possession_team', 'possession'])['duration'].max().reset_index()
        pos_sum = possession_durations.groupby('possession_team')['duration'].sum()
        
        total_duration = pos_sum.sum()
        home_poss_pct = (pos_sum.get(home_team, 0) / total_duration * 100.0) if total_duration > 0 else 50.0
        away_poss_pct = (pos_sum.get(away_team, 0) / total_duration * 100.0) if total_duration > 0 else 50.0
        
        # Shots and xG
        shots = match_events[match_events['type'] == 'Shot'].copy()
        shots['xg'] = pd.to_numeric(shots['shot_statsbomb_xg'], errors='coerce').fillna(0.0)
        
        home_shots_df = shots[shots['team'] == home_team]
        away_shots_df = shots[shots['team'] == away_team]
        
        home_shots = len(home_shots_df)
        away_shots = len(away_shots_df)
        home_xg = home_shots_df['xg'].sum()
        away_xg = away_shots_df['xg'].sum()
        
        # Match outcome
        if home_score > away_score:
            home_result, away_result = 'Win', 'Loss'
            home_pts, away_pts = 3, 0
        elif home_score < away_score:
            home_result, away_result = 'Loss', 'Win'
            home_pts, away_pts = 0, 3
        else:
            home_result, away_result = 'Draw', 'Draw'
            home_pts, away_pts = 1, 1
            
        # Pressing aggregations
        def get_team_pressing_stats(team_name, opp_name, ppda_val, is_home_val, result_val, pts_val, opp_score, own_score, opp_shots, own_shots, own_xg, opp_xg, poss_pct):
            if len(pressures_df) == 0:
                return {
                    'match_id': match_id, 'team': team_name, 'opponent': opp_name, 'is_home': int(is_home_val),
                    'ppda': ppda_val, 'pressures_count': 0, 'pressures_def_third': 0, 'pressures_mid_third': 0,
                    'pressures_att_third': 0, 'counter_pressures_count': 0, 'successful_pressures_count': 0,
                    'pressing_success_rate': 0.0, 'dangerous_regains_count': 0, 'possession_pct': poss_pct,
                    'shots_conceded': opp_shots, 'goals_conceded': opp_score, 'shots_created': own_shots,
                    'goals_scored': own_score, 'xg_scored': own_xg, 'xg_conceded': opp_xg,
                    'match_result': result_val, 'points': pts_val
                }
                
            team_press = pressures_df[pressures_df['team'] == team_name]
            total_p = len(team_press)
            
            p_def = len(team_press[team_press['zone'] == 'Defensive Third'])
            p_mid = len(team_press[team_press['zone'] == 'Middle Third'])
            p_att = len(team_press[team_press['zone'] == 'Attacking Third'])
            
            cp = team_press['is_counter_press'].sum()
            succ = team_press['success'].sum()
            dang = team_press['danger'].sum()
            
            success_rate = (succ / total_p * 100.0) if total_p > 0 else 0.0
            
            return {
                'match_id': match_id,
                'team': team_name,
                'opponent': opp_name,
                'is_home': int(is_home_val),
                'ppda': ppda_val,
                'pressures_count': total_p,
                'pressures_def_third': p_def,
                'pressures_mid_third': p_mid,
                'pressures_att_third': p_att,
                'counter_pressures_count': cp,
                'successful_pressures_count': succ,
                'pressing_success_rate': success_rate,
                'dangerous_regains_count': dang,
                'possession_pct': poss_pct,
                'shots_conceded': opp_shots,
                'goals_conceded': opp_score,
                'shots_created': own_shots,
                'goals_scored': own_score,
                'xg_scored': own_xg,
                'xg_conceded': opp_xg,
                'match_result': result_val,
                'points': pts_val
            }
            
        home_stats = get_team_pressing_stats(
            home_team, away_team, home_ppda, True, home_result, home_pts, 
            away_score, home_score, away_shots, home_shots, home_xg, away_xg, home_poss_pct
        )
        away_stats = get_team_pressing_stats(
            away_team, home_team, away_ppda, False, away_result, away_pts, 
            home_score, away_score, home_shots, away_shots, away_xg, home_xg, away_poss_pct
        )
        
        match_aggregations.append(home_stats)
        match_aggregations.append(away_stats)
        
    combined_pressures = pd.concat(all_pressures, ignore_index=True) if all_pressures else pd.DataFrame()
    combined_matches = pd.DataFrame(match_aggregations)
    
    return combined_pressures, combined_matches
