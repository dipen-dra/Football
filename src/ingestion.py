import os
import pandas as pd
from statsbombpy import sb
import warnings

# Suppress statsbombpy credentials warnings if any
warnings.filterwarnings("ignore", message="credentials were not supplied")

def get_competitions() -> pd.DataFrame:
    """
    Fetch all available competitions from StatsBomb Open Data.
    Returns:
        pd.DataFrame: Competitions data.
    """
    return sb.competitions()

def get_matches(competition_id: int, season_id: int) -> pd.DataFrame:
    """
    Fetch matches for a specific competition and season.
    Args:
        competition_id (int): Competition ID.
        season_id (int): Season ID.
    Returns:
        pd.DataFrame: Matches data.
    """
    return sb.matches(competition_id=competition_id, season_id=season_id)

def get_match_events(match_id: int, cache_dir: str = "data/raw/events") -> pd.DataFrame:
    """
    Fetch events for a single match and cache them locally in Parquet format.
    Args:
        match_id (int): Match ID.
        cache_dir (str): Directory where raw event files are cached.
    Returns:
        pd.DataFrame: Events data for the match.
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"events_{match_id}.parquet")
    
    if os.path.exists(cache_path):
        try:
            return pd.read_parquet(cache_path)
        except Exception as e:
            # If parquet is corrupted, refetch
            pass

    # Fetch from API
    events_df = sb.events(match_id=match_id)
    
    # Coordinates in location columns need to be serializable.
    # statsbombpy returns columns like 'location' and 'pass_end_location' as list [x, y].
    # Parquet supports nested columns, but list elements can sometimes be tricky.
    # We keep them as is since pandas to_parquet handles list columns fine in modern pyarrow.
    events_df.to_parquet(cache_path, index=False)
    return events_df

def load_all_events_for_matches(
    matches_df: pd.DataFrame, 
    cache_dir: str = "data/raw/events", 
    progress_callback=None
) -> pd.DataFrame:
    """
    Load all events for a set of matches, fetching and caching them as needed.
    Args:
        matches_df (pd.DataFrame): DataFrame of matches (must contain 'match_id').
        cache_dir (str): Directory where raw event files are cached.
        progress_callback (callable, optional): Callback function for progress, takes (current, total, message).
    Returns:
        pd.DataFrame: Combined events data for all matches.
    """
    all_events = []
    total_matches = len(matches_df)
    
    for i, match_id in enumerate(matches_df['match_id']):
        home_team = matches_df.iloc[i]['home_team']
        away_team = matches_df.iloc[i]['away_team']
        match_date = matches_df.iloc[i]['match_date']
        
        msg = f"Loading events for {home_team} vs {away_team} ({match_date})"
        if progress_callback:
            progress_callback(i, total_matches, msg)
            
        events_df = get_match_events(match_id, cache_dir=cache_dir)
        # Ensure match_id is a column in the events DataFrame
        events_df['match_id'] = match_id
        all_events.append(events_df)
        
    if progress_callback:
        progress_callback(total_matches, total_matches, "Finished loading all events.")
        
    if not all_events:
        return pd.DataFrame()
        
    # Concatenate all matches' events.
    # StatsBomb events from different matches might have slightly different columns,
    # so we concatenate with outer join.
    combined_df = pd.concat(all_events, ignore_index=True, sort=False)
    return combined_df
