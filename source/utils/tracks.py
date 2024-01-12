import pandas as pd
from typing import Union, Tuple, List, Optional, Dict
import tqdm

def get_track_title(tracks_df : pd.DataFrame, track_id : str) -> str:
    """
    Get the track name from the tracks dataframe.
    """
    return tracks_df[tracks_df['track_id'] == track_id]['title'].iloc[0]

def get_track_artist(tracks_df : pd.DataFrame, track_id : str) -> str:
    """
    Get the track artist from the tracks dataframe.
    """
    return tracks_df[tracks_df['track_id'] == track_id]['artist'].iloc[0]
