"""
This script builds up a dataset by using Spotify API to get necessary tracks and its corresponding
    audio features. It also processes the charts dataframe to remove the tracks that are not in the
    audio features dataframe.
"""
import pandas as pd
import spotipy
from spotipy.oauth2 import SpotifyOAuth

from tqdm import tqdm
import yaml
import os
import os.path as osp

# Functions to get the audio features of tracks
from typing import List, Dict, Any
import time
from functools import wraps

import argparse


SECRETS_PATH = "./secrets.yaml"
CHARTS_PATH = "../data/charts.csv"

AUDIO_FEATURES_COLUMNS = [
    'acousticness',
    'danceability',
    'duration_ms',
    'energy',
    'instrumentalness',
    'key',
    'liveness',
    'loudness',
    'mode',
    'speechiness',
    'tempo',
    'time_signature',
    'valence',
    'id'
]

def parse_args():
    parser = argparse.ArgumentParser(description='Build the dataset.')
    parser.add_argument('--secrets_path', type=str, default=SECRETS_PATH,
                        help='The path to the secrets.yaml file.')
    parser.add_argument('--charts_path', type=str, default=CHARTS_PATH,
                        help='The path to the charts.csv file.')
    parser.add_argument('--save_path', type=str, default='../data',
                        help='The path to save the audio features and processed charts.')
    args = parser.parse_args()
    return args

def load_secrets(secrets_path : str):
    """
    Safely load the secrets from the secrets.yaml file.

    Parameters:
        secrets_path (str): The path to the secrets.yaml file.

    Returns:
        dict: The secrets dictionary.
    """
    with open(secrets_path, 'r') as stream:
        try:
            secrets = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return secrets

def build_spotify_object(secrets : dict) -> spotipy.Spotify:
    ""
    sp_auth = SpotifyOAuth(
        client_id=secrets['CLIENT_ID'],
        client_secret=secrets['CLIENT_SECRET'],
        redirect_uri=secrets['REDIRECT_URI'],
        scope=secrets['SCOPE'], 
    )
    return spotipy.Spotify(auth_manager=sp_auth)

def sleep_wrapper(t = 0.33):
    """
    Wrapper function to sleep for a certain amount of time after each call to the Spotify API.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            time.sleep(t)
            return func(*args, **kwargs)
        return wrapper
    return decorator

@sleep_wrapper(0.66)
def get_audio_feature_api(sp : spotipy.Spotify, track_ids: List[str]) -> Dict[str, Any]:
    """
    Get the audio features of a track by using Spotify API.
    This call slows down the code by some specified seconds so that
    Spotify API does not block the requests.
    """
    return sp.audio_features(track_ids)

def get_audio_features(sp : spotipy.Spotify, track_ids: List[str], tracks_per_call : int = 100) -> List[Dict[str, Any]]:
    """
    Get the audio features of a list of tracks.
    """
    # Split the track ids into chunks of 100
    track_ids_chunks = [track_ids[i:i + tracks_per_call] for i in range(0, len(track_ids), tracks_per_call)]
    # Get the audio features of each chunk
    audio_features = []
    for chunk in tqdm(track_ids_chunks):
        try:
            audio_features.extend(get_audio_feature_api(sp, chunk))
        except Exception as e:
            print(e)
            return audio_features
    return audio_features


if __name__ == "__main__":
    args = parse_args()

    sp = build_spotify_object(load_secrets(args.secrets_path))

    # Load the charts
    charts = pd.read_csv(args.charts_path)
    charts.head()

    # Get the all unique track ids from the charts
    unique_track_ids = charts["url"].apply(lambda x: x.split("/")[-1]).unique().tolist()

    # Test calls
    try:
        audio_features = get_audio_features(sp, unique_track_ids[:1], tracks_per_call = 100)
    except Exception as e:
        print(e)

    # For all unique track ids, get the audio features
    audio_features = get_audio_features(sp, unique_track_ids, tracks_per_call = 100)

    # Remove all the None values
    audio_features = [af for af in audio_features if af is not None]

    # Create a dataframe from the audio features
    audio_features_df = pd.DataFrame(audio_features)

    # Select only the important columns
    audio_features_df = audio_features_df[AUDIO_FEATURES_COLUMNS]

    # Save the audio features
    audio_features_df.to_csv(f'{osp.join(args.save_path,"audio_features.csv")}', index = False)

    audio_features_track_ids = audio_features_df["id"].tolist()

    # Drop all the tracks that are not in the charts
    charts = charts[charts["url"].apply(lambda x: x.split("/")[-1]).isin(audio_features_track_ids)]

    # Check whether urls in charts and audio features are the same set
    assert set(charts["url"].apply(lambda x: x.split("/")[-1]).tolist()) == set(audio_features_track_ids)

    # Save the charts
    charts.to_csv(f"{osp.join(args.save_path,'charts_processed.csv')}", index = False)