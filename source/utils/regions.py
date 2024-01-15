"""
This utility file contains functions to work with regional dataframes.
Regional dataframes is defined as the chart dataframe that is filtered by a single region.
"""

from typing import Union, Tuple, Optional, Dict
import pandas as pd
import tqdm
from .charts import get_charts_by_date, get_charts_by_region
from .datetime import datetime_start_end_generator

def _assert_regional_df(df : pd.DataFrame):
    assert 'region' in df.columns, "The dataframe should have a 'region' column"
    assert 'date' in df.columns, "The dataframe should have a 'date' column"
    assert 'track_id' in df.columns, "The dataframe should have a 'track_id' column"
    assert 'streams' in df.columns, "The dataframe should have a 'streams' column"
    assert 'rank' in df.columns, "The dataframe should have a 'rank' column"
    assert df['region'].nunique() == 1, "The regional dataframe should be filtered by a single region"

def assert_regional_wrapper(func):
    def wrapper(*args, **kwargs):
        _assert_regional_df(args[0])
        return func(*args, **kwargs)
    return wrapper

@assert_regional_wrapper
def get_regional_charts_delta_rank(regional_df : pd.DataFrame,
                                   date : Union[str, Tuple[str,str]],
                                   operation : str = 'sum',
                                   normalize_streams : bool = True) -> pd.DataFrame:
    """
    Transform daily data to weekly data by summing or averaging the streams. Delta rank is defined as
    the ranking of a track between a date range.  Use :operation: to specify whether to sum or average
    the streams. Use :normalize_streams: to normalize the streams between 0 and 1.

    Parameters:
        df (pd.DataFrame): The dataframe to filter. Should be the charts dataframe
        date (str or tuple): The date or the tuple of dates to filter the dataframe
        operation (str): The operation to perform on the streams. Either 'mean' or 'sum'
        normalize_streams (bool): Whether to normalize the streams between 0 and 1

    Returns:
        pd.DataFrame: The filtered dataframe

    Note:
    - This function should be used for a region subset of the dataframe.
    """


    # This should be a view and should not change the original df
    df = regional_df.copy()
    df = get_charts_by_date(df, date)
    if operation == 'mean':
        df_group = df.groupby(['track_id']).mean().reset_index()
    elif operation == 'sum':
        df_group = df.groupby(['track_id']).sum().reset_index()
    else:
        raise ValueError("Operation should be either 'mean' or 'sum'")
    
    # Sort the tracks by streams to get the rankings
    df_group = df_group.sort_values(by=['streams'], ascending=False).reset_index()

    # Assign the new rankings
    df_group['delta_rank'] = df_group.index + 1
    df_group.drop(columns=['rank'], inplace=True)
    
    # Merge the two dataframes to get the title and artist
    df_group = df_group.merge(df.drop(columns=['rank','streams']), on='track_id', how='left')

    # Remove duplicates will look same after we merge the two dataframes
    df_group = df_group.drop_duplicates(subset=['track_id'])


    # Normalize the streams between 0 and 1
    if normalize_streams:
        df_group['streams'] = (df_group['streams'] - df_group['streams'].min()) / (df_group['streams'].max() - df_group['streams'].min())

    return df_group

@assert_regional_wrapper
def calculate_regional_popularity(regional_df : pd.DataFrame, delta_k : int = 10):
    """
    Given the regional dataframe, calculate the popularity score of each track. If weight for each track_id is given,
    calculate the weighted popularity score.
    """
    popularities = {}

    for track_id in tqdm.tqdm(regional_df['track_id'].unique()):
        # Get the track
        track_df = regional_df[regional_df['track_id'] == track_id]

        # Calculate the popularity scores.
        score = track_df[track_df["rank"] <= delta_k].shape[0]

        # Popularity is the score weighted by the average rank
        popularities[track_id] = score 

    return popularities

@assert_regional_wrapper
def calculate_popularity_metrics(regional_df : pd.DataFrame,
                                 date : Tuple[str, str],
                                 delta_k: int = 200) -> pd.DataFrame:
    """
    Calculate popularity metrics for tracks within a specific region and date range.
    
    This function calculates two metrics:
    - The popularity score, which is the number of days a track is within the top delta_k ranks.
    - The average stream proportion for days when the track is within the top delta_k ranks.

    Args:
        charts_df: A DataFrame containing the chart data with columns 'track_id', 'date', 'streams', and 'rank'.
        region: The region for which to calculate the metrics.
        date_range: A tuple containing the start and end date as strings in the format 'YYYY-MM-DD'.
        delta_k: The threshold rank for considering a track as popular.

    Returns:
        A DataFrame similar to charts_df with additional columns 'popularity' and 'average_stream_proportion'.
    
    Example:
        >>> calculate_popularity_metrics(charts_df, 'United States', ('2017-01-01', '2019-12-31'), 25)
    """
    # Filter the DataFrame for the given region and date range
    date_filtered_df = get_charts_by_date(regional_df, date)
    region = regional_df['region'].unique()[0]

    # Calculate stream proportion for each day
    date_filtered_df['stream_proportion'] = date_filtered_df.groupby('date')['streams'].transform(lambda x: x / x.sum())
    
    # Calculate popularity and stream proportion average for each track
    popularities = date_filtered_df[date_filtered_df["rank"] <= delta_k].groupby('track_id').size()
    stream_proportion_average = date_filtered_df[date_filtered_df["rank"] <= delta_k].groupby('track_id')['stream_proportion'].mean()

    # Map the calculated metrics back to the DataFrame
    date_filtered_df['popularity'] = date_filtered_df['track_id'].map(popularities).fillna(0)
    date_filtered_df['average_stream_proportion'] = date_filtered_df['track_id'].map(stream_proportion_average).fillna(0)
    
    return date_filtered_df

@assert_regional_wrapper
def calculate_popularity_metrics_delta(
        regional_df : pd.DataFrame,
        date_range : Tuple[str, str],
        delta_k : int,
        delta_t : int = 6,
):
    datetime_generator =  datetime_start_end_generator(date_range[0], date_range[1], delta_t)
    weekly_dict = {}
    for start_date, end_date in datetime_generator:
        weekly_popularity = calculate_popularity_metrics(regional_df, (start_date, end_date), delta_k)
        weekly_popularity['weighted_popularity'] = weekly_popularity['popularity'] * weekly_popularity['average_stream_proportion']
        weekly_dict[start_date] = weekly_popularity
    return weekly_dict
