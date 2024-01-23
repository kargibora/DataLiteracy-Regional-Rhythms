"""
This utility file contains functions to work with regional dataframes.
Regional dataframes is defined as the chart dataframe that is filtered by a single region.
"""

from typing import Union, Tuple, Optional, Dict, List, Any
import pandas as pd
import numpy as np
import tqdm
from sklearn.cluster import KMeans
from collections import Counter

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
                                   normalize_streams : bool = False) -> pd.DataFrame:
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

def get_regional_weekly_charts_ranking(df : pd.DataFrame, date : Union[str, Tuple[str,str]], operation : str = 'sum', normalize_streams : bool = True) -> pd.DataFrame:
    """
    Transform daily data into weekly data. Rearange the rankings by getting the mean of streams and assigning a
    new ranking based on the mean. 

    Parameters:
        df (pd.DataFrame): The dataframe to filter. Should be the charts dataframe
        date (str or tuple): The date or the tuple of dates to filter the dataframe
        operation (str): The operation to perform on the streams. Either 'mean' or 'sum'

    Returns:
        pd.DataFrame: The filtered dataframe
    """
    # This should be a view and should not change the original df
    df = df.copy()
    df = get_charts_by_date(df, date)

    temp_df = df[['track_id', 'streams']].copy()


    if operation == 'mean':
        df_group = temp_df.groupby(['track_id']).mean().reset_index()
    elif operation == 'sum':
        df_group = temp_df.groupby(['track_id']).sum().reset_index()
    else:
        raise ValueError("Operation should be either 'mean' or 'sum'")
    
    # Sort the tracks by streams to get the rankings
    df_group = df_group.sort_values(by=['streams'], ascending=False).reset_index()

    # Assign the new rankings
    df_group['rank'] = df_group.index + 1

    # Merge the two dataframes so that df_group also include title, artist, region
    df_group = df_group.merge(df[['track_id', 'title', 'artist', 'region']], on='track_id', how='left')

    # Remove duplicates will look same after we merge the two dataframes
    df_group = df_group.drop_duplicates(subset=['track_id'])

    df_group["start_date"] = date[0] if isinstance(date,tuple) else date
    df_group["end_date"] = date[1] if isinstance(date,tuple) else date

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

def get_region_influence_ranking(similarity_matrix : np.ndarray, regions : List[str]) -> pd.DataFrame:
    """
    Calculate the influence of each region based on the similarity matrix.

    Parameters:
        similarity_matrix (np.ndarray): The similarity matrix for the regions. The shape should be (num_dates, num_regions, num_regions).
        regions (List[str]): The list of regions.
    
    Returns:
        pd.DataFrame: The dataframe containing the region and influence.
    """

    # create the dataframe
    df = pd.DataFrame(columns=['region', 'influence'])
    # create the region array
    region_array = np.array(regions)
    # create the region influence array
    region_influence_array = np.zeros(len(regions))

    # calculate the mean of the similarity matrix across the dates
    similarity_matrix_mean = np.mean(similarity_matrix, axis=0)

    # calculate the influence for each region but skip the diagonal
    for i in range(similarity_matrix_mean.shape[0]):
        # calculate the influence for the region
        region_influence = np.sum(similarity_matrix_mean[i, :]) - similarity_matrix_mean[i, i]
        # add the influence to the region influence array
        region_influence_array[i] = region_influence

    # assign the values to the dataframe
    df['region'] = region_array
    df['influence'] = region_influence_array
    # sort the dataframe by the influence
    df = df.sort_values(by=['influence'], ascending=False).reset_index(drop=True)
    return df

def k_means_clustering(similarity_matrix : np.ndarray, regions : List[str], num_clusters : int) -> Dict[int, List[str]]:
    """
    Perform k-means clustering on the similarity matrix.

    Parameters:
        similarity_matrix (np.ndarray): The similarity matrix for the regions. The shape should be (num_dates, num_regions, num_regions).
        num_clusters (int): The number of clusters to create.
    
    Returns:
        Dict[int, List[str]]: The dictionary containing the cluster number and the list of regions in the cluster.
    """

    # find the average similarity matrix across the dates for each region
    similarity_matrix_mean = np.mean(similarity_matrix, axis=0)

    # perform k-means clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=0)
    kmeans.fit(similarity_matrix_mean)
    cluster_labels = kmeans.predict(similarity_matrix_mean)

    # Create a dictionary mapping country names to cluster labels
    cluster_dict = dict(zip(regions, cluster_labels))

    # Count the occurrences of each cluster label
    cluster_counts = Counter(cluster_labels)

    # Print the length of each cluster
    for cluster_label, count in cluster_counts.items():
        print(f"Cluster {cluster_label}: {count}")

    return cluster_dict

    

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
