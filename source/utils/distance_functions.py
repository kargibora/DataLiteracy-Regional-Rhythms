"""
This module contains various distance functions and similarity measures for analyzing music data.

Functions:
- get_distance_intersection: Calculates the distance between two dataframes based on the number of songs that are in both dataframes.
- get_distance_jackard: Calculates the Jaccard distance between two dataframes based on the songs that are in both dataframes.
- get_distance_ranking: Calculates the distance between two dataframes based on the difference in rankings of the common songs.
- kendall_tau_dist: Kendall Tau distance between two permutations.
- get_distance_kendalltau: Calculates the Kendall Tau distance between two dataframes based on the rankings of the common songs.
- get_distance_linear_combination: Returns a function that calculates the distance between two dataframes based on the linear combination of the Jaccard distance and the Kendall Tau distance.
- get_distance_multiplicative_combination: Returns a function that calculates the distance between two dataframes based on the multiplicative combination of the Jaccard distance and the Kendall Tau distance.

Test Functions:
- test_kendall_tau_dist: Tests the custom implementation of Kendall Tau distance function.
- test_get_distance_kendalltau: Tests the get_distance_kendalltau function with different scenarios

Note: This module requires the pandas library to be installed.
"""
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Any, Callable
import datetime
from tqdm import tqdm
from .charts import get_charts_by_region
from .regions import get_regional_weekly_charts_ranking


def get_distance_intersection(df1 : pd.DataFrame, df2 : pd.DataFrame) -> float:
    """
    Calculates the distance between two dataframes based on the number of songs that are in both dataframes.

    Parameters:
        df1 (pd.DataFrame): The first dataframe
        df2 (pd.DataFrame): The second dataframe

    Returns:
        float: The distance between the two dataframes (range: [-len(df1), 0])

    """
    return -len(set(df1["track_id"]).intersection(set(df2["track_id"]))) 

def get_distance_jackard(df1 : pd.DataFrame, df2 : pd.DataFrame) -> float:
    """
    Calculates the Jaccard distance (IOU) between two dataframes based on the songs that are in both dataframes.

    Parameters:
        df1 (pd.DataFrame): The first dataframe
        df2 (pd.DataFrame): The second dataframe
    
    Returns:
        float: The Jaccard distance between the two dataframes (range: [0, 1])

    Note:
    - The Jaccard distance is defined as 1 - Jaccard similarity.
    """

    # find the songs that are in both dataframes
    intersection = len(set(df1["track_id"]).intersection(set(df2["track_id"])))
    # find the songs that are in either of the dataframes
    union = len(set(df1["track_id"]).union(set(df2["track_id"])))
    # return the distance
    return 1 - intersection / union

def get_distance_ranking(df1 : pd.DataFrame, df2 : pd.DataFrame) -> float:
    """
    Calculates the distance between two dataframes based on the difference in rankings of the common songs.

    Parameters:
        df1 (pd.DataFrame): The first dataframe
        df2 (pd.DataFrame): The second dataframe

    Returns:
        float: The distance between the two dataframes (range: [0, 1])
    
    Note:
    - The distance is calculated as the average difference between the rankings of the common songs.
    """

    # find the songs that are in both dataframes
    intersection = set(df1["track_id"]).intersection(set(df2["track_id"]))
    # find the songs that are in either of the dataframes
    union = set(df1["track_id"]).union(set(df2["track_id"]))
    # calculate the difference between the rankings
    difference = 0
    for track_id in intersection:
        difference += abs(df1[df1["track_id"] == track_id]["rank"].iloc[0] - df2[df2["track_id"] == track_id]["rank"].iloc[0])
    return difference / len(union)

def kendall_tau_dist(p1: list, p2: list) -> tuple:
    """
    Kendall Tau distance between two permutations.

    Parameters:
        p1 (list): The first permutation
        p2 (list): The second permutation

    Returns:
        tuple: The Kendall Tau distance and normalized distance between the two permutations
    """

    # p1, p2 are 0-based lists or np.arrays permutations
    n = len(p1)
    index_of = [None] * n  # lookup into p2
    for i in range(n):
        v = p2[i]
        index_of[v] = i

    d = 0  # raw distance = number pair mis-orderings
    for i in range(n):  # scan thru p1
        for j in range(i + 1, n):
            if index_of[p1[i]] > index_of[p1[j]]:
                d += 1
    normer = n * (n - 1) / 2.0  # total num pairs
    nd = d / normer  # normalized distance
    return (d, nd)

def get_distance_kendalltau(df1 : pd.DataFrame, df2 : pd.DataFrame) -> float:
    """
    Calculates the Kendall Tau distance between two dataframes based on the rankings of the common songs.

    Parameters:
        df1 (pd.DataFrame): The first dataframe
        df2 (pd.DataFrame): The second dataframe

    Returns:
        float: The distance between the two dataframes (range: [0, 1])

    """


    # find the songs that are in both dataframes
    intersection = set(df1["track_id"]).intersection(set(df2["track_id"]))
    # align so that the each dataframe is from the most popular to the least popular
    df1 = df1[df1["track_id"].isin(intersection)].sort_values(by=['rank'], ascending=True).reset_index()
    df2 = df2[df2["track_id"].isin(intersection)].sort_values(by=['rank'], ascending=True).reset_index()

    # assign translate the ids from hash to int but keep the information about the original id
    df1["track_id"] = df1["track_id"].astype('category').cat.codes
    df2["track_id"] = df2["track_id"].astype('category').cat.codes

    # calculate the kendal tau distance
    kendalltau = kendall_tau_dist(df1["track_id"], df2["track_id"])[1]

    return kendalltau 

def get_similarity_linear_combination(jackard_weight : float = 0.5, kendaltau_weight : float = 0.5) -> callable:
    """
    Returns a function that calculates the similarity between two dataframes based on the linear combination of the Jaccard similarity and the Kendall Tau similarity.

    Parameters:
        jackard_weight (float): The weight of the Jaccard similarity
        kendaltau_weight (float): The weight of the Kendall Tau similarity

    Returns:
        callable: A function that calculates the similarity between two dataframes based on the linear combination of the Jaccard similarity and the Kendall Tau similarity.
    """
    def similarity(df1 : pd.DataFrame, df2 : pd.DataFrame) -> float:
        return jackard_weight * (1-get_distance_jackard(df1, df2)) + kendaltau_weight * (1-get_distance_kendalltau(df1, df2))
    return similarity

def get_similarity_multiplicative_combination() -> callable:
    """
    Returns a function that calculates the similarity between two dataframes based on the multiplicative combination of the Jaccard similarity and the Kendall Tau similarity.

    Returns:
        callable: A function that calculates the similarity between two dataframes based on the multiplicative combination of the Jaccard similarity and the Kendall Tau similarity.
    """
    def similarity(df1 : pd.DataFrame, df2 : pd.DataFrame) -> float:
        return (1 - get_distance_jackard(df1, df2)) * (1-get_distance_kendalltau(df1, df2))
    return similarity

def test_kendall_tau_dist():
    """
    Tests the custom implementation of Kendall Tau distance function.
    """

    assert kendall_tau_dist([0,1,2,3], [0,1,2,3]) == (0, 0)
    assert kendall_tau_dist([0,1,2,3], [3,2,1,0]) == (6, 1)
    assert kendall_tau_dist([0,1,2,3], [1,2,3,0]) == (3, 1/2)

def test_get_distance_kendalltau():
    """"
    Tests the get_distance_kendaltau function with different scenarios.
    """

    df1 = pd.DataFrame({"track_id": [0,1,2,3], "rank": [1,2,3,4]})
    df2 = pd.DataFrame({"track_id": [0,1,2,3], "rank": [1,2,3,4]})
    assert get_distance_kendalltau(df1, df2) == 0

    df1 = pd.DataFrame({"track_id": [0,1,2,3], "rank": [1,2,3,4]})
    df2 = pd.DataFrame({"track_id": [0,1,2,3], "rank": [4,3,2,1]})
    assert get_distance_kendalltau(df1, df2) == 1

    df1 = pd.DataFrame({"track_id": [0,1,2,3], "rank": [1,2,3,4]})
    df2 = pd.DataFrame({"track_id": [0,1,2,3], "rank": [2,3,4,1]})
    assert get_distance_kendalltau(df1, df2) == 1/2

    df1 = pd.DataFrame({"track_id": [0,1,2,3], "rank": [1,2,3,4]})
    df2 = pd.DataFrame({"track_id": [3,2,1,0], "rank": [4,3,2,1]})
    assert get_distance_kendalltau(df1, df2) == 0

    # test when the ids are strings
    df1 = pd.DataFrame({"track_id": ["0","1","2","3"], "rank": [1,2,3,4]})
    df2 = pd.DataFrame({"track_id": ["0","1","2","3"], "rank": [1,2,3,4]})
    assert get_distance_kendalltau(df1, df2) == 0

    df1 = pd.DataFrame({"track_id": ["0","1","2","3"], "rank": [1,2,3,4]})
    df2 = pd.DataFrame({"track_id": ["0","1","2","3"], "rank": [4,3,2,1]})
    assert get_distance_kendalltau(df1, df2) == 1

    # ids are words
    df1 = pd.DataFrame({"track_id": ["a","b","c","d"], "rank": [1,2,3,4]})
    df2 = pd.DataFrame({"track_id": ["a","b","c","d"], "rank": [1,2,3,4]})
    assert get_distance_kendalltau(df1, df2) == 0

    df1 = pd.DataFrame({"track_id": ["a","b","c","d"], "rank": [1,2,3,4]})
    df2 = pd.DataFrame({"track_id": ["a","b","c","d"], "rank": [4,3,2,1]})
    assert get_distance_kendalltau(df1, df2) == 1

def get_similarity_matrix(df : pd.DataFrame, 
                        regions : List[str],  
                        start_date : str, 
                        end_date : str, 
                        mode : str, 
                        similarity_function_name : str,
                        similarity_function : Callable[[pd.DataFrame, pd.DataFrame], float] = get_distance_jackard,
                        ) -> Tuple[np.ndarray, np.ndarray, List[Tuple[datetime.datetime, datetime.datetime]], Dict[str, Any]]:
    """
    This function calculates the similarity matrix for the given regions and the given period using the given similarity function.
    The similarity matrix is a 3D matrix where the first dimension is the time, the second dimension is the regions and the third dimension is the regions again.
    The similarity matrix is calculated for each time period (weekly, monthly or yearly) and the similarity between the regions are calculated using the given similarity function.
    The similarity function should take two dataframes and return a float value.

    Parameters:
        df (pd.DataFrame): The dataframe that contains the charts data.
        regions (List[str]): The list of regions to calculate the similarity matrix.
        start_date (str): The start date of the period.
        end_date (str): The end date of the period.
        mode (str): The mode of the period. It can be either 'weekly', 'monthly' or 'yearly'.
        similarity_function_name (str): The name of the similarity function. It can be either 'kendalltau', 'jackard' or 'euclidean'.
        similarity_function (Callable[[pd.DataFrame, pd.DataFrame], float]): The similarity function to calculate the similarity between the regions. It should take two dataframes and return a float value. Default is get_distance_jackard.

    Returns:
        Tuple[np.ndarray, np.ndarray, List[Tuple[datetime.datetime, datetime.datetime]], Dict[str, Any]]: The similarity matrix, the region array, the date ranges and the info dictionary.

    """

    # create the region array
    region_array = np.array(regions)
    # get the dates according to the mode
    if mode == "weekly":
        dates = pd.date_range(start=start_date, end=end_date, freq='W-MON')
    elif mode == "monthly":
        dates = pd.date_range(start=start_date, end=end_date, freq='MS')
    elif mode == "yearly":
        dates = pd.date_range(start=start_date, end=end_date, freq='AS')
    else:
        raise ValueError("Mode should be either 'weekly', 'monthly' or 'yearly'")
    
    # create a tuple of the date ranges
    date_ranges = [(dates[i], dates[i+1]) for i in range(len(dates)-1)]

    # create the similarity matrix including the dates
    similarity_matrix = np.zeros((len(date_ranges), len(regions), len(regions)))

    # use tqdm to show the progress bar
    for i, date_range in tqdm(enumerate(date_ranges), desc=f"Calculating similarity matrix for {mode} mode using {similarity_function_name} similarity function", total=len(date_ranges)):
        # get the regional weekly charts for the date range
        regional_charts = [get_regional_weekly_charts_ranking(get_charts_by_region(df, region), date_range) for region in regions]
        # calculate the difference between the regions
        for j, region1 in enumerate(regions):
            for k, region2 in enumerate(regions):
                if region1 == region2:
                    similarity_matrix[i, j, k] = 0
                # since the matrix is diagonal if the similarity is already calculated just copy it
                elif k < j:
                    similarity_matrix[i, j, k] = similarity_matrix[i, k, j]
                else:
                    # append the difference to the dictionary value which is an array
                    similarity_matrix[i, j, k] = similarity_function(regional_charts[j], regional_charts[k])

    # also return an information dictionary that contains the period, aggreagate mode (daily, weekly, monthly, yearly) and the difference function
    info_dict = {"period": (start_date, end_date), "mode": mode, "similarity_function": similarity_function_name}
            
    return similarity_matrix, region_array, date_ranges, info_dict