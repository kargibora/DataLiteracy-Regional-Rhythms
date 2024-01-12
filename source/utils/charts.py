"""
This utility file includes some functions to filter the charts dataframe.
"""

from typing import Union, Tuple, List, Optional, Dict
import pandas as pd

def get_charts_by_date(df : pd.DataFrame, date : Union[str, Tuple[str,str]]) -> pd.DataFrame:
    """
    Get the elements of the dataframe that match the date.
    If date is a tuple, return the elements between the two dates.

    Parameters:
        df (pd.DataFrame): The dataframe to filter
        date (str or tuple): The date or the tuple of dates to filter the dataframe

    Returns:
        pd.DataFrame: The filtered dataframe
    """
    # If the date is consist of strings, transform it to pd.DateTime format %Y-%m-%d
    if isinstance(date, tuple) and isinstance(date[0], str):
        date = (pd.to_datetime(date[0]), pd.to_datetime(date[1]))
    elif isinstance(date, str):
        date = pd.to_datetime(date)

    # If date is a string, transform it to a pd.datetime
    if isinstance(date,tuple):
        return df[(df['date'] >= date[0]) & (df['date'] <= date[1])]
    else:
        return df[df['date'] == date]

def get_charts_by_region(df : pd.DataFrame, region : Union[str, List[str]], seperete_dict : Optional[bool] = False) -> pd.DataFrame:
    """
    Return the charts by region.
    If region is a list, return the charts for the regions in the list.

    Parameters:
        df (pd.DataFrame): The dataframe to filter
        region (str or list): The region or the list of regions to filter the dataframe
        seperete_dict (bool): If True, return a dictionary of dataframes seperated by region

    Returns:
        pd.DataFrame: The filtered dataframe
    """
    seperation = {}
    if seperete_dict:
        for region in df['region'].unique():
            seperation[region] = df[df['region'] == region]
        return seperation
    else:
        if isinstance(region, list):
            return df[df['region'].isin(region)]
        return df[df['region'] == region]
    