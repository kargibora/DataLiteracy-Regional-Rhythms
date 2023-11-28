from typing import List, Tuple, Union
import datetime
import pandas as pd

def get_regional_data(df : pd.DataFrame , countries: List[str]):
    """Get the data for the given countries"""
    return df[df["region"].isin(countries)]

def get_date_data(df : pd.DataFrame, date: Union[datetime.date, Tuple[datetime.date, datetime.date]]):
    """Get the data for the given date
    If tuple is given, return the data between the dates"""
    if isinstance(date, (tuple,list)):
        return df[(df["date"] >= date[0]) & (df["date"] <= date[1])]
    else:
        return df[df["date"] == date]
    
def convert_date_to_datetime(df : pd.DataFrame, date_only : bool = False):
    """Convert the date column to datetime"""
    df["date"] = pd.to_datetime(df["date"])
    if date_only:
        df["date"] = df["date"].dt.date
    return df

