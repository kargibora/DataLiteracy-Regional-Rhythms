# Wrapper for SpotifyCharts dataset
import pandas as pd
import numpy as np
from typing import Union, List, Optional
from datetime import datetime, timedelta
import pprint

class SpotifyCharts(object):
    """
    A wrapper for using Spotify Charts dataset in ease
    """
    def __init__(self, path):
        assert path.endswith('.csv'), 'Path must be a .csv file'
        self.path = path
        self.df = pd.read_csv(path)
        # self._preprocess()

    def __str__(self) -> str:
        return f'SpotifyCharts(path={self.path})'
    
    def __repr__(self) -> str:
        # Print the columns and some easily readable information
        return f'SpotifyCharts(path={self.path}, columns={self.df.columns}, shape={self.df.shape})'
    
    def _preprocess(self):
        """
        Preprocess the data. This function will be called in the constructor.
        """
        raise NotImplementedError

    def get_by_region(self, region_names : Union[str, List[str]]):
        return self.df[self.df['region'].isin(region_names)]

    def get_by_date(self, date_range : Union[str, List[str]]):
        """
        Return the piece of data in the given date range.
        """
        if isinstance(date_range, str):
            # If the date range is a string, then it is a single date
            return self.df[self.df['date'] == date_range]
        elif isinstance(date_range, list):
            # If the date range is a list, then it is a range of dates
            return self.df[(self.df['date'] >= date_range[0]) & (self.df['date'] <= date_range[1])]
        else:
            raise TypeError('date_range must be a string or a list of strings')
        

    def query(self, query : str):
        return self.df.query(query)
    
    def pprint(self, column_name):
        """
        Beatifully print informations & statitsics about the given column.
        """
        describe = self.df[column_name].describe()
        value_counts = self.df[column_name].value_counts()
        nunique = self.df[column_name].nunique()
        unique = self.df[column_name].unique()

        # Print beatifully in a table-wise format
        print(f"{'-'*50}")
        print(f'Column name: {column_name}')
        print(f'Number of unique values: {nunique}')
        print(f'Unique values: {unique}')
        print(f'Value counts: {value_counts}')
        print(f'Description: {describe}')
        print(f"{'-'*50}")