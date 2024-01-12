"""
This utility function includes some functions to deal with
datetime.date objects.
"""

import pandas as pd

def datetime_start_end_generator(start_date, end_date, delta_t=7):
    """
    Generate datetime.date objects between start_date and end_date specified with delta_t days or months using Pandas.

    Parameters:
        start_date (str): The start date in 'YYYY-MM-DD' format.
        end_date (str): The end date in 'YYYY-MM-DD' format.
        delta_t (int or str): Number of days between each date in the range if an integer is provided, 
                              or 'M' for month-based ranges. Default is 7 days.

    Yields:
        Tuple[pd.Timestamp, pd.Timestamp]: Tuples of start and end dates.
    """
    # Convert start_date and end_date to pandas datetime
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    # Check if delta_t is a month indicator 'M'
    if delta_t == 30:
        current_date = start_date
        while current_date <= end_date:
            next_month_date = current_date + pd.DateOffset(months=1)
            yield_date = min(next_month_date, end_date)
            yield (current_date, yield_date)
            current_date = next_month_date
    elif delta_t == 365:
        current_date = start_date
        while current_date <= end_date:
            next_year_date = current_date + pd.DateOffset(years=1)
            yield_date = min(next_year_date, end_date)
            yield (current_date, yield_date)
            current_date = next_year_date
    else:
        # For delta as days
        for date in pd.date_range(start=start_date, end=end_date, freq=f'{delta_t}D'):
            yield_date = min(date + pd.Timedelta(days=delta_t - 1), end_date)
            yield (date, yield_date)
