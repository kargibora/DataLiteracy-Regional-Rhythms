import pandas as pd

def datetime_start_end_generator(start_date, end_date, delta=7):
    """
    Generate datetime.date objects between start_date and end_date specified with delta days using Pandas.

    Parameters:
        start_date (str): The start date in 'YYYY-MM-DD' format.
        end_date (str): The end date in 'YYYY-MM-DD' format.
        delta (int): Number of days between each date in the range. Default is 7 days.

    Yields:
        Tuple[pd.Timestamp, pd.Timestamp]: Tuples of start and end dates.
    """
    # Convert start_date and end_date to pandas datetime
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    # Generate date ranges
    for date in pd.date_range(start=start_date, end=end_date, freq=f'{delta}D'):
        yield_date = min(date + pd.Timedelta(days=delta - 1), end_date)
        yield (date, yield_date)
