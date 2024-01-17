# Plotting libraries
import matplotlib.pyplot as plt
import seaborn as sns
from tueplots import bundles
from tueplots.constants.color import rgb

import pandas as pd
import numpy as np
import os
import tqdm
from scipy.stats import kendalltau, spearmanr, pearsonr
from typing import List, Tuple

from .regions import get_charts_by_region, get_regional_charts_delta_rank,calculate_popularity_metrics
from .tracks import get_track_title

SAVE_DIR = '../../figures/'


def setup_plotting_icml2022(**bundles_kwargs):
    """
    Set up the plotting environment.
    """
    
    plt.rcParams.update(bundles.icml2022(
        **bundles_kwargs,
    ))

    # Update the plt savefig path
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['savefig.format'] = '.pdf'
    plt.rcParams["savefig.directory"] = SAVE_DIR


def plot_multiple_tracks_time_series(dfs, x_column, y_column, labels, title='Time Series Plot', xlabel='Date', ylabel='Value', ax=None, marker_every=5):
    """
    Plot multiple time series graphs on the same axes, based on the specified DataFrames and columns.

    Parameters:
        dfs (list of pd.DataFrame): A list of DataFrames containing the data.
        x_column (str): The column name for the x-axis (date).
        y_column (str): The column name for the y-axis (value).
        labels (list of str): The labels for each DataFrame's line in the legend.
        title (str): The title of the plot.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
        ax (matplotlib.axes._subplots.AxesSubplot, optional): Matplotlib Axes object to plot on.
        marker_every (int): Interval for showing markers.
    """

    if len(dfs) != len(labels):
        raise ValueError("The number of DataFrames and labels must be the same")

    # Set plot style
    sns.set(style="whitegrid")

    # Create the plot
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))


    for df, label in zip(dfs, labels):
        # Ensure correct types
        df[x_column] = pd.to_datetime(df[x_column])
        sns.lineplot(x=x_column, y=y_column, data=df, label=label, ax=ax, marker='o', markevery=marker_every)

    # Set titles and labels
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # Show plot if ax is not provided
    if ax is None:
        plt.show()


def plot_correlation_heatmap(df, features, title="Correlation Heatmap", half=False):
    """
    Plot a heatmap for the correlation matrix of specified features in the DataFrame.

    Parameters:
        df (pd.DataFrame): The DataFrame containing the features and correlation values.
        features (list of str): The list of features to include in the heatmap.
        title (str): The title of the heatmap.
        half (bool): If True, render only the lower half of the heatmap.
    """
    # Calculate the correlation matrix
    corr_matrix = df[features].corr()

    # Create a mask to display only the lower half of the heatmap if half is True
    mask = None
    if half:
        mask = np.zeros_like(corr_matrix, dtype=bool)
        mask[np.triu_indices_from(mask)] = True

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(230, 20, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr_matrix, mask=mask, cmap=cmap, vmax=.3, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True)

    # Set the title of the heatmap
    plt.title(title)

    # Show the plot
    plt.show()

def plot_popularity_rank_correlation(charts_df : pd.DataFrame,
                                     TEST_REGION : str,
                                     k_range : List[int],
                                     date_range : Tuple[str, str] = ('2017-01-01', '2017-12-31')):
    kendel_coeffs = {}
    spearman_coeffs = {}
    corr_coeffs = {}

    for k in k_range:
        regional_df = get_charts_by_region(charts_df, TEST_REGION)
        test_df_yearly = get_regional_charts_delta_rank(
            regional_df,
            date = date_range,
            normalize_streams=False)

        test_df_popularity = calculate_popularity_metrics(
            regional_df,
            date_range,
            delta_k = k,
        )

        test_df_unique = test_df_popularity.drop_duplicates(subset="track_id", inplace=False)
        test_df_unique['weighted_popularity'] = test_df_unique['popularity'] * test_df_unique['average_stream_proportion']

        # sort by popularity
        test_df_unique.sort_values('weighted_popularity', ascending=False, inplace=True)
        test_df_unique.reset_index(inplace=True)

        # Get the top 5 tracks
        test_df_unique.head(5)

        # Compare the two lists and their rankings
        x = []  
        y = []
        for track_id in test_df_unique["track_id"].unique():
            x.append(test_df_unique[test_df_unique["track_id"] == track_id]["weighted_popularity"].values[0])
            y.append(test_df_yearly[test_df_yearly["track_id"] == track_id]["delta_rank"].values[0])

        kendel_coeffs[k] = kendalltau(x, y)
        spearman_coeffs[k] = spearmanr(x, y)
        corr_coeffs[k] = pearsonr(x, y)

    
    # Plot the coeffs
    plt.figure(figsize=(15, 10))  # Increase the size of the figure
    plt.style.use('ggplot')  # Use a different style for a more appealing look

    # Use markers for better visibility of data points
    plt.plot(kendel_coeffs.keys(), [kendel_coeffs[k][0] for k in kendel_coeffs.keys()], marker='o', label="Kendall")
    plt.plot(spearman_coeffs.keys(), [spearman_coeffs[k][0] for k in spearman_coeffs.keys()], marker='o', label="Spearman")
    plt.plot(corr_coeffs.keys(), [corr_coeffs[k][0] for k in corr_coeffs.keys()], marker='o', label="Correlation")

    # Also plot the p-values on top of the data points for better visibility
    for k in kendel_coeffs.keys():
        plt.text(k, kendel_coeffs[k][0], "{:.2e}".format(kendel_coeffs[k][1]), fontsize=10)
    for k in spearman_coeffs.keys():
        plt.text(k, spearman_coeffs[k][0], "{:.2e}".format(spearman_coeffs[k][1]), fontsize=10)
    for k in corr_coeffs.keys():
        plt.text(k, corr_coeffs[k][0], "{:.2e}".format(corr_coeffs[k][1]), fontsize=10)
    
    plt.title(f'Correlation of rank with weighted popularity for {TEST_REGION} between (2017-01-01, 2017-12-31)', fontsize=16)  # Increase the font size of the title
    plt.xlabel('$ \Delta k$', fontsize=14)  # Increase the font size of the x-label
    plt.ylabel('Correlation', fontsize=14)  # Increase the font size of the y-label

    plt.grid(True, which='both', color='gray', linewidth=0.5)  # Make the grid more subtle
    plt.tight_layout()
    plt.legend(fontsize=12)  # Increase the font size of the legend

def plot_popular_tracks_time_series(charts_df: pd.DataFrame, region: str, date_range: str, delta_k: int) -> None:
    """
    Plots the time series of popular tracks based on the given parameters.

    Args:
        charts_df (pd.DataFrame): The DataFrame containing the charts data.
        region (str): The region for which the analysis is performed.
        date_range (str): The date range for the analysis.
        delta_k (int): The delta value for calculating popularity metrics.

    Returns:
        None
    """
    test_df = calculate_popularity_metrics(
        get_charts_by_region(charts_df, region),
        date_range,
        delta_k=delta_k,
    )

    test_df['weighted_popularity'] = test_df['popularity'] * test_df['average_stream_proportion']
    popularities_sorted = test_df.sort_values('weighted_popularity', ascending=False).set_index('track_id')['weighted_popularity'].to_dict()
    popular_track_ids = np.random.choice(list(popularities_sorted.keys())[:5], 5, replace=False)
    popular_tracks_df = []
    popular_track_labels = []
    for track_id in popular_track_ids:
        track_df = test_df[test_df['track_id'] == track_id]
        popular_tracks_df.append(track_df)
        popular_track_labels.append(f"{get_track_title(track_df, track_id)[:20]}")

    y_columns = ['weighted_popularity', 'popularity', 'stream_proportion', 'rank']

    # Create a 2x2 grid of subplots
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    axs = axs.flatten()  # Flatten the 2D array of axes for easy iteration

    # Loop through each column and create a plot
    for i, y_col in enumerate(y_columns):
        plot_multiple_tracks_time_series(
            popular_tracks_df,
            "date",
            y_col,
            title=f'Time Series Plot for {y_col}',
            labels=popular_track_labels,
            xlabel='Date',
            ylabel=y_col,
            ax=axs[i],
            marker_every=5  # Show a marker every 5 points
        )

    # save_current_plot(f"time_series_plot_{region}_delta_{delta_k}.png")

    # Make the legends smaller
    for ax in axs:
        ax.legend(fontsize='small')
    plt.tight_layout()
    plt.show()

def save_current_plot(filename):
    """
    Save the current plot to the specified filename.

    Parameters:
        filename (str): The filename to save the plot to.
    """
    plt.savefig(os.path.join(SAVE_DIR,filename), bbox_inches='tight')

