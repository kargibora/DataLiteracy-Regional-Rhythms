import matplotlib.pyplot as plt
import seaborn as sns
from tueplots import bundles
import pandas as pd
import numpy as np
import os
import tqdm

plt.rcParams.update(bundles.beamer_moml())

# Update the font
plt.rcParams["font.family"] = "serif"

# Update the plt savefig path
SAVE_DIR = '../../figures/'
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.format'] = 'jpg'
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




def save_current_plot(filename):
    """
    Save the current plot to the specified filename.

    Parameters:
        filename (str): The filename to save the plot to.
    """
    plt.savefig(os.path.join(SAVE_DIR,filename), bbox_inches='tight')

