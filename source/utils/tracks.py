import pandas as pd
from typing import Union, Tuple, List, Optional, Dict
import tqdm

def get_track_title(tracks_df : pd.DataFrame, track_id : str) -> str:
    """
    Get the track name from the tracks dataframe.
    """
    return tracks_df[tracks_df['track_id'] == track_id]['title'].iloc[0]

def get_track_artist(tracks_df : pd.DataFrame, track_id : str) -> str:
    """
    Get the track artist from the tracks dataframe.
    """
    return tracks_df[tracks_df['track_id'] == track_id]['artist'].iloc[0]

# def interpolate_track_history(track_df : pd.DataFrame, date_range : Tuple[datetime.date, datetime.date], delta : int = 7):
#     """
#     For a track, interpolate the missing dates between the start and end date.
#     Also interpolate its rank and streams for missing dates.
#     """
#     raise NotImplementedError

# def plot_track_history_curve(region_df : pd.DataFrame, track_id):
#     """
#     Plot the history of a track in a region.
#     """
#     track_df = region_df[region_df['track_id'] == track_id]
#     track_df = track_df.sort_values(by=['date'])
#     plt.plot(track_df['date'], track_df['rank'])
#     plt.xlabel("Date")
#     plt.ylabel("Rank")
#     plt.title(f"Track history of {track_id}")
#     plt.show()

# def plot_features(region_df : pd.DataFrame,
#                   x_col : str,
#                   y_col : Union[str, List[str]],
#                   figname : str):
    
#     if isinstance(y_col, str):
#         y_col = [y_col]

#     # Make the fig sqrt(len(y_col)) x sqrt(len(y_col))
#     fig, ax = plt.subplots(nrows=int(np.sqrt(len(y_col))+1), ncols=int(np.sqrt(len(y_col))+1), figsize=(15,15))
#     for i, col in enumerate(y_col):
#         # get the axis double indices
#         r = int(i / np.sqrt(len(y_col)))
#         c = int(i % np.sqrt(len(y_col)))

#         # Get color based on r,c 
#         color = plt.cm.get_cmap('tab10')(i)
#         sns.scatterplot(data=region_df, x=x_col, y=col, ax=ax[r,c], color=color,alpha=0.5)
#         sns.regplot(data=region_df, x=x_col, y=col, ax=ax[r,c], color=color, scatter=False)
#         ax[r,c].set_xlabel(x_col)
#         ax[r,c].set_ylabel(col)
#     plt.savefig(os.path.join(SAVE_DIR, figname))

# # Belki şey yapılır bak:
# # 1. Jadamard mıdır nedir o distance hesapla (intersect)/(union)

# def calculate_regional_overall_popularity(region_df : pd.DataFrame, k : int):
#     """
#     For a track, calculate the popularity score based on the first and last appearance in the charts.
#     This function assumes that the region column i
#     """
#     # Assert 'region' is a unique value
#     assert len(region_df['region'].unique()) == 1, "The dataframe should only contain one region"

#     # Get the tracks first and last appearences and save it in a dataframe
#     overall_popularity, overall_weights = calculate_regional_popularity(region_df, k)

#     # Normalize weights by min max scaling
#     weights = np.array(list(overall_weights.values()))
#     weights = 1 - (weights - np.min(weights)) / (np.max(weights) - np.min(weights))
#     overall_weights = dict(zip(overall_weights.keys(), weights))

#     region_df['popularity'] = region_df['track_id'].apply(lambda x: overall_popularity[x])

#     # Find the track with highest last_date - first_date
#     region_df.drop_duplicates(subset=['track_id'], inplace=True)
#     region_audio_df = region_df.merge(audio_df, on='track_id', how='left')
    
#     # Remove the popularity = 0 tracks since they are noisy and bring NOTHING !!!!
#     region_audio_df = region_audio_df[region_audio_df['popularity'] > 0]

#     plot_features(region_audio_df,
#                   'popularity',
#                   ['danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence'],
#                   f'popularity_vs_features_{k}.pdf')

#     # Plot the heatmap of the correlaitons
#     corr = region_audio_df[['popularity', 'danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence']].corr()
#     sns.heatmap(corr, annot=True)
#     plt.savefig(os.path.join(SAVE_DIR, f'popularity_vs_features_corr_{k}_unweighted.pdf'))
    
#     # Plot the curve of the random track starting from
#     # track_id = df_track_group['track_id'].iloc[0]
    
# calculate_regional_overall_popularity(get_charts_by_region(charts_df, 'United States'),
#                                       k=200)
