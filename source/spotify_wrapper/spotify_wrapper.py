# Design a spotify wrapper for the Spotify API
import spotipy
from typing import Optional, List

class SpotifyWrapper(object):
    def __init__(self, client_id, client_secret, app_name, redirect_uri, scope):
        self.client_id = client_id
        self.client_secret = client_secret
        self.app_name = app_name
        self.redirect_uri = redirect_uri
        self.scope = scope

        self.sp = spotipy.Spotify(auth_manager=spotipy.SpotifyOAuth(client_id=self.client_id,
                                                                    client_secret=self.client_secret,
                                                                    redirect_uri=self.redirect_uri,
                                                                    scope=self.scope
                                                                    ))
        
    def get_track(self, track_id : str):
        """
        Get track by id. 

        Parameters:
        - track_id (str): The id of the track.

        Returns:
        - A dictionary containing the track data.
        """
        return self.sp.track(track_id)

    def get_tracks(self, track_ids : List[str]):
        """
        From a list of track ids, get the tracks.

        Parameters:
        - track_ids (List[str]): A list of track ids.   

        Returns:
        - A list of dictionaries containing the track data.
        """
        return [self.get_track(tid) for tid in track_ids]


    def get_playlist(self, playlist_id):
        pass

    def get_playlists(self, playlist_ids):
        pass

    def get_playlists_by_name(self, playlist_name):
        pass

    def get_artist(self, artist_id : Optional[str] = None, track_id : Optional[str] = None):
        """
        Get artist by using either`artist_id` or `track_id`.

        Parameters:
        - artist_id (Optional[str]): The id of the artist.
        - track_id (Optional[str]): The id of the track.

        Returns:
        - A dictionary containing the artist data.
        """
        if artist_id is None and track_id is None:
            raise ValueError("Either artist_id or track_id must be specified.")
        
        if artist_id is not None:
            return self.sp.artist(artist_id)
        
        if track_id is not None:
            track = self.get_track(track_id)
            return track['artists'][0]

    def get_audio_features(self, track_id):
        """
        Get audio features of a track. Since there is a limit of 100 tracks per request,
        split the tracks into batches of 100.

        Parameters:
        - track_id (str): The id of the track.

        Returns:
        - A dictionary containing the audio features.
        """
        if len(track_id) > 100:
            splitted = [track_id[i:i+100] for i in range(0, len(track_id), 100)]
            audio_features =  [self.sp.audio_features(tracks) for tracks in splitted]
            # Flatten the lists so that they are not nested
            audio_features = [item for sublist in audio_features for item in sublist]
        else:
            audio_features = self.sp.audio_features(track_id)

        return audio_features

    

