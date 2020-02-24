import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd

client_credentials_manager = SpotifyClientCredentials(
        "26399552a8ce4d1285397254189cac50",
        "fdacbbba2dd34dbeb127dedb459f7ea3")
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

# get the track IDs in Germany Top 50
ger_pl = sp.playlist("spotify:playlist:37i9dQZEVXbJiZcmkrIHGU")
ger_t_list = []
for track in ger_pl["tracks"]["items"]:
    ger_t_list = ger_t_list + [track["track"]["uri"]]


audio = sp.audio_analysis(ger_t_list[0])
audio.keys()
# Beats are subdivisions of bars. Tatums are subdivisions of beats.
tatum_data = pd.DataFrame(audio["tatums"])
tatum_data = tatum_data.assign(end = tatum_data.start + tatum_data.duration)
(tatum_data.start - tatum_data.end.shift()).max()
# Sequence with only one series (tatum durations). Could be used to train a
# simple model.

# Audio segments attempts to subdivide a song into many segments, with each 
# segment containing a roughly consistent sound throughout its duration.
# A segment contains 30 features.
ger_data = pd.DataFrame()
for track in ger_t_list:
    audio = sp.audio_analysis(track)
    segments_data =  pd.DataFrame(audio["segments"])
    pitch = segments_data.pitches.apply(pd.Series)
    pitch.columns = ["p" + str(i) for i in range(1,13)]
    timbre = segments_data.timbre.apply(pd.Series)
    timbre.columns = ["t" + str(i) for i in range(1,13)]
    segments_data = segments_data.drop(["pitches", "timbre", "loudness_end"],
                                       axis = 1)
    segments_data = segments_data.join([pitch, timbre])
    segments_data["track_id"] = track
    ger_data = ger_data.append(segments_data)
ger_data.groupby("track_id").count()
