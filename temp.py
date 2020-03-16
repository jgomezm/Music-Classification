import pandas as pd
import pickle
import datetime

countriesOfInterest = ["HK", "JP", 'ZA', 'TN', 'TR', 'GB', 'MX', 'US', 'CO',
                       'EC', 'AU', 'NZ']

stats = pd.DataFrame()
comb_data= pd.DataFrame()
for country in countriesOfInterest:
    data = pickle.load(open( "Raw Track Data\\" + country + "_train.p", "rb" ))
    data = data.append(
        pickle.load(open( "Raw Track Data\\" + country + "_val.p", "rb" )))
    data = data.append(
        pickle.load(open( "Raw Track Data\\" + country + "_test.p", "rb" )))
    stats.loc[country,"n_tracks"] = data.track_id.nunique()
    stats.loc[country,"n_playlists"] = data.Playlist.nunique()
    stats.loc[country,"mean_n_segments"] = data.groupby("track_id").size().mean()
    duration = data.groupby("track_id").duration.sum()
    stats.loc[country,"mean_duration"] = str(datetime.timedelta(seconds=duration.mean()))
    comb_data = comb_data.append(data)

stats.loc["aggregate","n_tracks"] = comb_data.track_id.nunique()
stats.loc["aggregate","n_playlists"] = comb_data.Playlist.nunique()
stats.loc["aggregate","mean_n_segments"] = comb_data.groupby("track_id").size().mean()
duration = comb_data.groupby("track_id").duration.sum()
stats.loc["aggregate","mean_duration"] = str(datetime.timedelta(seconds=duration.mean()))
