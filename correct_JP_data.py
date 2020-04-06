import pickle
import pandas as pd

data = pickle.load(open( "Raw Track Data\\" + "JP" + "_" + "train" + ".p", "rb" ))
data.reset_index(drop = True, inplace = True)
data['change'] = data.track_id.eq(data.track_id.shift())
change = pd.Series(data[data.change == False].index)

tracks = data.track_id[data.change == False]
dup = tracks[tracks.duplicated()]
data = data[~data.track_id.isin(dup)]
pickle.dump(data, open( "Raw Track Data\\" + "JP" + "_" + "train" + ".p", "wb" ))
