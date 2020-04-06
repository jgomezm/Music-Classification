import pandas as pd
import numpy as np
import pickle
from sklearn.utils import class_weight
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import tensorflow.keras as keras
from tensorflow.keras import Sequential
from tensorflow.keras import regularizers
from tensorflow.keras.layers import LSTM, Dense, LSTM, Flatten, BatchNormalization, Dropout
from tensorflow.keras.utils import to_categorical
import tensorflow as tf

#%%
def splitSeconds(n, country, t, seconds, samplerate):
    length = seconds * samplerate
    data = pickle.load( open( "Raw Track Data\\" + country + "_" + t + ".p", "rb" ))
    # indices are not unique
    data.reset_index(drop = True, inplace = True)
    tracks = data.track_id.unique()
    tracks = np.random.choice(tracks, size=n, replace=False)
    trackFeats = data[data.track_id.isin(tracks)]
    dur = trackFeats.iloc[:,1]
    long = trackFeats.loc[trackFeats.index.repeat(dur * samplerate)].reset_index(drop = True)
    long['change'] = long.track_id.eq(long.track_id.shift())
    change = pd.Series(long[long.change == False].index)
    long = long.iloc[:, 5:30]
    sizes = change.shift(-1) - change
    sizes.iloc[-1] = long.shape[0] - change.iloc[-1]
    sizes = sizes // length
    indices = []
    for i, size in enumerate(sizes):
        patch = np.random.randint(0, size) # upper bound is size - 1
        indices += list(range(change[i] + length*patch, change[i] + length*(patch+1)))
    long = long.loc[indices]
    long = np.stack(np.split(long, np.arange(1, n) * length))
    return long, np.repeat(np.array([country]), long.shape[0])

#%%

def getSamples(train_n, val_n, seconds, samplerate, countriesOfInterest,
               enc, verbose = 0):
    train_labels = pd.DataFrame()
    val_labels = pd.DataFrame()
    train_x = []
    train_labels = []
    val_x = []
    val_labels = []
    for country in countriesOfInterest:
        if verbose > 0:
            print("getting",country)
        x1, y1 = splitSeconds(train_n, country, "train", seconds, samplerate)
        x2, y2 = splitSeconds(val_n, country, "val", seconds, samplerate)
        train_x = train_x + x1.tolist()
        train_labels = train_labels + y1.tolist()
        val_x = val_x + x2.tolist()
        val_labels = val_labels + y2.tolist()
    #train_x = np.array(train_x)
    y = np.dstack(train_x)
    train_x = np.rollaxis(y,-1)
    train_labels = np.array(train_labels)
    #val_x = np.array(val_x)
    y = np.dstack(val_x)
    val_x = np.rollaxis(y,-1)
    val_labels = np.array(val_labels)
    class_weights = class_weight.compute_class_weight('balanced',
                                                     np.unique(train_labels),
                                                     list(train_labels))
    train_labels = enc.transform(np.array(train_labels).reshape(-1,1)).toarray()
    val_labels = enc.transform(np.array(val_labels).reshape(-1,1)).toarray()
    return train_x, train_labels, val_x, val_labels, class_weights


def train(iterations, learn_rate, train_n, val_n, seconds, samplerate,
          countriesOfInterest, enc, epochs, tensorboard_callback, model_dir,
          model):
    for i in range(iterations):
        adam = keras.optimizers.Adam(lr=learn_rate)
        model.compile(loss = "categorical_crossentropy", optimizer= adam, metrics=["acc"])
        train_x, train_labels, val_x, val_labels, class_weights = getSamples(train_n, val_n, seconds, samplerate, countriesOfInterest, enc)
        print(np.sum(train_labels, axis = 0))
        model.fit(train_x, train_labels,
                  epochs = i * epochs + epochs, 
                  initial_epoch = i * epochs,
                  shuffle = True,
                  validation_data = (val_x, val_labels),
                  batch_size = 1024,
                  class_weight = class_weights,
                 callbacks=[tensorboard_callback],
                 verbose = 1)
        model.save_weights(model_dir)
        if i%2 == 0:
            learn_rate = learn_rate/2
        if i % 1 == 0:
            preds = model.predict(val_x, batch_size = 1024, verbose = 1)
         #   print(np.sum(train_labels, axis = 0))
            plt.imshow(
                confusion_matrix(
                    enc.inverse_transform(preds), 
                    enc.inverse_transform(val_labels), 
                   # normalize = "all"
                )
            )
            plt.pause(.5)
            plt.show()
            preds = model.predict(train_x, batch_size = 1024, verbose = 1)
            plt.imshow(
                confusion_matrix(
                    enc.inverse_transform(preds), 
                    enc.inverse_transform(train_labels), 
                #    normalize = "all"
                )
            )
            plt.pause(.5)
            plt.show()