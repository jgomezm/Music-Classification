import pandas as pd
import numpy as np
import pickle
from sklearn.utils import class_weight
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
#import tensorflow.keras as keras
import plaidml.keras
import os
plaidml.keras.install_backend()
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
import keras
from keras import Sequential
from keras import regularizers
from keras.layers import LSTM, Dense, Flatten, BatchNormalization, Dropout
from keras.utils import to_categorical
#import tensorflow as tf
import gc

#%%
def splitSeconds(n, country, t, seconds, samplerate):
    length = seconds * samplerate
    data = pickle.load( open( "Raw Track Data\\" + country + "_" + t + ".p", "rb" ) )
    tracks = data.track_id.unique()
    tracks = np.random.choice(tracks, size=n, replace=False)
    trackFeats = data[data.track_id.isin(tracks)]
    del data
    dur = trackFeats.iloc[:,1]
    long = trackFeats.loc[trackFeats.index.repeat(dur * samplerate)].reset_index(drop = True)
    del trackFeats
    long = long.sort_values(by = ["track_id", "start"])
    long['change'] = long.track_id.eq(long.track_id.shift())
    change = long[long.change == False].index
    long = long.iloc[:, 5:30]
    indices = np.concatenate((np.arange(0, long.shape[0], length), change))
    indices = np.sort(indices)
    indices = np.unique(indices)
    partition = np.split(np.array(long), indices)
    del long
    samples = []
    for i in partition:
        if i.shape[0] == length:
            samples = samples + [i]
    samples = np.stack(samples)
    gc.collect()
    return samples, np.repeat(np.array([country]), samples.shape[0])



#%%

def getSamples(train_n, val_n, seconds, samplerate, countriesOfInterest,
               enc, verbose = 0):
    train_labels = pd.DataFrame()
    val_labels = pd.DataFrame()
    train_x = None
    train_labels = []
    val_x = None
    val_labels = []
    for country in countriesOfInterest:
        if verbose > 0:
            print("getting",country)
        x1, y1 = splitSeconds(train_n, country, "train", seconds, samplerate)
        x2, y2 = splitSeconds(val_n, country, "val", seconds, samplerate)
        if train_x is None:
            train_x = x1
            train_labels = y1
            val_x = x2
            val_labels = y2
        else:
            train_x = np.append(train_x, x1, axis = 0)
            train_labels = np.append(train_labels, y1, axis = 0)
            del x1, y1
            val_x = np.append(val_x, x2, axis = 0)
            val_labels = np.append(val_labels, y2, axis = 0)
            del x2, y2
        gc.collect()
  #  train_x = np.array(train_x)
    gc.collect()
    #y = np.dstack(train_x)
    #del train_x
    #train_x = np.rollaxis(y,-1)
    #del y
   # train_labels = np.array(train_labels)
    #val_x = np.array(val_x)
    gc.collect()
    #y = np.dstack(val_x)
    #del val_x
    #val_x = np.rollaxis(y,-1)
    #del y
   # val_labels = np.array(val_labels)
    class_weights = class_weight.compute_class_weight('balanced',
                                                     countriesOfInterest,
                                                     list(train_labels))
    train_labels = enc.transform(np.array(train_labels).reshape(-1,1)).toarray()
    val_labels = enc.transform(np.array(val_labels).reshape(-1,1)).toarray()
    return train_x, train_labels, val_x, val_labels, class_weights


def train(iterations, learn_rate, train_n, val_n, seconds, samplerate,
          countriesOfInterest, enc, epochs, model_dir,
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
                  batch_size = 2048,
                  class_weight = class_weights,
         #        callbacks=[tensorboard_callback],
                 verbose = 1)
        #model.save_weights(model_dir)
        #if i%2 == 0:
        #    learn_rate = learn_rate/2
        if i % 1 == 0:
            preds = model.predict(val_x, batch_size = 2048, verbose = 1)
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
            preds = model.predict(train_x, batch_size = 2048, verbose = 1)
            plt.imshow(
                confusion_matrix(
                    enc.inverse_transform(preds), 
                    enc.inverse_transform(train_labels), 
                #    normalize = "all"
                )
            )
            plt.pause(.5)
            plt.show()