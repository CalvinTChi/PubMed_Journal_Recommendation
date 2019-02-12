from keras.layers import Dense, Flatten, Embedding, Conv1D, MaxPooling1D, Activation
from tensorflow.contrib.keras.api.keras.initializers import Constant
from keras.preprocessing.sequence import pad_sequences
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras.utils import to_categorical
from utils import *
import keras.optimizers
import numpy as np
import pandas as pd
import pickle, gensim, math, sys, os

global trainIterator
trainIterator = pd.read_table("data/train.txt", delimiter="\t", header = 0, chunksize=BATCH_SIZE)
trainIterator = iter(trainIterator)
quartiles = {0: 2.5, 1: 5, 2: 10, 3: 15}

def if2quartile(ifactor):
    if ifactor <= quartiles[0]:
        return 0
    elif ifactor > quartiles[0] and ifactor <= quartiles[1]:
        return 1
    elif ifactor > quartiles[1] and ifactor <= quartiles[2]:
        return 2
    elif ifactor > quartiles[2] and ifactor <= quartiles[3]:
        return 3
    elif ifactor > quartiles[3]:
        return 4

# INPUT: pandas df of rows x features1, where features = [abstract, PMID, category, journalAbbrev, impact_factor]
# OUTPUT: (1) pandas df of rows x word2vec feature, (2) vector of labels in [0, 7].
def generate_feature_label_pair(mat):
    X = tokenizer.texts_to_sequences(mat.iloc[:, 0])
    X = pad_sequences(X, maxlen = MAX_SEQ_LENGTH, padding='post')
    Y = mat.iloc[:, 4].as_matrix()
    Y = np.array([if2quartile(ifactor) for ifactor in Y])
    Y = to_categorical(Y, num_classes = len(quartiles) + 1)
    return X, Y

def sample_generator():
    while True:
        try:
            chunk = next(trainIterator)
        except:
            trainIterator = pd.read_table("data/train.txt", delimiter="\t", header = 0, chunksize=BATCH_SIZE)
            trainIterator = iter(trainIterator)
            chunk = next(trainIterator)
        X, Y = generate_feature_label_pair(chunk)
        yield X, Y

def create_model():
    embedding_layer = Embedding(MAX_NB_WORDS + 1,
                                EMBEDDING_DIM,
                                embeddings_initializer = Constant(embedding_matrix),
                                input_length = MAX_SEQ_LENGTH,
                                trainable = False)
    model = Sequential()
    model.add(embedding_layer)
    # convolution 1st layer
    model.add(Conv1D(128, 5, activation='relu', input_shape = (200, 1)))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(5))

    # convolution 2nd layer
    model.add(Conv1D(128, 5, activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(35))
    
    model.add(Flatten())
    model.add(Dense(128, activation = 'relu'))
    model.add(Dense(len(quartiles) + 1, activation = 'softmax'))
    model.compile(loss = 'categorical_crossentropy',
                 optimizer = keras.optimizers.Adam(lr=0.001), 
                 metrics = ['accuracy'])
    return model

def main():
    # Get number of training samples
    with open("data/train.txt") as f:
        nTrain = sum(1 for _ in f)
    dev = pd.read_table("data/dev.txt", delimiter="\t", header = 0)
    devX, devY = generate_feature_label_pair(dev)
    nBatches = math.ceil(nTrain / BATCH_SIZE)
    model = create_model()
    model.fit_generator(sample_generator(), steps_per_epoch = nBatches, epochs=2, validation_data=(devX, devY))
    model.save("model/impact_factor1.h5")

if __name__ == "__main__":
    main()



