from keras.layers import Dense, Flatten, Embedding, Conv1D, MaxPooling1D, Activation
from tensorflow.contrib.keras.api.keras.initializers import Constant
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.utils import to_categorical
import keras.optimizers
import numpy as np
import pandas as pd
import pickle
import gensim
import math
import sys
import os

BATCH_SIZE = 512
EMBEDDING_DIM = 200
MAX_SEQ_LENGTH = 500
global trainIterator
trainIterator = pd.read_table("data/train.txt", delimiter="\t", header = 0, chunksize=BATCH_SIZE)
trainIterator = iter(trainIterator)
tokenizer = pickle.load(open("data/tokenizer.p", "rb"))
embedding_matrix = pickle.load(open("data/embedding.p", "rb"))

# INPUT: pandas df of rows x features1, where features = [abstract, PMID, category, journalAbbrev, impact_factor]
# OUTPUT: (1) pandas df of rows x word2vec feature, (2) vector of labels in [0, 7].
def generate_feature_label_pair(mat):
    X = tokenizer.texts_to_sequences(mat.iloc[:, 0])
    X = pad_sequences(X, maxlen = MAX_SEQ_LENGTH, padding='post')
    Y = mat.iloc[:, 4].tolist()
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
    word_index = tokenizer.word_index
    embedding_layer = Embedding(len(word_index) + 1,
                                EMBEDDING_DIM,
                                embeddings_initializer = Constant(embedding_matrix),
                                input_length = MAX_SEQ_LENGTH,
                                trainable = False)
    model = Sequential()
    model.add(embedding_layer)
    model.add(Conv1D(128, 5, activation='relu', input_shape = (200, 1)))
    model.add(MaxPooling1D(5))
    model.add(Conv1D(128, 5, activation='tanh'))
    model.add(MaxPooling1D(35))
    model.add(Flatten())
    model.add(Dense(128, activation = 'tanh'))
    model.add(Dense(1, kernel_initializer='normal'))
    # mae = mean absolute error
    model.compile(loss = 'mean_squared_error',
                 optimizer = keras.optimizers.Adam(lr=0.001, clipnorm = 1), 
                 metrics = ['mae'])
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



