from keras.layers import Dense, Flatten, Embedding, Conv1D, MaxPooling1D, Activation
from tensorflow.contrib.keras.api.keras.initializers import Constant
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
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

JOURNAL_MIN_NUM = 40
EMBEDDING_DIM = 200
BATCH_SIZE = 512
MAX_SEQ_LENGTH = 500
embedding_matrix = pickle.load(open("data/embedding.p", "rb"))
tokenizer = pickle.load(open("data/tokenizer.p", "rb"))
labelEncoder = pickle.load(open("data/label_encoder.p", "rb"))
trainIterator = pd.read_table("data/train_j.txt", delimiter="\t", header = 0, chunksize=BATCH_SIZE)
trainIterator = iter(trainIterator)

# INPUT: chunk = data frame of subset of data, p = percentage dataset to allocate to development and test dataset
# OUTPUT: dataframes of train, development, and test dataset
def train_dev_test_split(chunk, p = [0.1, 0.1]):
    n = chunk.shape[0]
    devP = round(n * p[0])
    testP = round(n * p[1])
    dev = chunk.iloc[0:devP, :]
    test = chunk.iloc[devP:(devP + testP), :]
    train = chunk.iloc[(devP + testP):n, :]
    return train, dev, test

# INPUT: pandas df of rows x features1, where features = [abstract, PMID, category, journalAbbrev, impact_factor]
# OUTPUT: (1) pandas df of rows x word2vec feature, (2) vector of labels corresponding to journals.
def generate_feature_label_pair(mat):
    X = tokenizer.texts_to_sequences(mat.iloc[:, 0])
    X = pad_sequences(X, maxlen = MAX_SEQ_LENGTH, padding='post')
    Y = mat.iloc[:, 3].tolist()
    Y = labelEncoder.transform(Y)
    Y = to_categorical(Y, num_classes = len(labelEncoder.classes_))
    return X, Y

def sample_generator():
    global trainIterator
    while True:
        try:
            chunk = next(trainIterator)
        except:
            trainIterator = pd.read_table("data/train_j.txt", delimiter="\t", header = 0, chunksize=BATCH_SIZE)
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
    model.add(Conv1D(128, 5, activation='relu'))
    model.add(MaxPooling1D(35))
    model.add(Flatten())
    model.add(Dense(128, activation = 'relu'))
    model.add(Dense(len(labelEncoder.classes_), activation = 'softmax'))
    model.compile(loss = 'categorical_crossentropy',
                 optimizer = keras.optimizers.Adam(lr=0.001), 
                 metrics = ['accuracy'])
    return model

def main():
    # Get number of training samples
    with open("data/train_j.txt") as f:
        nTrain = sum(1 for _ in f)
    dev = pd.read_table("data/dev_j.txt", delimiter="\t", header = 0)
    devX, devY = generate_feature_label_pair(dev)
    nBatches = math.ceil(nTrain / BATCH_SIZE)
    trainIterator2 = pd.read_table("data/train_j.txt", delimiter="\t", header = 0)
    model = create_model()
    model.fit_generator(sample_generator(), steps_per_epoch = nBatches, epochs=2, validation_data=(devX, devY))
    model.save("model/journal_embedding_model.h5")

if __name__ == "__main__":
    main()


