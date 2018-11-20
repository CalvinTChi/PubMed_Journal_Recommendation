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

JOURNAL_MIN_NUM = 40
EMBEDDING_DIM = 200
BATCH_SIZE = 512
MAX_SEQ_LENGTH = 500
embedding_matrix = pickle.load(open("data/embedding.p", "rb"))
tokenizer = pickle.load(open("data/tokenizer.p", "rb"))
global trainIterator
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
    Y = [label_mapping[label] for label in Y]
    Y = to_categorical(Y)
    return X, Y

def sample_generator():
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
    model.add(Dense(8, activation = 'softmax'))
    model.compile(loss = 'categorical_crossentropy',
                 optimizer = keras.optimizers.Adam(lr=0.001), 
                 metrics = ['accuracy'])
    return model

# p = percentage dataset to allocate to development and test dataset. rest is training
# Splits entire dataset into train, dev, and test
def journal_prediction_split_dataset(p = [0.1, 0.1]):
    metadata = pd.read_table("data/metadata.txt", delimiter = "\t", header = 0)
    journalAbbrev = metadata['journalAbbrev'].tolist()
    count = {x: journalAbbrev.count(x) for x in set(journalAbbrev)}
    journalSet = set([j for j in journalAbbrev  if count[j] >= JOURNAL_MIN_NUM])
    abstracts = pd.read_table("data/abstracts.txt", delimiter="\t", header = 0, chunksize = 50000)
    train = pd.DataFrame(columns=['abstract', 'PMID', 'category', 'journalAbbrev', 'impact_factor'])
    dev = pd.DataFrame(columns=['abstract', 'PMID', 'category', 'journalAbbrev', 'impact_factor'])
    test = pd.DataFrame(columns=['abstract', 'PMID', 'category', 'journalAbbrev', 'impact_factor'])
    train.to_csv("data/train_j.txt", index = False, sep = '\t')
    dev.to_csv("data/dev_j.txt", index = False, sep = '\t')
    test.to_csv("data/test_j.txt", index = False, sep = '\t')
    for chunk in abstracts:
        chunk = chunk.iloc[np.where(chunk['journalAbbrev'].isin(journalSet))]
        train, dev, test = train_dev_test_split(chunk, p)
        with open("data/train_j.txt", "a") as f:
            train.to_csv(f, header = False, index = False, sep = '\t')
        with open("data/dev_j.txt", "a") as f:
            dev.to_csv(f, header = False, index = False, sep = '\t')
        with open("data/test_j.txt", "a") as f:
            test.to_csv(f, header = False, index = False, sep = '\t')

def main():
    if not os.path.isfile("data/train_j.txt") or not os.path.isfile("data/dev_j.txt") or not os.path.isfile("data/test_j.txt"):
        journal_prediction_split_dataset()
    # Get number of training samples
    with open("data/train_j.txt") as f:
        nTrain = sum(1 for _ in f)
    dev = pd.read_table("data/dev_j.txt", delimiter="\t", header = 0)
    devX, devY = generate_feature_label_pair(dev)
    nBatches = math.ceil(nTrain / BATCH_SIZE)
    print(devY[:10])
    sys.exit(2)
    model = create_model()
    model.fit_generator(sample_generator(), steps_per_epoch = nBatches, epochs=2, validation_data=(devX, devY))
    model.save("model/journal_baseline.h5")

if __name__ == "__main__":
    main()








