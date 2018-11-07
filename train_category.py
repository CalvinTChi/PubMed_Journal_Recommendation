from keras.models import Sequential
from keras.layers import Dense, Flatten, Embedding, Conv1D, MaxPooling1D, Activation
from keras.utils import to_categorical
import keras.optimizers
import numpy as np
import pandas as pd
import pickle
import gensim
import math
import sys
import os

label_mapping = {'bioinformatics': 0, 'development': 1, 'epigenetics': 2, 'mendelian': 3, 
                'omics': 4, 'population_genetics': 5, 'statistical_genetics': 6, 'structure': 7}
BATCH_SIZE = 512
EMBEDDING_DIM = 200
MAX_SEQ_LENGTH = 500
VOCAB_SIZE = 
global trainIterator
trainIterator = pd.read_table("data/train.txt", delimiter="\t", header = 0, chunksize=BATCH_SIZE)
trainIterator = iter(trainIterator)

# INPUT: pandas df of rows x features1, where features = [abstract, PMID, category, journalAbbrev, impact_factor]
# OUTPUT: (1) pandas df of rows x word2vec feature, (2) vector of labels in [0, 7].
def generate_feature_label_pair(mat):
    n = mat.shape[0]
    X = np.zeros(shape = (n, 200))
    Y = np.zeros(shape = n)
    for i in range(n):
        abstract = mat.iloc[i, 0].split(" ")
        embedding = np.zeros(200)
        nWords = 0
        for word in abstract:
            if word in word2vec:
                vec = word2vec[word]
                vec = vec / np.linalg.norm(vec)
                embedding = np.add(embedding, vec)
                nWords += 1  
        X[i, :] = embedding / nWords
        Y[i] = label_mapping[mat.iloc[i, 2]]
    X = np.expand_dims(X, axis = 2)
    Y = to_categorical(Y)
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

def create_model(tokenizer, embedding_matrix):
    word_index = tokenizer.word_index
    embedding_layer = Embedding(len(word_index) + 1,
                                EMBEDDING_DIM,
                                weights = [embedding_matrix],
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
    model.add(Dense(8, activation = 'softmax'))
    model.compile(loss = 'categorical_crossentropy',
                 optimizer = keras.optimizers.Adam(lr=0.001, clipnorm = 1), 
                 metrics = ['accuracy'])
    print(model.summary())
    sys.exit(0)
    return model

def main():
    # Get number of training samples
    with open("data/train.txt") as f:
        nTrain = sum(1 for _ in f)
    dev = pd.read_table("data/dev.txt", delimiter="\t", header = 0)
    devX, devY = generate_feature_label_pair(dev)
    nBatches = math.ceil(nTrain / batch_size)
    tokenizer = pickle.load(open("data/tokenizer.p", "rb"))
    embedding_matrix = pickle.load(open("data/embedding.p", "rb"))
    model = create_model(tokenizer, embedding_matrix)
    model.fit_generator(sample_generator(), steps_per_epoch = nBatches, epochs=2, validation_data=(devX, devY))

    model.save("model/category1")

if __name__ == "__main__":
    main()



