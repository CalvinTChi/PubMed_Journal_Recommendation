from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv1D, MaxPooling1D, Activation
from keras.utils import to_categorical
import keras.optimizers
import numpy as np
import pandas as pd
import os
import gensim
import math

label_mapping = {'bioinformatics': 0, 'development': 1, 'epigenetics': 2, 'mendelian': 3, 
                'omics': 4, 'population_genetics': 5, 'statistical_genetics': 6, 'structure': 7}
batch_size = 512
global trainIterator
trainIterator = pd.read_table("data/train.txt", delimiter="\t", header = 0, chunksize=BATCH_SIZE)


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

def train_dev_test_split():
    abstracts = pd.read_table("data/abstracts.txt", delimiter="\t", header = 0)
    abstracts = abstracts.sample(frac=1)
    n = abstracts.shape[0]
    tenP = round(n * 0.1)
    test = abstracts.iloc[0:tenP, :]
    dev = abstracts.iloc[tenP:(2 * tenP), :]
    train = abstracts.iloc[(2 * tenP):n, :]
    test.to_csv("data/test.txt", index = False, sep = '\t')
    dev.to_csv("data/dev.txt", index = False, sep = '\t')
    train.to_csv("data/train.txt", index = False, sep = '\t')

def sample_generator():
    while True:
        train = pd.read_table("data/train.txt", delimiter="\t", header = 0, chunksize=batch_size)
        for chunk in train:
            X, Y = generate_feature_label_pair(chunk)
            yield X, Y

def create_model():
    model = Sequential()
    model.add(Conv1D(128, 5, activation='relu', input_shape = (200, 1)))
    model.add(MaxPooling1D(5))
    model.add(Conv1D(128, 5, activation='tanh'))
    model.add(MaxPooling1D(35))
    model.add(Flatten())
    model.add(Dense(128, activation = 'tanh'))
    model.add(Dense(8, activation = 'softmax'))
    model.compile(loss = 'categorical_crossentropy',
                 optimizer = keras.optimizers.Adam(lr=0.0001, clipnorm = 1), metrics = ['accuracy'])
    return model

def main():
    if not os.path.isfile("data/train.txt") or not os.path.isfile("data/dev.txt") or not os.path.isfile("data/test.txt"):
        train_dev_test_split()
    nTrain = 332305 
    dev = pd.read_table("data/dev.txt", delimiter="\t", header = 0)
    devX, devY = generate_feature_label_pair(dev)

    nBatches = math.ceil(nTrain / batch_size)
    model = create_model()
    model.fit_generator(sample_generator(), steps_per_epoch = nBatches, epochs=2, validation_data=(devX, devY))

    model.save("model/category1")

if __name__ == "__main__":
    main()



