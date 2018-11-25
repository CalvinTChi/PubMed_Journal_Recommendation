from keras.layers import Input, Dense, Flatten, Embedding, Conv1D, MaxPooling1D, Activation
from tensorflow.contrib.keras.api.keras.initializers import Constant
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.models import Model
from keras.utils import to_categorical
import keras.optimizers
import numpy as np
import pandas as pd
import pickle
import gensim
import math
import sys
import os

EMBEDDING_DIM = 200
BATCH_SIZE = 512
MAX_SEQ_LENGTH = 500
label_mapping = {'bioinformatics': 0, 'development': 1, 'epigenetics': 2, 'mendelian': 3, 
                'omics': 4, 'population_genetics': 5, 'statistical_genetics': 6, 'structure': 7}
embedding_matrix = pickle.load(open("data/embedding.p", "rb"))
tokenizer = pickle.load(open("data/tokenizer.p", "rb"))
labelEncoder = pickle.load(open("data/label_encoder.p", "rb"))
trainIterator = pd.read_table("data/train_j.txt", delimiter="\t", header = 0, chunksize=BATCH_SIZE)
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
# OUTPUT: (1) pandas df of rows x word2vec feature, (2) prediction targets
def generate_feature_label_pair(mat):
    X = tokenizer.texts_to_sequences(mat.iloc[:, 0])
    X = pad_sequences(X, maxlen = MAX_SEQ_LENGTH, padding='post')
    Yc = mat.iloc[:, 2].tolist()
    Yc = [label_mapping[label] for label in Yc]
    Yc = to_categorical(Yc, num_classes = len(label_mapping))
    Yj = mat.iloc[:, 3].tolist()
    Yj = labelEncoder.transform(Yj)
    Yj = to_categorical(Yj, num_classes = len(labelEncoder.classes_))
    Yi = mat.iloc[:, 4].as_matrix()
    Yi = np.array([if2quartile(ifactor) for ifactor in Yi])
    Yi = to_categorical(Yi, num_classes = len(quartiles) + 1)
    return X, {"category": Yc, "journal": Yj, "if": Yi}

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
    sequence_input = Input(shape=(MAX_SEQ_LENGTH,), dtype='int32')
    word_index = tokenizer.word_index
    embedded_layer = Embedding(len(word_index) + 1,
                                   EMBEDDING_DIM,
                                   embeddings_initializer = Constant(embedding_matrix),
                                   input_length = MAX_SEQ_LENGTH,
                                   trainable = False)
    embedded_sequences = embedded_layer(sequence_input)
    x = Conv1D(128, 5, activation='relu', input_shape = (200, 1))(embedded_sequences)
    x = MaxPooling1D(5)(x)
    x = Conv1D(128, 5, activation='relu')(x)
    x = MaxPooling1D(35)(x)
    x = Flatten()(x)
    x = Dense(128, activation = 'relu')(x)
    category_output = Dense(len(label_mapping), activation = 'softmax', name = "category")(x)
    journal_output = Dense(len(labelEncoder.classes_), activation = 'softmax', name = "journal")(x)
    if_output = Dense(len(quartiles) + 1, activation = 'softmax', name = "if")(x)
    model = Model(inputs = sequence_input, outputs = [category_output, journal_output, if_output])
    model.compile(loss = {'category': 'categorical_crossentropy', 'journal': 'categorical_crossentropy', 
                          'if': 'categorical_crossentropy'},
                 optimizer = keras.optimizers.Adam(lr=0.001), 
                 metrics = {'category': 'accuracy', 'journal': 'accuracy', 'if': 'accuracy'})
    
    return model

def main():
    #Get number of training samples
    with open("data/train_j.txt") as f:
        nTrain = sum(1 for _ in f)
    dev = pd.read_table("data/dev_j.txt", delimiter="\t", header = 0)
    devX, devY = generate_feature_label_pair(dev)
    nBatches = math.ceil(nTrain / BATCH_SIZE)
    model = create_model()
    model.fit_generator(sample_generator(), steps_per_epoch = nBatches, epochs=2, 
        validation_data=(devX, devY))
    model.save("model/multitask2.h5")

if __name__ == "__main__":
    main()


