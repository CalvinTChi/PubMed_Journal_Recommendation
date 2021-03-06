from keras.layers import Dense, Flatten, Embedding, Conv1D, MaxPooling1D, Activation, Input, concatenate, Dropout
from tensorflow.contrib.keras.api.keras.initializers import Constant
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Model, load_model
import tensorflow as tf
from utils import *
from keras.backend import int_shape
from sklearn.metrics import accuracy_score, auc
from keras.layers.normalization import BatchNormalization
import keras.optimizers
import math, pickle, sys
import numpy as np
import pandas as pd

topic_model = load_model("model/category1.h5")
global category_graph
category_graph = tf.get_default_graph()
if_model = load_model("model/impact_factor1.h5")
global if_graph
if_graph = tf.get_default_graph()

trainIterator = pd.read_table("data/train_j.txt", delimiter="\t", header = 0, chunksize=BATCH_SIZE)
trainIterator = iter(trainIterator)

# INPUT: pandas df of rows x features1, where features = [abstract, PMID, category, journalAbbrev, impact_factor]
# OUTPUT: (1) pandas df of rows x word2vec feature, (2) vector of labels corresponding to journals.
def generate_feature_label_pair(mat):
    X = tokenizer.texts_to_sequences(mat.iloc[:, 0])
    X = pad_sequences(X, maxlen = MAX_SEQ_LENGTH, padding='post')
    Y = mat.iloc[:, 3].tolist()
    Y = labelEncoder.transform(Y)
    Y = to_categorical(Y, num_classes = len(labelEncoder.classes_))
    return X, Y

def get_topic_embedding(model, X):
    f = Model(inputs=model.input, outputs=model.layers[-1].input)
    with category_graph.as_default():
        return f.predict(X)

# WHY IS MODEL LAYER -2
def get_if_embedding(model, X):
    f = Model(inputs=model.input, outputs=model.layers[-2].output)
    with if_graph.as_default():
        return f.predict(X)

def convert2embedding(X):
    topic_embedding = get_topic_embedding(topic_model, X)
    if_embedding = get_if_embedding(if_model, X)
    embedding = np.concatenate((topic_embedding, if_embedding), axis = 1)
    return embedding

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
        embedding = convert2embedding(X)
        yield [np.array(X), np.array(embedding)], Y

def create_model():
    text_inputs = Input(shape = (MAX_SEQ_LENGTH, ), name = "text_input")
    embedding_layer = Embedding(MAX_NB_WORDS + 1,
                                EMBEDDING_DIM,
                                embeddings_initializer = Constant(embedding_matrix),
                                input_length = MAX_SEQ_LENGTH,
                                trainable = False)
    x = embedding_layer(text_inputs)

    # convolution 1st layer
    x = Conv1D(128, 5, activation='relu', input_shape = (200, 1))(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(5)(x)

    # convolution 2nd layer
    x = Conv1D(128, 5, activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(35)(x)
    x = Flatten()(x)

    embedding_input = Input(shape = (EMBEDDING_SIZE, ), name = "embedding_input")
    all_features = concatenate([x, embedding_input])

    x = Dense(units=1000, activation='relu', input_shape=(int_shape(all_features),))(all_features)
    x = BatchNormalization()(x)
    x = Dropout(0.1)(x)
    x = Dense(units=1000, activation='relu', input_shape=(int_shape(all_features),))(x)   
    outputs = Dense(units=len(labelEncoder.classes_), activation = 'softmax')(x)

    model = Model([text_inputs, embedding_input], outputs)
    model.compile(loss = 'categorical_crossentropy',
                 optimizer = keras.optimizers.Adam(lr=0.001), 
                 metrics = ['accuracy'])
    return model

def main():
    # Get number of training samples
    with open("data/train_j.txt") as f:
        nTrain = sum(1 for _ in f)
    dev = pd.read_table("data/dev_j.txt", delimiter="\t", header = 0)
    devText, devY = generate_feature_label_pair(dev)
    devEmbedding = convert2embedding(devText)
    nBatches = math.ceil(nTrain / BATCH_SIZE)
    model = create_model()
    model.fit_generator(sample_generator(), steps_per_epoch = nBatches, epochs=2, validation_data=([devText, devEmbedding], devY))
    model.save_weights("model/embedding3.h5")

if __name__ == "__main__":
    main()

