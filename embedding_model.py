from keras.layers import Dense, Activation
from tensorflow.contrib.keras.api.keras.initializers import Constant
from keras.models import Model
import tensorflow as tf
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.layers import concatenate
from keras.models import load_model
import keras.optimizers
#from visualize_embedding import get_topic_embedding
#from visualize_embedding import get_if_embedding
from baseline import generate_feature_label_pair
import math
import numpy as np
import pandas as pd
import keras
import pickle
import sys

MAX_SEQ_LENGTH = 500
BATCH_SIZE = 512
MAX_SEQ_LENGTH = 500
EMBEDDING_SIZE = 256

embedding_matrix = pickle.load(open("data/embedding.p", "rb"))
tokenizer = pickle.load(open("data/tokenizer.p", "rb"))
labelEncoder = pickle.load(open("data/label_encoder.p", "rb"))
topic_model = load_model("model/category1.h5")
global category_graph
category_graph = tf.get_default_graph()
if_model = load_model("model/impact_factor1.h5")
global if_graph
if_graph = tf.get_default_graph()

trainIterator = pd.read_table("data/train_j.txt", delimiter="\t", header = 0, chunksize=BATCH_SIZE)
trainIterator = iter(trainIterator)

def get_topic_embedding(model, X):
    f = Model(inputs=model.input, outputs=model.layers[-1].input)
    with category_graph.as_default():
        return f.predict(X)

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
        yield embedding, Y

def create_model():
    text_inputs = Input(shape = (MAX_SEQ_LENGTH, ))
    word_index = tokenizer.word_index
    embedding_layer = Embedding(len(word_index) + 1,
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
    x = Conv1D(128, 5, activation='relu'))(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(35)(x)

    embedding_input = Input(shape = (EMBEDDING_SIZE, ), name = "embedding_input")
    all_features = concatenate([x, embedding_input])

    x = Dense(units=1200, activation='relu', input_shape=(all_features.output_shape,))(all_features)
    x = BatchNormalization()(x)
    x = Dense(units=800, activation='relu')(x)
    x = BatchNormalization()(x)
    outputs = Dense(units=len(labelEncoder.classes_), activation = 'softmax')(x)

    model = Model([text_inputs, embedding_input], outputs)
    model.compile(loss = 'categorical_crossentropy',
                 optimizer = keras.optimizers.Adam(lr=0.001), 
                 metrics = ['accuracy'])

    #model = Sequential()
    #model.add(Dense(units=1200, activation='relu', input_shape=(EMBEDDING_SIZE,)))
    #model.add(BatchNormalization())
    #model.add(Dense(units=800, activation='relu'))
    #model.add(BatchNormalization())

    #model.add(Dense(units=len(labelEncoder.classes_), activation = 'softmax'))
    #model.compile(loss = 'categorical_crossentropy',
    #             optimizer = keras.optimizers.Adam(lr=0.001), 
    #             metrics = ['accuracy'])
    return model

def main():
    # Get number of training samples
    with open("data/train_j.txt") as f:
        nTrain = sum(1 for _ in f)
    dev = pd.read_table("data/dev_j.txt", delimiter="\t", header = 0)
    devX, devY = generate_feature_label_pair(dev)
    devX = convert2embedding(devX)
    nBatches = math.ceil(nTrain / BATCH_SIZE)
    model = create_model()
    #model.fit_generator(sample_generator(), steps_per_epoch = nBatches, epochs=2, validation_data=(devX, devY))
    #model.save("model/embedding2.h5")

if __name__ == "__main__":
    main()

