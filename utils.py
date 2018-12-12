#from keras.preprocessing.sequence import pad_sequences
#import numpy as np
#import pandas as pd
import pickle

EMBEDDING_SIZE = 256
BATCH_SIZE = 512
EMBEDDING_DIM = 200
MAX_SEQ_LENGTH = 500
tokenizer = pickle.load(open("data/tokenizer.p", "rb"))
labelEncoder = pickle.load(open("data/label_encoder.p", "rb"))
embedding_matrix = pickle.load(open("data/embedding.p", "rb"))

def create_embedding_model():
    text_inputs = Input(shape = (MAX_SEQ_LENGTH, ), name = "text_input")
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

