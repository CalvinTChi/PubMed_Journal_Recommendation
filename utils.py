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

def create_model():
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