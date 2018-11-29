from keras.layers import Dense, Flatten, Embedding, Conv1D, MaxPooling1D, Activation, Input, concatenate, Dropout
from tensorflow.contrib.keras.api.keras.initializers import Constant
from keras.models import Model, load_model
import tensorflow as tf
from utils import *
from keras.backend import int_shape
from sklearn.metrics import accuracy_score, auc
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.sequence import pad_sequences
import keras.optimizers
import matplotlib.pyplot as plt
import math, pickle, sys
import numpy as np
import pandas as pd

EMBEDDING_SIZE = 256

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

def rank_predictions(class_prob):
    return sorted(range(len(class_prob)), key=lambda i: class_prob[i], reverse=True)

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

def k_coverage_accuracy(ytrue, ypred, k):
    cover = []
    ypred = ypred[:, :k]
    for i in range(ypred.shape[0]):
        if ytrue[i] in ypred[i, :]:
            cover.append(1)
        else:
            cover.append(0)
    return np.mean(cover)

def plot_auc(pCoverage, accuracies, title, filename):
    plt.plot(pCoverage, accuracies)
    plt.xlabel("percent coverage")
    plt.ylabel("coverage accuracy")
    plt.title(title)
    plt.savefig("pics/" + filename + ".png", dpi=300)

def main(args):
    if len(args) == 0:
        print("Usage: evaluate.py <model_name>")
        sys.exit(2)
    filename = args[0] + ".h5"
    test = pd.read_table("data/test_j.txt", delimiter="\t", header = 0)
    testX, testY = generate_feature_label_pair(test, 3)
    if args[0][:-1] == "embedding":
        testEmbedding = convert2embedding(testX)
        model = create_embedding_model()
        model.load_weights("model/" + filename)
    else:
        model = load_model("model/" + filename)

    if args[0][:-1] == "multitask":
        _, probYPred, _ = model.predict(testX)
    elif args[0][:-1] == "embedding":
        probYPred = model.predict([testX, testEmbedding])
    else:
        probYPred = model.predict(testX)
    # Calculate accuracy
    classYPred = np.argmax(probYPred, axis=1)
    print("Accuracy on test dataset: %s" % (round(accuracy_score(testY, classYPred), 3)))
    
    # Calculate coverage auc
    rankYPred = np.apply_along_axis(rank_predictions, 1, probYPred)
    topK = np.arange(0, len(labelEncoder.classes_), 10)
    pCoverage = [(x + 1) / len(labelEncoder.classes_) for x in topK]
    accuracies = []
    for k in topK:
        accuracies.append(k_coverage_accuracy(testY, rankYPred, k))
    print("Coverage AUC on test dataset: %s" % (round(auc(pCoverage, accuracies), 3)))
    
    # Find the coverage that gives 90% accuracy
    idx90 = next(idx for idx, value in enumerate(accuracies) if value > 0.9) 
    print("Coverage that yields 90%% accuracy: %s" % (topK[idx90]))

    # Plot title
    if args[0][:-1] == "embedding":
        title = "Embedding Model %s AUC" % (round(auc(pCoverage, accuracies), 3))
    elif args[0][:-1] == "journal_baseline":
        title = "Baseline CNN Model %s AUC" % (round(auc(pCoverage, accuracies), 3))
    elif args[0][:-1] == "multitask":
        title = "Multitask CNN Model %s AUC" % (round(auc(pCoverage, accuracies), 3))

    # Plot coverage curve
    plot_auc(pCoverage, accuracies, title, args[0])

if __name__ == "__main__":
    try:
        arg = sys.argv[1:]
    except:
        print("Usage: evaluate.py <model_name>")
        sys.exit(2)
    if sys.argv[1:][0][:-1] == "embedding":
        topic_model = load_model("model/category1.h5")
        global category_graph
        category_graph = tf.get_default_graph()
        if_model = load_model("model/impact_factor1.h5")
        global if_graph
        if_graph = tf.get_default_graph()
    main(sys.argv[1:])

