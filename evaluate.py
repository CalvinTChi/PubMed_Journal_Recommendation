from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import accuracy_score
from sklearn.metrics import auc
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pandas as pd
import pickle
import math
import sys
import os

BATCH_SIZE = 512
EMBEDDING_DIM = 200
MAX_SEQ_LENGTH = 500
tokenizer = pickle.load(open("data/tokenizer.p", "rb"))
labelEncoder = pickle.load(open("data/label_encoder.p", "rb"))
topic_model = None
category_graph = None
if_model = None
if_graph = None

def generate_feature_label_pair(mat):
    X = tokenizer.texts_to_sequences(mat.iloc[:, 0])
    X = pad_sequences(X, maxlen = MAX_SEQ_LENGTH, padding='post')
    Y = mat.iloc[:, 3].tolist()
    Y = labelEncoder.transform(Y)
    return X, Y

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
    model = load_model("model/" + filename)
    test = pd.read_table("data/test_j.txt", delimiter="\t", header = 0)
    testX, testY = generate_feature_label_pair(test)
    if args[0][:-1] == "embedding":
        global topic_model
        topic_model = load_model("model/category1.h5")
        global category_graph
        category_graph = tf.get_default_graph()
        global if_model
        if_model = load_model("model/impact_factor1.h5")
        global if_graph
        if_graph = tf.get_default_graph()
        testX = convert2embedding(testX)
    # Calculate accuracy
    classYPred = model.predict_classes(testX)
    print("Accuracy on test dataset: %s" % (round(accuracy_score(testY, classYPred), 3)))
    
    # Calculate coverage auc
    probYPred = model.predict_proba(testX)
    rankYPred = np.apply_along_axis(rank_predictions, 1, probYPred)
    topK = np.arange(0, len(labelEncoder.classes_), 10)
    pCoverage = [(x + 1) / len(labelEncoder.classes_) for x in topK]
    accuracies = []
    for k in topK:
        accuracies.append(k_coverage_accuracy(testY, rankYPred, k))
    print("Coverage AUC on test dataset: %s" % (round(auc(pCoverage, accuracies), 3)))
    
    # Find the coverage that gives 90% accuracy
    idx90 = next(idx for idx, value in enumerate(accuracies) if value > 0.9) 
    print("Coverage that yields %s accuracy" % (topK[idx90]))

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
    main(sys.argv[1:])



