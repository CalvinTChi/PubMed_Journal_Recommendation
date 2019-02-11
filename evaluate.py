from keras.utils import to_categorical
from keras.models import Model, load_model
import tensorflow as tf
from utils import *
from keras.backend import int_shape
from sklearn.metrics import accuracy_score, auc
from keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
import math, pickle, sys
import numpy as np
import pandas as pd

B = 100

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

# INPUT: pandas df of rows x features, where features = [abstract, PMID, category, journalAbbrev, impact_factor]
# OUTPUT: (1) pandas df of rows x word2vec feature, (2) vector of labels corresponding to journals.
def generate_feature_label_pair(mat):
    X = tokenizer.texts_to_sequences(mat.iloc[:, 0])
    X = pad_sequences(X, maxlen = MAX_SEQ_LENGTH, padding='post')
    Y = mat.iloc[:, 3].tolist()
    Y = labelEncoder.transform(Y)
    return X, Y

def rank_predictions(class_prob):
    return sorted(range(len(class_prob)), key=lambda i: class_prob[i], reverse=True)

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

def calculate_auc(probYPred, testY):
    rankYPred = np.apply_along_axis(rank_predictions, 1, probYPred)
    topK = np.arange(0, len(labelEncoder.classes_), 10)
    pCoverage = [(x + 1) / len(labelEncoder.classes_) for x in topK]
    accuracies = []
    for k in topK:
        accuracies.append(k_coverage_accuracy(testY, rankYPred, k))
    return auc(pCoverage, accuracies), accuracies

def main(args):
    if len(args) == 0:
        print("Usage: evaluate.py <model_name>")
        sys.exit(2)
    filename = args[0] + ".h5"

    df = pd.DataFrame(index = range(B + 1), 
        columns = ['Accuracy', 'AUC', 'k90'])

    test = pd.read_table("data/test_j.txt", delimiter="\t", header = 0)
    testX, testY = generate_feature_label_pair(test)

    if args[0][:-1] == "embedding":
        testEmbedding = convert2embedding(testX)
        model = create_embedding_model()
        model.load_weights("model/" + filename)
        probYPred = model.predict([testX, testEmbedding])
    elif args[0][:-1] == "multitask":
        model = load_model("model/" + filename)
        _, probYPred, _ = model.predict(testX)
    else:
        model = load_model("model/" + filename)
        probYPred = model.predict(testX)

    # Calculate accuracy
    classYPred = np.argmax(probYPred, axis=1)
    accuracy = round(accuracy_score(testY, classYPred), 3)
    print("Accuracy on test dataset: %s" % (accuracy))
    df.loc[0, "Accuracy"] = accuracy
    
    # Calculate coverage auc
    auc, accuracies = calculate_auc(probYPred, testY)
    print("Coverage AUC on test dataset: %s" % (auc))
    df.loc[0, "AUC"] = round(auc, 3)
    
    # Find the coverage that gives 90% accuracy
    idx90 = next(idx for idx, value in enumerate(accuracies) if value > 0.9)
    k90 = topK[idx90]
    print("Coverage that yields 90%% accuracy: %s" % (k90))
    df.loc[0, "k90"] = k90

    # Plot title
    if args[0][:-1] == "embedding":
        title = "Embedding Model %s AUC" % (round(auc, 3))
    elif args[0][:-1] == "journal_baseline":
        title = "Baseline CNN Model %s AUC" % (round(auc, 3))
    elif args[0][:-1] == "multitask":
        title = "Multitask CNN Model %s AUC" % (round(auc, 3))

    # Plot coverage curve 
    pCoverage = [(x + 1) / len(labelEncoder.classes_) for x in topK]
    plot_auc(pCoverage, accuracies, title, args[0])

    # Bootstrap test samples to get error bars
    for i in range(B):
        if i % 10 == 0:
            print(i)
        testB = test.sample(n = df.shape[0], replace = True)
        testB_X, testB_Y = generate_feature_label_pair(testB)
        if args[0][:-1] == "multitask":
            _, probYPred, _ = model.predict(testB_X)
        elif args[0][:-1] == "embedding":
            testB_embedding = convert2embedding(testB_X)
            probYPred = model.predict([testB_X, testB_embedding])
        else:
            probYPred = model.predict(testB_X)
        # calculate accuracy
        classYPred = np.argmax(probYPred, axis=1)
        df.loc[i + 1, "Accuracy"] = round(accuracy_score(testB_Y, classYPred), 3)

        # calculate coverage auc
        auc, accuracies = calculate_auc(probYPred, testB_Y)
        df.loc[i + 1, "AUC"] = auc

        # Find the coverage that gives 90% accuracy
        idx90 = next(idx for idx, value in enumerate(accuracies) if value > 0.9)
        k90 = topK[idx90]
        df.loc[i + 1, "k90"] = k90

    output_file = args[0][:-1] + "_performance.csv"
    df.to_csv(output_file, index = False)

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
    #sys.exit(2)
    main(sys.argv[1:])

