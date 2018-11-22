from keras.preprocessing.sequence import pad_sequences
from keras import backend as K
from keras.models import load_model
from keras.models import Model
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import getopt
import numpy as np
import pandas as pd
import keras
import pickle
import sys

label_mapping = {0:'bioinformatics', 1:'development', 2:'epigenetics', 3:'mendelian', 
                4:'omics', 5:'population_genetics', 6:'statistical_genetics', 7:'structure'}
topic_colors = {'bioinformatics': 'pink', 'development': 'g', 'epigenetics': 'r', 'mendelian': 'c',
				'omics': 'm', 'population_genetics': 'y', 'statistical_genetics': 'k', 'structure': 'b'}
if_color = {'nature': 'b', 'gene': 'r', 'science': 'g'}
tokenizer = pickle.load(open("data/tokenizer.p", "rb"))
trainIterator = pd.read_table("data/train.txt", delimiter="\t", header = 0, chunksize=1000)
MAX_SEQ_LENGTH = 500
global model

def generate_abstract_label(mat, yIdx):
    X = tokenizer.texts_to_sequences(mat.iloc[:, 0])
    X = pad_sequences(X, maxlen = MAX_SEQ_LENGTH, padding='post')
    Y = mat.iloc[:, yIdx].tolist()
    return X, Y

def get_topic_embedding(model, X):
	f = Model(inputs=model.input, outputs=model.layers[-1].input)
	return f.predict(X)
	#embedding = f.predict(X)
	#W = model.layers[-1].get_weights()[0]
	#b = model.layers[-1].get_weights()[1]
	#b = np.reshape(b, (1, len(b)))
	#embedding = np.matmul(embedding, W) + np.matmul(np.ones((embedding.shape[0], 1)), b)
	#return embedding

def get_if_embedding(model, X):
	f = Model(inputs=model.input, outputs=model.layers[-2].output)
	return f.predict(X)

def get_journal_chunks(lst):
	frames = []
	for chunk in trainIterator:
		frames.append(chunk.loc[chunk.iloc[:, 3].isin(lst), :])
	return pd.concat(frames)

def plot_topic_pca(embedding, Y, title, filename):
	pca = PCA(n_components=5)
	x_r = pca.fit_transform(embedding)
	fig, ax = plt.subplots()
	for y in np.unique(Y):
	    ix = np.where(np.array(Y) == y)
	    ax.scatter(x_r[ix[0], 0], x_r[ix[0], 1], c = topic_colors[y], label = y, s = 10, alpha = 0.5)
	lgd = ax.legend(bbox_to_anchor=(1.05, 0.7))
	plt.rcParams["figure.figsize"] = [8, 6]
	plt.title(title)
	plt.xlabel("PC1")
	plt.ylabel("PC2")
	plt.savefig("pics/" + filename + ".png", bbox_extra_artists=(lgd,), bbox_inches='tight')

def plot_if_pca(embedding, Y, title, filename):
	pca = PCA(n_components=5)
	x_r = pca.fit_transform(embedding)
	fig, ax = plt.subplots()
	for y in np.unique(Y):
	    ix = np.where(np.array(Y) == y)
	    ax.scatter(x_r[ix[0], 0], x_r[ix[0], 1], c = if_color[y], label = y, s = 10, alpha = 0.5)
	lgd = ax.legend(bbox_to_anchor=(1.2, 0.6))
	plt.rcParams["figure.figsize"] = [8, 6]
	plt.title(title)
	plt.xlabel("PC1")
	plt.ylabel("PC2")
	plt.savefig("pics/" + filename + ".png", bbox_extra_artists=(lgd,), bbox_inches='tight')

def main(args):
	if len(args) == 0:
		print("Usage: visualize_embedding.py <embedding_type1> <embedding_type2> ...")
		sys.exit(2)
	for arg in args:
		print(arg)
		if arg == "topic":
			title = "PCA of Last Layer Embedding for Topic Prediction"
			model = load_model("model/category1.h5")
			chunk = next(iter(trainIterator))
			yIdx = 2
			filename = "topic_embedding"
			X, Y = generate_abstract_label(chunk, yIdx)
			embedding = get_topic_embedding(model, X)
			plot_topic_pca(embedding, Y, title, filename)
		elif arg == "if":
			title = "PCA of Last Layer Embedding for Impact Factor Prediction"
			model = load_model("model/impact_factor1.h5")
			lst = ["gene", "nature", "science"]
			chunk = get_journal_chunks(lst)
			yIdx = 3
			filename = "if_embedding"
			X, Y = generate_abstract_label(chunk, yIdx)
			embedding = get_if_embedding(model, X)
			plot_if_pca(embedding, Y, title, filename)
		
if __name__ == "__main__":
	try:
		arg = sys.argv[1:]
	except:
		print("Usage: visualize_embedding.py <embedding_type1> <embedding_type2> ...")
		sys.exit(2)
	main(sys.argv[1:])

