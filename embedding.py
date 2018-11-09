from keras.preprocessing.text import Tokenizer
import numpy as np
import pandas as pd
import gensim
import pickle
import os
import sys

BATCH_SIZE = 1000
MAX_NB_WORDS = 750000
EMBEDDING_DIM = 200

global trainIterator
trainIterator = pd.read_table("data/train.txt", delimiter="\t", header = 0, chunksize=BATCH_SIZE)
word2vec = gensim.models.KeyedVectors.load_word2vec_format('data/PubMed-and-PMC-w2v.bin', 
                                                    binary=True)

trainIterator = iter(trainIterator)
uniqueWords = set()
tokenizer = Tokenizer(num_words=MAX_NB_WORDS)

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

def batch_generator():
    while True:
        chunk = next(trainIterator)
        chunk = chunk.iloc[:, 0]
        yield chunk.tolist()

def process_batch(batch):
    count_unique_words(batch)
    tokenizer.fit_on_texts(batch)

def count_unique_words(batch):
    for abstract in batch:
        abstract = abstract.split(" ")
        uniqueWords.update(abstract)

def prepare_embedding_matrix():
    word_index = tokenizer.word_index
    embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
    for word, i in word_index.items():
        if word in word2vec:
            vec = word2vec[word]
            vec = vec / np.linalg.norm(vec)
            embedding_matrix[i] = vec
    return embedding_matrix

def main():
    if not os.path.isfile("data/train.txt") or not os.path.isfile("data/dev.txt") or not os.path.isfile("data/test.txt"):
        train_dev_test_split()
    batches = batch_generator()
    for batch in batches:
        process_batch(batch)
    embedding_matrix = prepare_embedding_matrix()
    pickle.dump(tokenizer, open("data/tokenizer.p", "wb"))
    pickle.dump(embedding_matrix, open("data/embedding.p", "wb"))
    print("Number of unique words: %s" % len(uniqueWords))

if __name__ == "__main__":
    main()



