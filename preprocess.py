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

# p = percentage dataset to allocate to development and test dataset. rest is training
# Splits entire dataset into train, dev, and test
def split_dataset(p = [0.1, 0.1]):
    abstracts = pd.read_table("data/abstracts.txt", delimiter="\t", header = 0, chunksize = 50000)
    train = pd.DataFrame(columns=['abstract', 'PMID', 'category', 'journalAbbrev', 'impact_factor'])
    dev = pd.DataFrame(columns=['abstract', 'PMID', 'category', 'journalAbbrev', 'impact_factor'])
    test = pd.DataFrame(columns=['abstract', 'PMID', 'category', 'journalAbbrev', 'impact_factor'])
    train.to_csv("data/train.txt", index = False, sep = '\t')
    dev.to_csv("data/dev.txt", index = False, sep = '\t')
    test.to_csv("data/test.txt", index = False, sep = '\t')
    for chunk in abstracts:
        train, dev, test = train_dev_test_split(chunk, p)
        with open("data/train.txt", "a") as f:
            train.to_csv(f, header = False, index = False, sep = '\t')
        with open("data/dev.txt", "a") as f:
            dev.to_csv(f, header = False, index = False, sep = '\t')
        with open("data/test.txt", "a") as f:
            test.to_csv(f, header = False, index = False, sep = '\t')

# INPUT: chunk = data frame of subset of data, p = percentage dataset to allocate to development and test dataset
# OUTPUT: dataframes of train, development, and test dataset
def train_dev_test_split(chunk, p = [0.1, 0.1]):
    n = chunk.shape[0]
    devP = round(n * p[0])
    testP = round(n * p[1])
    dev = chunk.iloc[0:devP, :]
    test = chunk.iloc[devP:(devP + testP), :]
    train = chunk.iloc[(devP + testP):n, :]
    return train, dev, test

# save research topic, journal abbreviation, and impact factor as metadata 
# to perform data summary statistics on later
def generate_metadata():
    abstracts = pd.read_table("data/abstracts.txt", delimiter="\t", header = 0, chunksize = 50000)
    df = pd.DataFrame(columns=['category', 'journalAbbrev', 'impact_factor'])
    df.to_csv("data/metadata.txt", sep = '\t', index = False)
    for chunk in abstracts:
        chunk = chunk.iloc[:, 2:5]
        with open("data/metadata.txt", "a") as f:
            chunk.to_csv(f, header = False, index = False, sep = '\t')

def batch_generator():
    while True:
        chunk = next(trainIterator)
        chunk = chunk.iloc[:, 0]
        yield chunk.tolist()

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
            embedding_matrix[i] = vec / np.linalg.norm(vec)
    return embedding_matrix

# Shuffle rows of dataset
def shuffle_dataset():
    abstracts = pd.read_table("data/abstracts.txt", delimiter="\t", header = 0)
    abstracts = abstracts.sample(frac=1)
    abstracts.to_csv("data/abstracts.txt", index = False, sep = '\t')

def main():
    shuffle_dataset()
    if not os.path.isfile("data/train.txt") or not os.path.isfile("data/dev.txt") or not os.path.isfile("data/test.txt"):
        split_dataset()
    generate_metadata()
    batches = batch_generator()
    for batch in batches:
        tokenizer.fit_on_text(batch)
        count_unique_words(batch)
    embedding_matrix = prepare_embedding_matrix()
    pickle.dump(tokenizer, open("data/tokenizer.p", "wb"))
    pickle.dump(embedding_matrix, open("data/embedding.p", "wb"))
    print("Number of unique words: %s" % len(uniqueWords))

if __name__ == "__main__":
    main()









