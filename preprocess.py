from keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
import numpy as np
import pandas as pd
import gensim
import pickle
import os
import sys

JOURNAL_MIN_NUM = 40
BATCH_SIZE = 1000
MAX_NB_WORDS = 750000
EMBEDDING_DIM = 200
journalSet = None

trainIterator = pd.read_table("data/train.txt", delimiter="\t", header = 0, chunksize=BATCH_SIZE)
word2vec = gensim.models.KeyedVectors.load_word2vec_format('data/PubMed-and-PMC-w2v.bin', 
                                                   binary=True)
trainIterator = iter(trainIterator)
uniqueWords = set()
tokenizer = Tokenizer(num_words=MAX_NB_WORDS)

# p = percentage dataset to allocate to development and test dataset. rest is training (for journal prediction)
# Splits entire dataset into train, dev, and test
def journal_prediction_split_dataset(p = [0.1, 0.1]):
    metadata = pd.read_table("data/metadata.txt", delimiter = "\t", header = 0)
    journalAbbrev = metadata['journalAbbrev'].tolist()
    count = {x: journalAbbrev.count(x) for x in set(journalAbbrev)}
    global journalSet
    journalSet = set([j for j in journalAbbrev  if count[j] >= JOURNAL_MIN_NUM])
    # Process dev dataset
    dev = pd.read_table("data/dev.txt", delimiter="\t", header = 0)
    devJ = dev.iloc[np.where(dev['journalAbbrev'].isin(journalSet))]
    devJ.to_csv("data/dev_j.txt", index = False, sep = '\t')
    # Process test dataset
    test = pd.read_table("data/test.txt", delimiter="\t", header = 0)
    testJ = test.iloc[np.where(test['journalAbbrev'].isin(journalSet))]
    testJ.to_csv("data/test_j.txt", index = False, sep = '\t')
    # Process training dataset
    train = pd.read_table("data/train.txt", delimiter="\t", header = 0, chunksize = 50000)
    trainJ = pd.DataFrame(columns=['abstract', 'PMID', 'category', 'journalAbbrev', 'impact_factor'])
    trainJ.to_csv("data/train_j.txt", index = False, sep = '\t')
    for chunk in train:
        trainJ = chunk.iloc[np.where(chunk['journalAbbrev'].isin(journalSet))]
        with open("data/train_j.txt", "a") as f:
            trainJ.to_csv(f, header = False, index = False, sep = '\t')

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
    # Prepare datasets for category and impact factor prediction
    if not os.path.isfile("data/train.txt") or not os.path.isfile("data/dev.txt") or not os.path.isfile("data/test.txt"):
        split_dataset()
    generate_metadata()
    # Prepare datasets for journal prediction
    if not os.path.isfile("data/train_j.txt") or not os.path.isfile("data/dev_j.txt") or not os.path.isfile("data/test_j.txt"):
        journal_prediction_split_dataset()
        # Encode journal abbreviations into integers from [0, num_journals]
        encoder = LabelEncoder()
        encoder.fit(list(journalSet))
        pickle.dump(encoder, open("data/label_encoder.p", "wb"))
    # Tokenize all words in abstract
    batches = batch_generator()
    for batch in batches:
        tokenizer.fit_on_text(batch)
        count_unique_words(batch)
    #Prepare embedding matrix
    embedding_matrix = prepare_embedding_matrix()
    pickle.dump(tokenizer, open("data/tokenizer.p", "wb"))
    pickle.dump(embedding_matrix, open("data/embedding.p", "wb"))
    print("Number of unique words: %s" % len(uniqueWords))

if __name__ == "__main__":
    main()


