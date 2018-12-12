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
