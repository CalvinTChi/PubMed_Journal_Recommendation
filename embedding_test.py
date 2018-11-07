import gensim

model = gensim.models.KeyedVectors.load_word2vec_format('data/PubMed-and-PMC-w2v.bin', binary=True)

print(model['protein'])
