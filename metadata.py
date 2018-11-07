import numpy as np
import pandas as pd

abstracts = pd.read_table("data/abstracts.txt", delimiter="\t", header = 0, chunksize = 50000)

df = pd.DataFrame(columns=['category', 'journalAbbrev', 'impact_factor'])
df.to_csv("data/metadata.txt", sep = '\t', index = False)

n = 1
for chunk in abstracts:
    print("Chunk %s" % (n))
    chunk = chunk.iloc[:, 2:5]
    with open("data/metadata.txt", "a") as f:
        chunk.to_csv(f, header = False, index = False, sep = '\t')
    n += 1
