import requests
import math
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import sys
import time

def main():
    journal = pd.read_table("data/journals.txt", delimiter="\t", header = 0)
    h = {"User-Agent":"Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.75 Safari/537.36"}
    journalAbbrev = list(journal['abbreviation'])
    impactFactor = list(journal['Impact_factor'])
    nIF = 0
    for i in range(0, journal.shape[0], 1):
        if i % 100 == 0:
            print("journal %s, %s new impact factors filled." % (i, nIF))
        if math.isnan(impactFactor[i]):
            j = journalAbbrev[i]
            query = str(j) + " impact factor"
            query = query.replace(" ", "+")
            query = "https://www.google.com/search?q=" + query
            r = requests.get(query, headers=h).text
            time.sleep(3)
            soup = BeautifulSoup(r, 'lxml')
            ret = soup.select_one('div.Z0LcW')
            if ret is not None:
                journal.iloc[i, 4] = float(ret.text)
                nIF += 1
                if nIF == 10:
                    journal.to_csv("data/journals.txt", sep = '\t', index = False)
    print("%s number of new impact factors filled." % (nIF))
    idx = ~np.isnan(journal['Impact_factor'])
    journal = journal.loc[idx, :]
    print("%s number of journals in list." % (journal.shape[0]))
    journal.to_csv("data/journals.txt", sep = '\t', index = False)

if __name__ == "__main__":
    main()
