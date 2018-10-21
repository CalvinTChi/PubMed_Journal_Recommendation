import requests
import math
from bs4 import BeautifulSoup

def main():
	journal = pd.read_table("data/journals.txt", delimiter="\t", header = 0)
	h = {"User-Agent":"Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.75 Safari/537.36"}
	journalAbbrev = list(journal['abbreviation'])
	impactFactor = list(journal['Impact_factor'])
	nIF = 0
	for i in range(journal.shape[0]):
		if i % 1000 == 0:
			print("journal %s" % (i))
	    if math.isnan(impactFactor[i]):
	        j = journalAbbrev[i]
	        query = j + " impact factor"
	        query = query.replace(" ", "+")
	        query = "https://www.google.com/search?q=" + query
	        r = requests.get(query, headers=h).text
	        soup = BeautifulSoup(r, 'lxml')
	        ret = soup.select_one('div.Z0LcW')
	        if ret is not None:
	            journal.iloc[i, 4] = float(ret.text)
	            nIF += 1
	print("%s number of new impact factors filled." % (nIF))
	idx = ~np.isnan(journal['Impact_factor'])
	journal = journal.loc[idx, :]
	print("%s number of journals in list." % (journal.shape[0]))
	journal.to_csv("data/journals.txt", sep = '\t')

if __name__ == "__main__":
	main()
