from Bio import Entrez
from Bio import Medline
import numpy as np
import pandas as pd
import math
import time
import sys

def main():
	journal = pd.read_table("data/journals.txt", delimiter="\t", header = 0)
	journalAbbrev = journal["abbreviation"].tolist()

	Entrez.email = email = "calvin.t.chi@gmail.com"
	testSearch = "Bioinformatics"

	# Get review PMID
	handle = Entrez.esearch(db="pubmed", term=testSearch + " and Review[PT]")
	record = Entrez.read(handle)
	count = record['Count']
	handle = Entrez.esearch(db="pubmed", term=testSearch + " and Review[PT]", retmax = int(count))
	record = Entrez.read(handle)
	reviewID = set(record["IdList"])

	# Get all PMID
	handle = Entrez.esearch(db="pubmed", term=testSearch)
	record = Entrez.read(handle)
	count = record['Count']
	handle = Entrez.esearch(db="pubmed", term=testSearch, retmax = int(count))
	record = Entrez.read(handle)
	IDlist = set(record["IdList"])

	IDlist = list(IDlist.difference(reviewID))

	batch_size = 10000
	df = pd.DataFrame(columns=['abstract', 'PMID', 'category', 'journalAbbrev', 'impact_factor'])
	presentYear = 2018

	start_time = time.time()
	print("Start downloading...")
	nJournalProblem = 0
        nNoAbstract = 0
        nOld = 0
	nFields = 0
	for i in range(0, len(IDlist), batch_size):
	    print("%s hours elapsed: batch %s of %s downloaded" % (round((time.time() - start_time) / 3600.0, 2), 
	                                                           i, len(IDlist)))
	    handle = Entrez.efetch(db="pubmed", id=IDlist, rettype="medline", 
	                       retmode="text", retmax = batch_size, retstart = i)
	    records = Medline.parse(handle)
	    for record in records:
	        if {'AB', 'JT', 'PHST'} <= set(record.keys()):
	            journ = record['JT']
	            year = int(record['PHST'][0][:4])
	            abstract = record['AB']
	            if journ not in journalAbbrev or year < (presentYear - 10) or len(abstract) == 0:
	            	if journ not in journalAbbrev:
                            nJournalProblem += 1
                        elif year < (presentYear - 10):
                            nOld += 1
                        elif len(abstract) == 0
                            nNoAbstract += 1
	            	continue
	            impactFactor = float(journal.loc[journal['abbreviation'] == journ, 'Impact_factor'])
	            impactFactor = round(impactFactor, 3)
	            content = [[abstract, record['PMID'], "bioinformatics", journ, impactFactor]]
	            df = df.append(pd.DataFrame(content, columns = df.columns))
	        else:
	        	nFields += 1
	print("%s abstracts downloaded" % (df.shape[0]))
        print("%s number of abstracts without fields AB, JT, or PHST" % (nFields))
        print("%s number of abstracts with journals not in list" % (nJournalProblem))
        print("%s number of abstracts too old" % (nOld))
        print("%s number of abstracts with length 0" % (nNoAbstract))
	df.to_csv("data/abstracts.txt", sep='\t')

if __name__ == "__main__":
	main()






