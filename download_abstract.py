from Bio import Entrez
from Bio import Medline
import numpy as np
import pandas as pd
import math
import time
import sys


# INPUT: search term for PubMed
# RETURN: set of PMIDs of search results - search results that are review articles
def getArticlePMID(search):
    # Get review PMID
    handle = Entrez.esearch(db="pubmed", term=search + " and Review[PT]")
    record = Entrez.read(handle)
    count = record['Count']
    handle = Entrez.esearch(db="pubmed", term=search + " and Review[PT]", retmax = int(count))
    record = Entrez.read(handle)
    reviewID = set(record["IdList"])

    # Get all PMID
    handle = Entrez.esearch(db="pubmed", term=search)
    record = Entrez.read(handle)
    count = record['Count']
    handle = Entrez.esearch(db="pubmed", term=search, retmax = int(count))
    record = Entrez.read(handle)
    IDlist = set(record["IdList"])

    IDlist = IDlist.difference(reviewID)
    return IDlist

# INPUT: dictionary where keys are topics and values are search words for PubMed
# RETURN: dictionary where keys are topics and values are list of mutually exclusive PMIDs by topic
def getPMIDbyTopic(topics):
    intersection = set()
    allIDs = set()
    retID = {}

    for topic in topics:
        ret = set()
        for search in topics[topic]:
            ret.update(getArticlePMID(search))
        intersection.update(allIDs.intersection(ret))
        allIDs.update(ret)
        retID[topic] = ret
    
    for topic in retID:
        retID[topic] = list(retID[topic].difference(intersection))

    allIDs.clear()
    intersection.clear()
    ret.clear()
    return retID


def main():
    journal = pd.read_table("data/journals.txt", delimiter="\t", header = 0)
    journalAbbrev = journal["abbreviation"].tolist()
    journalAbbrev = [str(x).lower() for x in journalAbbrev]
    d = journal.shape[1]

    Entrez.email = email = "calvin.t.chi@gmail.com"

    # All search terms
    topics = {"mendelian": ["mendelian disease"], 
          "complex_trait": ["complex traits NOT epidemiology", "polygenic disorder NOT epidemiology"],
          "statistical_genetics": ["statistical genetics", "genetic epidemiology"],
          "population_genetics": ["population genetics"],
          "bioinformatics": ["bioinformatics", "bioinformatics analysis"],
          "omics": ["omics", "multi omics"],
          "structure": ["genome structure", "cytogenetics"],
          "epigenetics": ["epigenetics"],
          "development": ["developmental genetics"]
         }

    # Get mutually exclusive set of research article PMIDs for each topic
    retID = getPMIDbyTopic(topics)
    print("Number of research article abstracts to download: %s" % (sum([len(v) for k, v in retID.items()])))
    
    batch_size = 10000
    df = pd.DataFrame(columns=['abstract', 'PMID', 'category', 'journalAbbrev', 'impact_factor'])
    presentYear = 2018

    start_time = time.time()
    print("Download starting...")
    nJournalProblem = 0
    nMissingFields = 0
    for topic in retID:
        IDlist = retID[topic]
        for i in range(0, len(IDlist), batch_size):
            print("%s hours elapsed: abstract %s of %s downloaded" % (round((time.time() - start_time) / 3600.0, 2), 
                                                                   i + 1, len(IDlist)))
            handle = Entrez.efetch(db="pubmed", id=IDlist, rettype="medline", 
                               retmode="text", retmax = batch_size, retstart = i)
            records = Medline.parse(handle)
            for record in records:
                if {'AB', 'TA', 'PHST'} <= set(record.keys()):
                    journ = record['TA'].lower()
                    year = int(record['PHST'][0][:4])
                    abstract = record['AB']
                    if journ not in journalAbbrev or year < (presentYear - 10) or len(abstract) == 0:
                        if journ not in journalAbbrev:
                            nJournalProblem += 1
                        continue
                    impactFactor = float(journal.iloc[journalAbbrev.index(journ), d - 1])
                    impactFactor = round(impactFactor, 3)
                    content = [[abstract, record['PMID'], topic, journ, impactFactor]]
                    df = df.append(pd.DataFrame(content, columns = df.columns))
                else:
                    nMissingFields += 1
    print("%s abstracts downloaded" % (df.shape[0]))
    print("%s number of abstracts without fields AB, TA, or PHST" % (nMissingFields))
    print("%s number of abstracts with journals not in list" % (nJournalProblem))
    df.to_csv("data/abstracts.txt", sep='\t', index = False)

if __name__ == "__main__":
    main()

