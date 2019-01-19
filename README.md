# NLP259
Pubmed journal detection from abstract. This walkthrough is intended to aid the reproduction of results from this [report](https://drive.google.com/open?id=1pP6_46EBdg1Nuz-pZk5sknpvYdQh-FM8), using scripts in this repository. 

## Collecting the Data
[CiteFactor](http://www.citefactor.org) is a resource containing a list of journals and their impact factors (non-exhaustive). Run this script to scrape this list of journals and their impact factors. This script is based on this [script](http://www.biotechworld.it/bioinf/2016/01/02/scraping-impact-factor-data-from-the-web-using-httr-and-regex-in-r/) by Damiano Fantini.

`Rscript IF_scraping.R`

Next, download a comprehensive list of journals in PubMed from ftp://ftp.ncbi.nih.gov/pubmed/J_Medline.txt. 

Run this script to fill in impact factor of journals from Medline with scraped impact factors from CiteFactor.

`Rscript journal_IF.R`

Run this script to use Google search to fill in remaining missing impact factors of journals in Medline.

`python search_if.py`

Run this script to download abstracts from PubMed.

`python download_abstracts.py`

## Preprocess Data and Exploratory Data Analysis
Run this script to 

+ Split dataset into train, development, and test datasets.
+ Generate metadata for exploratory data analysis
+ Fit tokenizer: assign the top _n_ frequent words a unique index from 1 to _n_ (0 reserved for unassigned word)
+ Prepare embedding matrix for CNN
+ For journal detection, remove journals represented by less than 0.01% (~ 40 abstracts) of abstracts, and split dataset into train, development, and test datasets. This removed about 7.1% of all abstracts.

`python preprocess.py`

Run this script to make some exploratory data analysis plots from downloaded abstract

`Rscript eda.R`

## Topic and Impact Factor Prediction
Run `train_category.py` or `train_if.py` to train baseline CNN to predict topic or impact factor respectively. For example

`python train_category.py`

Run this script to visualize embeddings for topic prediction and/or impact factor prediction. To visualize embedding for impact factor or topic only, include only `topic` or `if` respectively.

`python visualize_embedding.py topic if`

## Journal Prediction
To train the (1) baseline CNN, (2) multi-task CNN, or (3) embedding-augmented CNN, run the scripts `baseline.py`, `multitask.py`, or `embedding_model.py` respectively. For example

`python embedding_model.py`

Following script evaluates performance of a model in terms of (1) accuracy, (2) coverage AUC, and (3) k to achieve 90% coverage accuracy. The script also plots the coverage accuracy vs percent coverage curve. To evaluate a model, include the saved model name at the end without the file ending. For example, to evaluate a trained model saved as `embedding2.h5`, run

`python evaluate.py embedding2`


