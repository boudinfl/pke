#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, sys
sys.path.append(os.environ['PATH_CODE']+'/pke')

from pke.unsupervised import TfIdf

from pke.unsupervised import KPMiner

# create a KPMiner extractor and set the input language to English (used
# for the stoplist in the candidate selection method). The input file
# is considered to be in Stanford XML CoreNLP.
extractor = KPMiner(input_file='C-1.xml', language='english')

# load the content of the document.
extractor.read_document(format='corenlp')

# select the keyphrase candidates, by default the 1-5-grams of words
# that do not contain punctuation marks or stopwords. Candidates
# occurring less than 3 times or after the 400th word are filtered out.
extractor.candidate_selection()

# available parameters are the least allowable seen frequency and the 
# number of words after which candidates are filtered out.
# >>> lasf = 5
# >>> cutoff = 123
# >>> extractor.candidate_selection(lasf=lasf, cutoff=cutoff)

# weight the candidates using KPMiner weighting function.
extractor.candidate_weighting()

# available parameters are the `df` counts that can be provided to the 
# weighting function and the sigma and alpha values of the weighting
# function.
# >>> counts = {'--NB_DOC--': 3, word1': 3, 'word2': 1, 'word3': 2}
# >>> alpha = 2.3
# >>> sigma = 3.0
# >>> extractor.candidate_weighting(df=counts, alpha=alpha, sigma=sigma)

# get the 10-highest scored candidates as keyphrases
keyphrases = extractor.get_n_best(n=10)

# available parameters are whether redundant candidates are filtered out
# (default to False) and if stemming is applied to candidates (default
# to True)
# >>> redundancy_removal=True
# >>> stemming=False
# >>> keyphrases = extractor.get_n_best(n=10,
# >>>     redundancy_removal=redundancy_removal,
# >>>     stemming=stemming)

print(keyphrases)