#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

import pke

base = os.path.dirname(__file__)

# create a Kea extractor and set the input language to English (used for
# the stoplist in the candidate selection method)
extractor = pke.supervised.Kea()

# load the content of the document, here in corenlp format
with open(base + os.sep + '2.txt') as f:
    doc = f.read()
extractor.load_document(doc)

# select the keyphrase candidates, for Kea the 1-3 grams that do not start or
# end with a stopword.
extractor.candidate_selection()

# load the df counts
df_counts = pke.load_document_frequency_file(
    input_file=base + os.sep + 'df.tsv.gz',
    delimiter='\t')

# weight the candidates using Kea model.
extractor.candidate_weighting(
    model_file=base + os.sep + 'model.pickle',
    df=df_counts)

# print the n-highest (10) scored candidates
for (keyphrase, score) in extractor.get_n_best(n=10):
    print(keyphrase, score)
