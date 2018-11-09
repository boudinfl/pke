#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pke

# create a Kea extractor and set the input language to English (used for
# the stoplist in the candidate selection method)
extractor = pke.supervised.Kea()

# load the content of the document, here in CoreNLP XML format
# the use_lemmas parameter allows to choose using CoreNLP lemmas or stems 
# computed using nltk
extractor.load_document('C-1.xml')

# select the keyphrase candidates, for Kea the 1-3 grams that do not start or
# end with a stopword.
extractor.candidate_selection()

# load the df counts
df_counts = pke.load_document_frequency_file(input_file="df.tsv.gz",
                                             delimiter='\t')

# weight the candidates using Kea model.
extractor.candidate_weighting(model_file="model.pickle", df=df_counts)

# print the n-highest (10) scored candidates
for (keyphrase, score) in extractor.get_n_best(n=10):
    print(keyphrase, score)
