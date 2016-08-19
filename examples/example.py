#!/usr/bin/env python
# -*- coding: utf-8 -*-

# this example uses SingleRank
from pke import TopicRank 

# create an unsupervised object
extractor = TopicRank(input_file='C-1.xml')

# load the content of the document, here in CoreNLP XML format
# the use_lemmas parameter allows to choose using CoreNLP lemmas or stems 
# computed using nltk
extractor.read_corenlp_document(use_lemmas=False)

# select the keyphrase candidates, for SingleRank the longest sequences of 
# nouns and adjectives
extractor.candidate_selection()

# weight the candidates using a random walk
extractor.candidate_weighting()

# print the n-highest (10) scored candidates
print (';'.join([u for u, v in extractor.get_n_best(n=10)])).encode('utf-8')









            