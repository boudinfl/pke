#!/usr/bin/env python
# -*- coding: utf-8 -*-

# this example uses TopicRank
from pke.unsupervised import TopicRank

# create a TopicRank extractor
extractor = TopicRank()

# load the content of the document, here in CoreNLP XML format
# the input language is set to English (used for the stoplist)
# normalization is set to stemming (computed with Porter's stemming algorithm)
extractor.load_document(input='C-1.xml',
                        language="en",
                        normalization='stemming')

# select the keyphrase candidates, for TopicRank the longest sequences of 
# nouns and adjectives
extractor.candidate_selection(pos={'NOUN', 'PROPN', 'ADJ'})

# weight the candidates using a random walk. The threshold parameter sets the
# minimum similarity for clustering, and the method parameter defines the 
# linkage method
extractor.candidate_weighting(threshold=0.74,
                              method='average')

# print the n-highest (10) scored candidates
for (keyphrase, score) in extractor.get_n_best(n=10, stemming=True):
    print(keyphrase, score)
