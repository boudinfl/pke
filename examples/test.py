#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys

sys.path.append(os.environ['PATH_CODE'])

# this example uses SingleRank
from pke import TopicRank 

# create an unsupervised object
extractor = TopicRank(input_file='C-1.txt')

# load the content of the document, here in CoreNLP XML format
# the use_lemmas parameter allows to choose using CoreNLP lemmas or stems 
# computed using nltk
extractor.read_document(format='raw')


for s in extractor.sentences:
	print ' '.join([s.words[i]+"/"+s.pos[i] for i in range(s.length)]).encode('utf-8')

# select the keyphrase candidates, for SingleRank the longest sequences of 
# nouns and adjectives
# extractor.candidate_selection()

# # weight the candidates using a random walk
# extractor.candidate_weighting()

# # print the n-highest (10) scored candidates
# print (';'.join([u for u, v in extractor.get_n_best(n=10)])).encode('utf-8')









            