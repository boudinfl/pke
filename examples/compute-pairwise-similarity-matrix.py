#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
from __future__ import absolute_import

import sys
import string
import logging

from nltk.corpus import stopwords

from pke import load_document_frequency_file, compute_pairwise_similarity_matrix

# setting info in terminal
logging.basicConfig(level=logging.INFO)

# path to the input set of documents
input_dir = sys.argv[1]

# path to the pairwise similarity scores
output_file = sys.argv[2]

# path to the collection of documents
collection_dir = sys.argv[3]

# path to the df counts, saved as a gzipped csv file
df_file = sys.argv[4]

# load the DF counts
df_counts = load_document_frequency_file(input_file=df_file)

# stoplist for terms in document vectors
stoplist = list(string.punctuation)
stoplist += ['-lrb-', '-rrb-', '-lcb-', '-rcb-', '-lsb-', '-rsb-']
stoplist += stopwords.words('english')

# compute the pairwise similarity measures and write output
compute_pairwise_similarity_matrix(input_dir=input_dir,
                                   output_file=output_file,
                                   collection_dir=collection_dir,
                                   df=df_counts,
                                   extension="xml",
                                   language="en",
                                   normalization="stemming",
                                   stoplist=stoplist)








