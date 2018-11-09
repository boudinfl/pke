# -*- coding: utf-8 -*-

import logging
import sys
from string import punctuation

from pke import compute_document_frequency

# setting info in terminal
logging.basicConfig(level=logging.INFO)

# path to the collection of documents
input_dir = sys.argv[1]

# path to the df weights dictionary, saved as a gzipped csv file
output_file = sys.argv[2]

# stoplist are punctuation marks
stoplist = list(punctuation)
stoplist += ['-lrb-', '-rrb-', '-lcb-', '-rcb-', '-lsb-', '-rsb-']

# compute idf weights
compute_document_frequency(input_dir=input_dir,
                           output_file=output_file,
                           extension='xml', # input file extension
                           language='en', # language of the input files
                           normalization="stemming", # use porter stemmer
                           stoplist=stoplist,  # stoplist
                           delimiter='\t',  # tab separated output
                           n=5)  # compute n-grams up to 5-grams