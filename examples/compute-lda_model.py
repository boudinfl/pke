# -*- coding: utf-8 -*-

import logging
import sys

from pke import compute_lda_model

# setting info in terminal
logging.basicConfig(level=logging.INFO)

# path to the collection of documents
input_dir = sys.argv[1]

# path to the df weights dictionary, saved as a gzipped csv file
output_file = sys.argv[2]

# number of topics for the LDA model
n_topics = int(sys.argv[3])

# compute idf weights
compute_lda_model(input_dir=input_dir,
                  output_file=output_file,
                  n_topics=n_topics,
                  format="corenlp",  # input files format
                  extension='xml',  # input files extension
                  use_lemmas=False,  # do not use Stanford lemmas
                  stemmer="porter",  # use porter stemmer
                  language="english")  # language for the stop_words
