# -*- coding: utf-8 -*-

import os
import sys
import codecs
import logging
import pke

# setting info in terminal
logging.basicConfig(level=logging.INFO)

# path to the collection of documents
input_dir = sys.argv[1]

# path to the reference file
reference_file = sys.argv[2]

# path to the df file
df_file = sys.argv[3]
logging.info('loading df counts from '+df_file)
df_counts = pke.load_document_frequency_file(df_file, delimiter='\t')

# path to the model, saved as a pickle
output_mdl = sys.argv[4]

pke.train_supervised_model(input_dir=input_dir,
                           reference_file=reference_file,
                           model_file=output_mdl,
                           df=df_counts,
                           format="corenlp",
                           use_lemmas=False,
                           stemmer="porter",
                           model=pke.supervised.Kea()
                           language='english',
                           extension="xml")