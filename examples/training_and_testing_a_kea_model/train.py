# -*- coding: utf-8 -*-

import logging

import pke

# setting info in terminal
logging.basicConfig(level=logging.INFO)

# path to the collection of documents
input_dir = 'train/'

# path to the reference file
reference_file = "gold-annotation.txt"

# path to the df file
df_file = "df.tsv.gz"
logging.info('Loading df counts from {}'.format(df_file))
df_counts = pke.load_document_frequency_file(input_file=df_file,
                                             delimiter='\t')

# path to the model, saved as a pickle
output_mdl = "model.pickle"

pke.train_supervised_model(input_dir=input_dir,
                           reference_file=reference_file,
                           model_file=output_mdl,
                           df=df_counts,
                           format="corenlp",
                           use_lemmas=False,
                           stemmer="porter",
                           model=pke.supervised.Kea(),
                           language='english',
                           extension="xml")
