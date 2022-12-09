# -*- coding: utf-8 -*-

import os
import logging
from glob import glob

import pke

# setting info in terminal
logging.basicConfig(level=logging.INFO)
base = os.path.dirname(__file__)

# path to the collection of documents
documents = []
for fn in glob(base + os.sep + 'train/*.txt'):
    with open(fn) as f:
        doc = f.read()
    doc_id = os.path.basename(fn).rsplit('.', 1)[0]
    documents.append((doc_id, doc))

logging.info('Loaded {} documents'.format(len(documents)))

# path to the reference file
reference = {}
with open(base + os.sep + 'gold-annotation.txt') as f:
    for line in f:
        doc_id, keywords = line.split(' : ')
        reference[doc_id] = keywords.split(',')

# path to the df file
df_file = base + os.sep + 'df.tsv.gz'
logging.info('Loading df counts from {}'.format(df_file))
df_counts = pke.load_document_frequency_file(
    input_file=df_file, delimiter='\t'
)

# path to the model, saved as a pickle
output_mdl = base + os.sep + 'model.pickle'
pke.train_supervised_model(
    documents,
    reference,
    model_file=output_mdl,
    language='en',
    normalization='stemming',
    df=df_counts,
    model=pke.supervised.Kea()
)
