# -*- coding: utf-8 -*-

import sys
import logging
from glob import glob
import xml.etree.ElementTree as etree

from pke import compute_lda_model

# setting info in terminal
logging.basicConfig(level=logging.INFO)

# path to the collection of xml documents
input_dir = sys.argv[1]

# path to the lda model, saved as a gzip file
output_file = sys.argv[2]

# number of topics for the LDA model
n_topics = int(sys.argv[3])


def read_corenlp_xml(path):
    sentences = []
    tree = etree.parse(path, etree.XMLParser())
    for sentence in tree.iterfind('./document/sentences/sentence'):
        # get the character offsets
        starts = [int(u.text) for u in
                  sentence.iterfind('tokens/token/CharacterOffsetBegin')]
        ends = [int(u.text) for u in
                sentence.iterfind('tokens/token/CharacterOffsetEnd')]
        doc = {
            'words': [u.text for u in
                      sentence.iterfind('tokens/token/word')],
            'lemmas': [u.text for u in
                       sentence.iterfind('tokens/token/lemma')],
            'POS': [u.text for u in sentence.iterfind('tokens/token/POS')],
            'char_offsets': [(starts[k], ends[k]) for k in
                             range(len(starts))]
        }
        sentences.append(
            [(doc['words'][i], doc['POS'][i])
             for i in range(len(doc['words']))])
    return sentences


documents = []
for fn in glob(input_dir + '*.xml'):
    doc = read_corenlp_xml(fn)
    documents.append(doc)

compute_lda_model(
    documents,
    output_file=output_file,
    n_topics=n_topics,
    language='en',
    normalization='stemming'
)
