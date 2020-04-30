#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import unicode_literals

import os
import pke
from collections import defaultdict
import gzip
import bisect

input_file = os.path.join('test', 'data', 'FT923-5089.xml')
pos = {'NOUN', 'PROPN', 'ADJ'}


def test_expandrank_candidate_selection():
    """Test ExtandRank candidate selection method."""

    extractor = pke.unsupervised.ExpandRank()
    extractor.load_document(input=input_file)
    extractor.candidate_selection(pos=pos)


def test_expandrank_candidate_weighting(expanded_docs):
    """Test ExtandRank candidate weighting method."""

    extractor = pke.unsupervised.ExpandRank()
    extractor.load_document(input=input_file)
    extractor.candidate_selection(pos=pos)
    extractor.candidate_weighting(window=10, pos=pos, expanded_documents=expanded_docs, normalized=False)
    keyphrases = [k for k, s in extractor.get_n_best(n=10)]
    print(keyphrases)

def test_singlerank_candidate_weighting():
    """Test SingleRank candidate weighting method."""

    extractor = pke.unsupervised.SingleRank()
    extractor.load_document(input=input_file)
    extractor.candidate_selection(pos=pos)
    extractor.candidate_weighting(window=10, pos=pos)
    keyphrases = [k for k, s in extractor.get_n_best(n=10)]
    print(keyphrases)

def load_pairwise_similarities(path):
    """Load the pairwise similarities for ExpandRank."""

    pairwise_sim = defaultdict(list)
    with gzip.open(path, 'rb') as f:
        lines = f.readlines()
        for line in lines:
            cols = line.decode('utf-8').strip().split()
            bisect.insort(pairwise_sim[cols[0].split("/")[-1]], (float(cols[2]), cols[1].split("/")[-1]))
            bisect.insort(pairwise_sim[cols[1].split("/")[-1]], (float(cols[2]), cols[0].split("/")[-1]))
    return pairwise_sim


if __name__ == '__main__':
    """ You will first need to perform the function compute_document_frequency(input_dir=input_dir, ...)
    and compute the pairwise similarity measures compute_pairwise_similarity_matrix(input_dir=input_dir,...)
    in order to have the parwise similarity measures file.

    python3 compute-df-counts.py path/to/folderfile df_count.gz
    python3 compute-pairwise-similarity-matrix.py path/to/folderfile output.gz None df_count.gz

    """

    # hyperparameters neighbor numbers
    n_expanded = 1
    pairwise = load_pairwise_similarities(os.path.join('test', 'data', 'Output.gz'))
    test_expandrank_candidate_selection()
    expanded_documents = [(v, u) for u, v in pairwise[os.path.basename(input_file)][-n_expanded:]]
    test_expandrank_candidate_weighting(expanded_documents)
    test_singlerank_candidate_weighting()
