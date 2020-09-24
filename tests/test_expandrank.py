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


def create_corpus(values, tmp_path, name='corpus.gz'):
    # from test_utils.py
    corpus_dir = tmp_path / name
    corpus_dir.mkdir()
    for k, v in values.items():
        (corpus_dir / k).write_text(v)
    return corpus_dir


def create_df(corpus_dir, tmp_path, name='corpus_df.gz'):
    # from test_utils.py
    corpus_df_file = tmp_path / name
    pke.utils.compute_document_frequency(
        str(corpus_dir), str(corpus_df_file), extension='txt', n=1)
    corpus_df = pke.utils.load_document_frequency_file(str(corpus_df_file))
    return corpus_df, corpus_df_file


def test_expandrank_candidate_selection():
    """Test ExtandRank candidate selection method."""

    extractor = pke.unsupervised.ExpandRank()
    extractor.load_document(input=input_file)
    extractor.candidate_selection(pos=pos)


def test_expandrank_candidate_weighting(tmp_path):
    """Test ExtandRank candidate weighting method."""

    # Create a corpus, compute the df, compute the pairwise_similarity_matric
    input_file = 'a.txt'
    corpus = {input_file: 'lorem sit amet', 'b.txt': 'lorem ipsum'}
    corpus_dir = create_corpus(corpus, tmp_path)
    # compute_df_count
    corpus_df, _ = create_df(corpus_dir, tmp_path)
    # compute_pairwise_similarity_matrix
    pairw_file = tmp_path / 'pairwise.gz'
    pke.utils.compute_pairwise_similarity_matrix(
        str(corpus_dir), str(pairw_file), extension='txt',
        collection_dir=None, df=corpus_df)
    pairwise = pke.utils.load_pairwise_similarities(str(pairw_file))

    n_expanded = 1
    expanded_documents = [(v, u) for u, v in pairwise[os.path.basename(input_file)][-n_expanded:]]

    extractor = pke.unsupervised.ExpandRank()
    extractor.load_document(input=input_file)
    extractor.candidate_selection(pos=pos)
    extractor.candidate_weighting(window=10, pos=pos, expanded_documents=expanded_documents, normalized=False)
    keyphrases = [k for k, s in extractor.get_n_best(n=10)]
