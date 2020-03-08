#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import unicode_literals

import pke

input_file = 'data_cluster/FT923-5089.xml'
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


if __name__ == '__main__':
    test_expandrank_candidate_selection()
    expanded_docs = []
    with open("data_cluster/test_cluster.txt", "r") as file:
        line = file.readline().split(",")
        if(line[0] == input_file):
            expanded_docs = line[1]
    test_expandrank_candidate_weighting(expanded_docs)

    test_singlerank_candidate_weighting()
