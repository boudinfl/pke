#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import unicode_literals

import os
import pke

test_file = os.path.join('tests', 'data', '1939.xml')
grammar = "NP: {<ADJ>*<NOUN|PROPN>+}"
pos = {'NOUN', 'PROPN', 'ADJ'}


def test_topicalpagerank_candidate_selection():
    """Test Single Topical PageRank candidate selection method."""

    extractor = pke.unsupervised.TopicalPageRank()
    extractor.load_document(input=test_file)
    extractor.candidate_selection(grammar=grammar)
    assert len(extractor.candidates) == 19


def test_topicalpagerank_candidate_weighting():
    """Test Single Topical PageRank weighting method."""

    extractor = pke.unsupervised.TopicalPageRank()
    extractor.load_document(input=test_file)
    extractor.candidate_selection(grammar=grammar)
    extractor.candidate_weighting(window=10, pos=pos)
    keyphrases = [k for k, s in extractor.get_n_best(n=3)]
    assert keyphrases == ['minimal supporting set',
                          'minimal set',

                          'linear diophantine equations']


if __name__ == '__main__':
    test_topicalpagerank_candidate_selection()
    test_topicalpagerank_candidate_weighting()