#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import unicode_literals

import os
import pke

test_file = os.path.join('tests', 'data', '1939.xml')
pos = {'NOUN', 'PROPN', 'ADJ'}


def test_singlerank_candidate_selection():
    """Test SingleRank candidate selection method."""

    extractor = pke.unsupervised.SingleRank()
    extractor.load_document(input=test_file)
    extractor.candidate_selection(pos=pos)
    assert len(extractor.candidates) == 20


def test_singlerank_candidate_weighting():
    """Test SingleRank candidate weighting method."""

    extractor = pke.unsupervised.SingleRank()
    extractor.load_document(input=test_file)
    extractor.candidate_selection(pos=pos)
    extractor.candidate_weighting(window=10, pos=pos)
    keyphrases = [k for k, s in extractor.get_n_best(n=3)]
    assert keyphrases == ['minimal supporting set',
                          'minimal set',
                          'linear diophantine equations']


if __name__ == '__main__':
    test_singlerank_candidate_selection()
    test_singlerank_candidate_weighting()
