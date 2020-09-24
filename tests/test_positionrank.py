#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import unicode_literals

import os
import pke

test_file = os.path.join('tests', 'data', '1939.xml')

grammar = "NP: {<ADJ>*<NOUN|PROPN>+}"
pos = {'NOUN', 'PROPN', 'ADJ'}


def test_positionrank_candidate_selection():
    """Test PositionRank candidate selection method."""

    extractor = pke.unsupervised.PositionRank()
    extractor.load_document(input=test_file)
    extractor.candidate_selection(grammar=grammar)
    assert len(extractor.candidates) == 19


def test_positionrank_candidate_weighting():
    """Test PositionRank candidate weighting method."""

    extractor = pke.unsupervised.PositionRank()
    extractor.load_document(input=test_file)
    extractor.candidate_selection(grammar=grammar)
    extractor.candidate_weighting(window=10, pos=pos)
    keyphrases = [k for k, s in extractor.get_n_best(n=3)]
    assert keyphrases == ['minimal supporting set',
                          'linear diophantine equations',
                          'minimal set']


if __name__ == '__main__':
    test_positionrank_candidate_selection()
    test_positionrank_candidate_weighting()