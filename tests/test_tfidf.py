#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import unicode_literals

import os
import pke

test_file = os.path.join('tests', 'data', '1939.xml')


def test_tfidf_candidate_selection():
    extractor = pke.unsupervised.TfIdf()
    extractor.load_document(test_file)
    extractor.candidate_selection()
    assert len(extractor.candidates) == 160


def test_tfidf_candidate_weighting():
    extractor = pke.unsupervised.TfIdf()
    extractor.load_document(test_file)
    extractor.candidate_selection()
    extractor.candidate_weighting()
    keyphrases = [k for k, s in extractor.get_n_best(n=3)]
    assert keyphrases == ['set of solutions', 'systems of linear',
                          'of solutions']


if __name__ == '__main__':
    test_tfidf_candidate_selection()
    test_tfidf_candidate_weighting()
