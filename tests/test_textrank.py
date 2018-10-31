#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pke

valid_pos = {'NOUN', 'PROPN', 'ADJ'}
test_file = 'examples/C-1.xml'


def test_textrank_candidate_selection():
    extractor = pke.unsupervised.TextRank()
    extractor.load_document(test_file)

    extractor.candidate_selection(pos=valid_pos)

    assert len(extractor.candidates) == 567


def test_textrank_candidate_weighting():
    extractor = pke.unsupervised.TextRank()
    extractor.load_document(test_file)

    extractor.candidate_selection(pos=valid_pos)

    extractor.candidate_weighting()

    keyphrases = [k for k, s in extractor.get_n_best(n=3)]

    assert keyphrases == ['dht based uddi registry hierarchies',
                          'multiple proxy uddi registries',
                          'new local uddi registry figure']


if __name__ == '__main__':
    test_textrank_candidate_selection()
    test_textrank_candidate_weighting()
