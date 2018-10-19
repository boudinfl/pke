#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pke

valid_pos = {'NN', 'NNS', 'NNP', 'NNPS', 'JJ', 'JJR', 'JJS'}


def test_candidate_selection():
    extractor = pke.unsupervised.TextRank(input_file='examples/C-1.xml',
                                          language='english')
    extractor.read_document(format='corenlp',
                            use_lemmas=False)

    extractor.candidate_selection(pos=valid_pos)

    assert len(extractor.candidates) == 567


def test_candidate_weighting():
    extractor = pke.unsupervised.TextRank(input_file='examples/C-1.xml',
                                          language='english')
    extractor.read_document(format='corenlp',
                            use_lemmas=False)

    extractor.candidate_selection(pos=valid_pos)

    extractor.candidate_weighting()

    keyphrases = [k for k, s in extractor.get_n_best(n=3)]

    assert keyphrases == ['dht based uddi registry hierarchies',
                          'multiple proxy uddi registries',
                          'new local uddi registry figure']


if __name__ == '__main__':
    test_candidate_selection()
    test_candidate_weighting()
