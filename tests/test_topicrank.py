#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pke

def test_candidate_selection():
    extractor = pke.unsupervised.TopicRank(input_file='examples/C-1.xml',
                                           language='english')
    extractor.read_document(format='corenlp',
                            use_lemmas=False)

    extractor.candidate_selection(pos=['NN', 'NNS', 'NNP', 'NNPS', 'JJ', 'JJR',
                                   'JJS'])

    assert len(extractor.candidates) == 567


def test_candidate_weighting():
    extractor = pke.unsupervised.TopicRank(input_file='examples/C-1.xml',
                                           language='english')
    extractor.read_document(format='corenlp',
                            use_lemmas=False)

    extractor.candidate_selection(pos=['NN', 'NNS', 'NNP', 'NNPS', 'JJ', 'JJR',
                                   'JJS'])

    extractor.candidate_weighting(threshold=0.74,
                                  method='average')

    keyphrases = [k for k, s in extractor.get_n_best(n=3)]

    assert keyphrases == ['registries', 'grid services', 'dht']