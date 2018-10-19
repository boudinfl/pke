#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pke

valid_pos = {'NN', 'NNS', 'NNP', 'NNPS', 'JJ', 'JJR', 'JJS'}


def test_candidate_selection():
    extractor = pke.unsupervised.StupidKE(input_file='examples/C-1.xml',
                                          language='english')
    extractor.read_document(format='corenlp',
                            use_lemmas=False)

    extractor.candidate_selection(pos=valid_pos)

    assert len(extractor.candidates) == 567


def test_candidate_weighting():
    extractor = pke.unsupervised.StupidKE(input_file='examples/C-1.xml',
                                           language='english')
    extractor.read_document(format='corenlp',
                            use_lemmas=False)

    extractor.candidate_selection(pos=valid_pos)

    extractor.candidate_weighting()

    keyphrases = [k for k, s in extractor.get_n_best(n=3)]
    assert keyphrases == ['scalable grid service discovery', 'uddi', 'abstract']
