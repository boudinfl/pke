#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pke

valid_pos = {'NN', 'NNS', 'NNP', 'NNPS', 'JJ', 'JJR', 'JJS'}
test_file = 'examples/C-1.xml'


def test_stupidke_candidate_selection():
    extractor = pke.unsupervised.StupidKE()
    extractor.load_document(test_file)

    extractor.candidate_selection(pos=valid_pos)

    assert len(extractor.candidates) == 567


def test_stupidke_candidate_weighting():
    extractor = pke.unsupervised.StupidKE()
    extractor.load_document(test_file)

    extractor.candidate_selection(pos=valid_pos)

    extractor.candidate_weighting()

    keyphrases = [k for k, s in extractor.get_n_best(n=3)]

    assert keyphrases == ['scalable grid service discovery', 'uddi', 'abstract']


if __name__ == '__main__':
    test_stupidke_candidate_selection()
    test_stupidke_candidate_weighting()
