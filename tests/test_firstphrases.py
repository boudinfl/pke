# -*- coding: utf-8 -*-

import pke

from .sample import sample_list
valid_pos = {'NOUN', 'PROPN', 'ADJ'}


def test_firstphrases_candidate_selection():
    extractor = pke.unsupervised.FirstPhrases()
    extractor.load_document(input=sample_list)
    extractor.candidate_selection(pos=valid_pos)
    assert len(extractor.candidates) == 12


def test_firstphrases_candidate_weighting():
    extractor = pke.unsupervised.FirstPhrases()
    extractor.load_document(input=sample_list)
    extractor.candidate_selection(pos=valid_pos)
    extractor.candidate_weighting()
    keyphrases = [k for k, s in extractor.get_n_best(n=3)]
    assert keyphrases == ['inverse problems', 'mathematical model', 'ion exchange']


if __name__ == '__main__':
    test_firstphrases_candidate_selection()
    test_firstphrases_candidate_weighting()
