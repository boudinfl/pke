#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pke

valid_pos = {'NOUN', 'PROPN', 'ADJ'}
test_file = os.path.join('tests', 'data', '1939.xml')


def test_topicrank_candidate_selection():
    extractor = pke.unsupervised.TopicRank()
    extractor.load_document(test_file)
    extractor.candidate_selection(pos=valid_pos)
    assert len(extractor.candidates) == 20


def test_topicrank_candidate_weighting():
    extractor = pke.unsupervised.TopicRank()
    extractor.load_document(test_file)
    extractor.candidate_selection(pos=valid_pos)
    extractor.candidate_weighting(threshold=0.74, method='average')
    keyphrases = [k for k, s in extractor.get_n_best(n=3)]
    assert keyphrases == ['set', 'systems', 'solutions']


if __name__ == '__main__':
    test_topicrank_candidate_selection()
    test_topicrank_candidate_weighting()
