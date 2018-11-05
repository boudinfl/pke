#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import unicode_literals

import pke

text = u"Compatibility of systems of linear constraints over the set of natural\
 numbers. Criteria of compatibility of a system of linear Diophantine equations\
, strict inequations, and nonstrict inequations are considered. Upper bounds fo\
r components of a minimal set of solutions and algorithms of construction of mi\
nimal generating sets of solutions for all types of systems are given. These cr\
iteria and the corresponding algorithms for constructing a minimal supporting s\
et of solutions can be used in solving all the considered types systems and sys\
tems of mixed types."

pos = {'NOUN', 'PROPN', 'ADJ'}

def test_textrank():
    """Test TextRank for keyword extraction using original paper's example."""

    extractor = pke.unsupervised.TextRank()
    extractor.load_document(input=text)
    extractor.candidate_weighting(T=.33, pos=pos)
    keyphrases = [k for k, s in extractor.get_n_best(n=3)]
    assert keyphrases == ['linear diophantine',
                          'natural numbers',
                          'types']

def test_textrank_with_candidate_selection():
    """Test TextRank with longest-POS-sequences candidate selection."""

    extractor = pke.unsupervised.TextRank()
    extractor.load_document(input=text)
    extractor.candidate_selection(pos=pos)
    extractor.candidate_weighting(pos=pos)
    keyphrases = [k for k, s in extractor.get_n_best(n=3)]
    assert keyphrases == ['linear diophantine equations',
                          'minimal generating sets',
                          'mixed types']


if __name__ == '__main__':
    test_textrank()
    test_textrank_with_candidate_selection()
