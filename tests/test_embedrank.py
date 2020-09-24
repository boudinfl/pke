#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import unicode_literals

import os
import sys

import mock
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


def test_embedrank_candidate_weighting():
    """Test SingleRank candidate weighting method."""
    extractor = pke.unsupervised.EmbedRank(
        embedding_path=os.path.join('tests', 'data', 'inspec_sent2vec.bin'))
    extractor.load_document(input=text)
    extractor.candidate_selection()
    extractor.candidate_weighting(l=1)
    keyphrases = [k for k, s in extractor.get_n_best(n=3)]
    assert keyphrases == ['systems', 'types systems', 'algorithms']


def test_import_embedrank_nosent2vec():
    # Without sent2vec this should not thorw an error

    # Make sent2vec unavailable
    with mock.patch.dict(sys.modules, {'sent2vec': None}):
        pke.unsupervised.EmbedRank()
