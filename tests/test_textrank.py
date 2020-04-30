#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import unicode_literals

import os
import pke

test_file = os.path.join('tests', 'data', '1939.xml')
pos = {'NOUN', 'PROPN', 'ADJ'}


def test_textrank():
  """Test TextRank for keyword extraction using original paper's example."""

  extractor = pke.unsupervised.TextRank()
  extractor.load_document(input=test_file)
  extractor.candidate_weighting(top_percent=.33, pos=pos)
  keyphrases = [k for k, s in extractor.get_n_best(n=3)]
  assert keyphrases == ['linear diophantine',
                        'upper bounds',
                        'inequations']


def test_textrank_with_candidate_selection():
  """Test TextRank with longest-POS-sequences candidate selection."""

  extractor = pke.unsupervised.TextRank()
  extractor.load_document(input=test_file)
  extractor.candidate_selection(pos=pos)
  extractor.candidate_weighting(pos=pos)
  keyphrases = [k
                for k, s in extractor.get_n_best(n=3)]
  assert keyphrases == ['linear diophantine equations',
                        'minimal supporting set',
                        'nonstrict inequations']


if __name__ == '__main__':
  test_textrank()
  test_textrank_with_candidate_selection()
