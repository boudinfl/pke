# -*- coding: utf-8 -*-
# Author: ygor Gallina
# Date: 19-10-2018

"""StupidKE keyphrase extraction model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from pke.base import LoadFile


class FirstPhrases(LoadFile):
    """Baseline model that extracts the first phrases of a document.

    Parameterized example::

        import pke

        # define the set of valid Part-of-Speeches
        pos = {'NOUN', 'PROPN', 'ADJ'}

        # 1. create a FirstPhrases baseline extractor.
        extractor = pke.unsupervised.FirstPhrases()

        # 2. load the content of the document.
        extractor.load_document(input='path/to/input',
                                language='en',
                                normalization=None)

        # 3. select the longest sequences of nouns and adjectives as candidates.
        extractor.candidate_selection(pos=pos)

        # 4. weight the candidates using their position
        extractor.candidate_weighting()

        # 5. get the 10-highest scored candidates as keyphrases
        keyphrases = extractor.get_n_best(n=10)

    """

    def candidate_selection(self, pos=None):
        """Candidate selection using longest sequences of PoS.

        Args:
            pos (set): set of valid POS tags, defaults to ('NOUN', 'PROPN',
                'ADJ').
        """

        if pos is None:
            pos = {'NOUN', 'PROPN', 'ADJ'}

        # select sequence of adjectives and nouns
        self.longest_pos_sequence_selection(valid_pos=pos)

    def candidate_weighting(self):
        """Candidate weighting function using position."""

        # rank candidates using inverse position
        for k in self.candidates.keys():
            # the '-' ensures that the first item will have the higher weight
            self.weights[k] = -min(self.candidates[k].offsets)
