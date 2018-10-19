# -*- coding: utf-8 -*-
# Author: ygor Gallina
# Date: 19-10-2018

"""StupidKE keyphrase extraction model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from pke.base import LoadFile
from pke.utils import load_document_frequency_file

from nltk.corpus import stopwords

import math
import string


class StupidKE(LoadFile):
    """StupidKE keyphrase extraction model.

    Parameterized example::

        import string
        import pke

        # 1. create a StupidKE extractor.
        extractor = pke.unsupervised.StupidKE(input_file='path/to/input.xml')

        # 2. load the content of the document.
        extractor.read_document(format='corenlp')

        # 3. select the the longest sequences of nouns and adjectives, that do
        #    not contain punctuation marks or stopwords as candidates.
        pos = set(['NN', 'NNS', 'NNP', 'NNPS', 'JJ', 'JJR', 'JJS'])
        stoplist = list(string.punctuation)
        stoplist += ['-lrb-', '-rrb-', '-lcb-', '-rcb-', '-lsb-', '-rsb-']
        stoplist += stopwords.words('english')
        extractor.candidate_selection(pos=pos, stoplist=stoplist)

        # 4. weight the candidates
        extractor.candidate_weighting(df=df)

        # 5. get the 10-highest scored candidates as keyphrases
        keyphrases = extractor.get_n_best(n=10)

    """

    def candidate_selection(self, pos=None, stoplist=None):
        """ The candidate selection as described in the TextRank paper.

            Args:
                pos (set): the set of valid POS tags, defaults to (NN, NNS,
                    NNP, NNPS, JJ, JJR, JJS).
                stoplist (list): the stoplist for filtering candidates, defaults
                    to the nltk stoplist. Words that are punctuation marks from
                    string.punctuation are not allowed.
        """

        # define default pos tags set
        if pos is None:
            pos = {'NN', 'NNS', 'NNP', 'NNPS', 'JJ', 'JJR', 'JJS'}

        # select sequence of adjectives and nouns
        self.longest_pos_sequence_selection(valid_pos=pos)

        # initialize stoplist list if not provided
        if stoplist is None:
            stoplist = stopwords.words(self.language)

        # filter candidates containing stopwords or punctuation marks
        self.candidate_filtering(
            stoplist=list(string.punctuation) +
            ['-lrb-', '-rrb-', '-lcb-', '-rcb-', '-lsb-', '-rsb-'] +
            stoplist)

    def candidate_weighting(self):
        """Candidate weighting function using position.
        """

        # Weigh the candidates
        for k in self.candidates.keys():
            # the '-' ensures that the first item will have the higher weight
            self.weights[k] = -min(self.candidates[k].offsets)
