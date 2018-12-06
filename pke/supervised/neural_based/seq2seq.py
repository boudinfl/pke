# -*- coding: utf-8 -*-
# Author: Florian Boudin
# Date: 11-11-2018

"""
Implementation of the Seq2Seq model for automatic keyphrase extraction.
"""

from __future__ import absolute_import
from __future__ import print_function

from pke.supervised.api import SupervisedLoadFile


class Seq2Seq(SupervisedLoadFile):

    def __init__(self):
        """Redefining initializer for Seq2Seq."""

        super(Seq2Seq, self).__init__()

        self.sequence = []
        """Input sequence."""

        self.vocabulary = ['<SOS>', '<EOS>', '<UNK>']
        """Vocabulary."""

    def document_to_ix(self):
        """Convert the document to a sequence of ix."""

        self.sequence.append(self.vocabulary.index('<SOS>'))
        for i, sentence in enumerate(self.sentences):
            for word in sentence.stems:
                try:
                    self.sequence.append(self.vocabulary.index(word))
                except ValueError:
                    self.sequence.append(self.vocabulary.index('<UNK>'))
        self.sequence.append(self.vocabulary.index('<EOS>'))

    def candidate_selection(self):
        pass

    def candidate_weighting(self):
        pass
