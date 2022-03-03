# -*- coding: utf-8 -*-

from dataclasses import dataclass

"""Data structures for the pke module."""


@dataclass
class Sentence:
    """The sentence data structure."""

    def __init__(self, words, pos=[], meta={}):

        self.words = words
        """list of words (tokens) in the sentence."""

        self.pos = pos
        """list of Part-Of-Speeches."""

        self.stems = []
        """list of stems."""

        self.length = len(words)
        """length (number of tokens) of the sentence."""

        self.meta = meta
        """meta-information of the sentence."""


@dataclass
class Candidate:
    """The keyphrase candidate data structure."""

    def __init__(self):

        self.surface_forms = []
        """ the surface forms of the candidate. """

        self.offsets = []
        """ the offsets of the surface forms. """

        self.sentence_ids = []
        """ the sentence id of each surface form. """

        self.pos_patterns = []
        """ the Part-Of-Speech patterns of the candidate. """

        self.lexical_form = []
        """ the lexical form of the candidate. """
