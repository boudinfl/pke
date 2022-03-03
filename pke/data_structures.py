# -*- coding: utf-8 -*-

from dataclasses import dataclass

"""Data structures for the pke module."""


@dataclass
class Sentence:
    """The sentence data structure."""

    def __init__(self, words):

        self.words = words
        """list of words (tokens) in the sentence."""

        self.pos = []
        """list of Part-Of-Speeches."""

        self.stems = []
        """list of stems."""

        self.length = len(words)
        """length (number of tokens) of the sentence."""

        self.meta = {}
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


@dataclass
class Document:
    """The Document data structure."""

    def __init__(self):

        self.input_file = None
        """ The path of the input file. """

        self.sentences = []
        """ The sentence container (list of Sentence). """

    @staticmethod
    def from_sentences(sentences, **kwargs):
        """Populate the sentence list.

        Args:
            sentences (Sentence list): content to create the document.
        """

        # initialize document
        doc = Document()

        # loop through the parsed sentences
        for i, sentence in enumerate(sentences):

            # add the sentence to the container
            s = Sentence(words=sentence['words'])

            # add the POS
            s.pos = sentence['POS']

            # add the lemmas
            s.stems = sentence['lemmas']

            # add the meta-information
            for (k, info) in sentence.items():
                if k not in {'POS', 'lemmas', 'words'}:
                    s.meta[k] = info

            # add the sentence to the document
            doc.sentences.append(s)

        return doc
