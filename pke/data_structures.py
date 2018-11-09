# -*- coding: utf-8 -*-

"""Data structures for the pke module."""


class Sentence(object):
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

    def __eq__(self, other):
        """Compares two sentences for equality."""

        # test whether they are instances of different classes
        if type(self) != type(other):
            return False

        # test whether they are of same length
        if self.length != other.length:
            return False

        # test whether they have the same words
        if self.words != other.words:
            return False

        # test whether they have the same PoS tags
        if self.pos != other.pos:
            return False

        # test whether they have the same stem forms
        if self.stems != other.stems:
            return False

        # test whether they have the same meta-information
        if self.meta != other.meta:
            return False

        # if everything is ok then they are equal
        return True


class Candidate(object):
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


class Document(object):
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
            input_file (str): path to the input file.
        """

        # initialize document
        doc = Document()

        # set the input file
        doc.input_file = kwargs.get('input_file', None)

        # loop through the parsed sentences
        for i, sentence in enumerate(sentences):

            # add the sentence to the container
            s = Sentence(words=sentence['words'])

            # add the POS
            s.pos = sentence['POS']

            # add the lemmas
            s.stems = sentence['lemmas']

            # add the meta-information
            for (k, infos) in sentence.items():
                if k not in {'POS', 'lemmas', 'words'}:
                    s.meta[k] = infos

            # add the sentence to the document
            doc.sentences.append(s)

        return doc

    def __eq__(self, other):
        """Compares two documents for equality."""

        # test whether they are instances of different classes
        if type(self) != type(other):
            return False

        # test whether they have the same language
        if self.language != other.language:
            return False

        # test whether they have the same input path
        if self.input_file != other.input_file:
            return False

        # test whether they contain the same lists of sentences
        if self.sentences != other.sentences:
            return False

        # if everything is ok then they are equal
        return True
