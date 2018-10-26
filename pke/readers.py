#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Readers for the pke module. """

import xml.etree.ElementTree as etree
from nltk.tag import str2tuple
from collections import defaultdict
from nltk.stem.snowball import SnowballStemmer as Stemmer
import logging
import spacy


class SpacyHelper:
    lang = None
    nlp = None

    @classmethod
    def _set_lang(cls, lang):
        if cls.lang == lang:
            pass
        cls.lang = lang
        cls.nlp = spacy.load(cls.lang)

    @classmethod
    def text2doc(cls, text, lang=None):
        if lang is None:
            lang = 'en'

        cls._set_lang(lang)
        return cls.nlp(text)


class Sentence(object):
    """ The sentence data structure. """

    def __init__(self, words):

        self.words = words
        """ tokens as a list. """

        self.pos = []
        """ Part-Of-Speeches as a list. """

        self.stems = []
        """ stems as a list. """

        self.length = len(words)
        """ length of the sentence. """

        self.meta = {}
        """ meta-information of the sentence. """

    def __eq__(self, other):
        if type(self) != type(other):
            return False
        if self.length != other.length:
            return False
        if self.words != other.words:
            return False
        if self.pos != other.pos:
            return False
        if self.stems != other.stems:
            return False
        if self.meta != other.meta:
            return False
        return True


class Candidate(object):
    """ The keyphrase candidate data structure. """

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
    """ The LoadFile class that provides base functions. """

    def __init__(self):
        """ Initializer for Document class.
        """

        self.input_file = None
        """ The path of the input file. """

        self.language = None
        """ The language of the input file. """

        self.sentences = []
        """ The sentence container (list of Sentence). """

        self.candidates = defaultdict(Candidate)
        """ The candidate container (dict of Candidate). """

        self.weights = {}
        """ The weight container (can be either word or candidate weights). """

    @staticmethod
    def from_corenlp_sentences(
            corenlp_sentences, **kwargs):
        """ Populate the sentence list.

            Args:
                corenlp_sentences (Sentence list): content to create the document.
                lang (str): language of the text.
                input_file (str): path to file from which read_content was extracted
                use_lemmas (bool): whether lemmas are used (if available)
                    instead of stems (computed by nltk), defaults to False.
                stemmer (str): the stemmer in nltk to use (if used), defaults
                    to porter (can be set to None for using word surface forms
                    instead of stems).
        """

        stemmer = kwargs.get('stemmer', 'porter')
        input_file = kwargs.get('input_file', None)
        lang = kwargs.get('lang', 'en')

        doc = Document()
        doc.input_file = input_file
        doc.language = lang

        # loop through the parsed sentences
        for i, sentence in enumerate(corenlp_sentences):

            # add the sentence to the container
            s = Sentence(words=sentence['words'])
            # add the POS
            s.pos = sentence['POS']

            if kwargs.get('use_lemmas', False):
                # add the lemmas
                try:
                    s.stems = sentence['lemmas']
                except KeyError:
                    logging.error('Lemmas are not available in the chosen input format')
            else:
                if stemmer is not None:
                    # add the stems
                    stem = Stemmer(stemmer).stem
                    s.stems = [stem(word) for word in s.words]
                else:
                    # otherwise computations are performed on surface forms
                    s.stems = s.words

            # lowercase the stems/lemmas
            s.stems = list(map(str.lower, s.stems))

            # add the meta-information
            # for k, infos in sentence.iteritems(): -- Python 2/3 compatible
            for (k, infos) in sentence.items():
                if k not in {'POS', 'lemmas', 'words'}:
                    s.meta[k] = infos

            doc.sentences.append(s)

        return doc

    @staticmethod
    def from_raw_text(raw_text, **kwargs):
        """
            Pre-process text and populate the sentence list.
            Args:
                raw_text (str): content to create the document.
                lang (str): language of the text.
                input_file (str): path to file from which read_content was extracted
                use_lemmas (bool): whether lemmas are used (if available)
                    instead of stems (computed by nltk), defaults to False.
                stemmer (str): the stemmer in nltk to use (if used), defaults
                    to porter (can be set to None for using word surface forms
                    instead of stems).
        """
        lang = kwargs.get('lang', 'en')
        sp_doc = SpacyHelper.text2doc(raw_text, lang=lang)
        sentences = []
        for sentence_id, sentence in enumerate(sp_doc.sents):
            sentences.append({
                "words": [token.text for token in sentence],
                "lemmas": [token.lemma_ for token in sentence],
                "POS": [token.pos_ for token in sentence],
                "char_offsets": [(token.idx, token.idx + len(token.text)) for token in sentence]
            })

        doc = Document.from_corenlp_sentences(sentences, **kwargs)
        return doc

    @staticmethod
    def from_readable(stream, **kwargs):
        """
            Read stream, pre-process text and populate the sentence list.
            Args:
                stream (readable): content to create the document.
                lang (str): language of the text.
                input_file (str): path to file from which read_content was extracted
                use_lemmas (bool): whether lemmas are used (if available)
                    instead of stems (computed by nltk), defaults to False.
                stemmer (str): the stemmer in nltk to use (if used), defaults
                    to porter (can be set to None for using word surface forms
                    instead of stems).
        """
        doc = Document.from_raw_text(stream.read(), **kwargs)
        return doc

    def __eq__(self, other):
        if type(self) != type(other):
            return False
        if self.language != other.language:
            return False
        if self.input_file != other.input_file:
            return False
        if self.sentences != other.sentences:
            return False
        if self.weights != other.weights:
            return False
        if self.candidates != other.candidates:
            return False
        return True


class Reader(object):
    def read(self, path):
        raise NotImplementedError


class MinimalCoreNLPReader(Reader):
    """ Minimal CoreNLP XML Parser in Python. """

    def __init__(self):
        self.parser = etree.XMLParser()

    def read(self, path, **kwargs):
        sentences = []

        tree = etree.parse(path, self.parser)
        for sentence in tree.iterfind('./document/sentences/sentence'):
            # get the character offsets
            starts = [int(u.text) for u in sentence.iterfind("tokens/token/CharacterOffsetBegin")]
            ends = [int(u.text) for u in sentence.iterfind("tokens/token/CharacterOffsetEnd")]
            sentences.append({
                "words": [u.text for u in sentence.iterfind("tokens/token/word")],
                "lemmas": [u.text for u in sentence.iterfind("tokens/token/lemma")],
                "POS": [u.text for u in sentence.iterfind("tokens/token/POS")],
                "char_offsets": [(starts[k], ends[k]) for k in range(len(starts))]
            })
            sentences[-1].update(sentence.attrib)

        doc = Document.from_corenlp_sentences(sentences, input_file=path, **kwargs)
        return doc


class PreProcessedTextReader(Reader):
    """ Reader for pre-processed text. """
    def __init__(self):
        pass

    def read(self, path, sep=None, encoding=None, **kwargs):
        if sep is None:
            sep = '/'
        if encoding is None:
            encoding = 'utf-8'

        sentences = []

        with open(path, 'r', encoding=encoding) as file:
            for line in file:
                tuples = line.strip().split()
                sentences.append({
                    "words": [str2tuple(u, sep=sep)[0] for u in tuples],
                    "POS": [str2tuple(u, sep=sep)[1] for u in tuples]
                })

        doc = Document.from_corenlp_sentences(sentences, input_file=path, **kwargs)
        return doc


class RawTextReader(Reader):
    """ Reader for raw text. """
    def __init__(self, lang=None):
        """ Constructor for RawTextReader.

            Args:
                lang: language of text to process.
        """
        if lang is None:
            lang = 'en'
        self.lang = lang

    def read(self, path, encoding=None):
        """ Read the input file or use input_text and use spacy to pre_treat.

            Args:
                path (str): file to read and pre-treat, defaults to None.
                encoding (str): file at path encoding.
        """
        if encoding is None:
            encoding = 'utf-8'

        with open(path, 'r', encoding=encoding) as file:
            text = file.read()

        doc = Document.from_raw_text(raw_text=text, input_file=path)
        return doc
