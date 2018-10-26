#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Readers for the pke module. """

import codecs
import spacy
import xml.etree.ElementTree as etree
from nltk.tag import str2tuple, pos_tag_sents
from nltk.tokenize import sent_tokenize, word_tokenize

class MinimalCoreNLPParser(object):
    """ Minimal CoreNLP XML Parser in Python. """
    def __init__(self, path):
        self.sentences = []
        parser = etree.XMLParser()
        tree = etree.parse(path, parser)
        for sentence in tree.iterfind('./document/sentences/sentence'):
            # get the character offsets
            starts = [int(u.text) for u in sentence.iterfind("tokens/token/CharacterOffsetBegin")]
            ends = [int(u.text) for u in sentence.iterfind("tokens/token/CharacterOffsetEnd")]
            self.sentences.append({
                "words" : [u.text for u in sentence.iterfind("tokens/token/word")],
                "lemmas" : [u.text for u in sentence.iterfind("tokens/token/lemma")],
                "POS" : [u.text for u in sentence.iterfind("tokens/token/POS")],
                "char_offsets" : [(starts[k], ends[k]) for k in range(len(starts))]
            })
            self.sentences[-1].update(sentence.attrib)


class PreProcessedTextReader(object):
    """ Reader for pre-processed text. """
    def __init__(self, path, sep=u'/'):
        self.sentences = []
        for line in codecs.open(path, 'r', 'utf-8'):
            tuples = line.strip().split()
            self.sentences.append({
                "words" : [str2tuple(u, sep=sep)[0] for u in tuples],
                "POS" : [str2tuple(u, sep=sep)[1] for u in tuples]
            })


class RawTextReader(object):
    """ Reader for raw text. """
    def __init__(self, path=None, input_text=None):
        self.sentences = []
        raw_text = input_text
        if path is not None:
            with codecs.open(path, 'r', 'utf-8') as f:
                raw_text = f.read()
        sentences = [word_tokenize(s) for s in sent_tokenize(raw_text)]
        tuples = pos_tag_sents(sentences)
        for sentence in tuples:
            self.sentences.append({
                "words" : [u[0] for u in sentence],
                "POS" : [u[1] for u in sentence]
            })

class SpacyDocReader(object):
    """ Reader from a spacy doc object. """
    def __init__(self, doc):
        self.sentences = []
        for sentence_id, sentence in enumerate(doc.sents):
            self.sentences.append({
                "words" : [token.text for token in sentence],
                "lemmas" : [token.lemma_ for token in sentence],
                "POS" : [token.pos_ for token in sentence],
                "char_offsets" : [(token.idx, token.idx+len(token.text)) for token in sentence]
            })