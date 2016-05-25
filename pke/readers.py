#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Readers for the pke module. """

import codecs
from lxml import etree
from nltk.tag import str2tuple

class MinimalCoreNLPParser:
    """ Minimal CoreNLP XML Parser in Python. """
    def __init__(self, path):
        self.sentences = []
        parser = etree.XMLParser()
        tree = etree.parse(path, parser)
        for sentence in tree.iterfind('./document/sentences/sentence'):
            self.sentences.append({
              "words" : [u.text for u in sentence.iterfind("tokens/token/word")],
              "lemmas" : [u.text for u in sentence.iterfind("tokens/token/lemma")],
              "POS" : [u.text for u in sentence.iterfind("tokens/token/POS")]
            })


class PreProcessedTextReader:
    """ Reader for pre-processed text. """
    def __init__(self, path, sep=u'/'):
        self.sentences = []
        for line in codecs.open(path, 'r', 'utf-8'):
            tuples = line.strip().split()
            self.sentences.append({
                "words" : [str2tuple(u, sep=u'/')[0] for u in tuples],
                "POS" : [str2tuple(u, sep=u'/')[1] for u in tuples]
              })