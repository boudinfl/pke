#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Readers for the pke module. """

from lxml import etree

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