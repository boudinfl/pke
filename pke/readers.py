#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Readers for the pke module."""

import xml.etree.ElementTree as etree
import spacy

from pke.data_structures import Document


class Reader(object):
    def read(self, path):
        raise NotImplementedError


class MinimalCoreNLPReader(Reader):
    """Minimal CoreNLP XML Parser."""

    def __init__(self):
        self.parser = etree.XMLParser()

    def read(self, path, **kwargs):
        sentences = []
        tree = etree.parse(path, self.parser)
        for sentence in tree.iterfind('./document/sentences/sentence'):
            # get the character offsets
            starts = [int(u.text) for u in
                      sentence.iterfind("tokens/token/CharacterOffsetBegin")]
            ends = [int(u.text) for u in
                    sentence.iterfind("tokens/token/CharacterOffsetEnd")]
            sentences.append({
                "words": [u.text for u in
                          sentence.iterfind("tokens/token/word")],
                "lemmas": [u.text for u in
                           sentence.iterfind("tokens/token/lemma")],
                "POS": [u.text for u in sentence.iterfind("tokens/token/POS")],
                "char_offsets": [(starts[k], ends[k]) for k in
                                 range(len(starts))]
            })
            sentences[-1].update(sentence.attrib)

        doc = Document.from_sentences(sentences, input_file=path, **kwargs)

        return doc


# FIX
def fix_spacy_for_french(nlp):
    """Fixes https://github.com/boudinfl/pke/issues/115.
    For some special tokenisation cases, spacy do not assign a `pos` field.

    Taken from https://github.com/explosion/spaCy/issues/5179.
    """
    from spacy.symbols import TAG
    if nlp.lang != 'fr':
        # Only fix french model
        return nlp
    if '' not in [t.pos_ for t in nlp('est-ce')]:
        # If the bug does not happen do nothing
        return nlp
    rules = nlp.Defaults.tokenizer_exceptions

    for orth, token_dicts in rules.items():
        for token_dict in token_dicts:
            if TAG in token_dict:
                del token_dict[TAG]
    try:
        nlp.tokenizer = nlp.Defaults.create_tokenizer(nlp)  # this property assignment flushes the cache
    except Exception as e:
        # There was a problem fallback on using `pos = token.pos_ or token.tag_`
        ()
    return nlp


class RawTextReader(Reader):
    """Reader for raw text."""

    def __init__(self, language=None):
        """Constructor for RawTextReader.

        Args:
            language (str): language of text to process.
        """

        self.language = language

        if language is None:
            self.language = 'en'

    def read(self, text, **kwargs):
        """Read the input file and use spacy to pre-process.

        Args:
            text (str): raw text to pre-process.
            max_length (int): maximum number of characters in a single text for
                spacy, default to 1,000,000 characters (1mb).
            spacy_model (model): an already loaded spacy model.
        """

        spacy_model = kwargs.get('spacy_model', None)

        if spacy_model is not None:
            spacy_model = fix_spacy_for_french(spacy_model)
            spacy_doc = spacy_model(text)
        else:
            max_length = kwargs.get('max_length', 10**6)
            nlp = spacy.load(self.language,
                            max_length=max_length)
            nlp = fix_spacy_for_french(nlp)
            spacy_doc = nlp(text)

        sentences = []
        for sentence_id, sentence in enumerate(spacy_doc.sents):
            sentences.append({
                "words": [token.text for token in sentence],
                "lemmas": [token.lemma_ for token in sentence],
                # FIX : This is a fallback if `fix_spacy_for_french` does not work
                "POS": [token.pos_ or token.tag_ for token in sentence],
                "char_offsets": [(token.idx, token.idx + len(token.text))
                                     for token in sentence]
            })

        doc = Document.from_sentences(sentences,
                                      input_file=kwargs.get('input_file', None),
                                      **kwargs)

        return doc

