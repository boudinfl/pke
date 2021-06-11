#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Readers for the pke module."""

import os
import sys
import json
import logging
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


def list_linked_spacy_models():
    """ Read SPACY/data and return a list of link_name """
    if spacy.__version__ > "3":
        spacy_data = os.path.join(spacy.info(silent=True)['location'], 'data')
    else:
        spacy_data = os.path.join(spacy.info(silent=True)['Location'], 'data')
    linked = [d for d in os.listdir(spacy_data) if os.path.islink(
        os.path.join(spacy_data, d))]
    # linked = [os.path.join(spacy_data, d) for d in os.listdir(spacy_data)]
    # linked = {os.readlink(d): os.path.basename(d) for d in linked if os.path.islink(d)}
    return linked


def list_downloaded_spacy_models():
    """ Scan PYTHONPATH to find spacy models """
    models = []
    # For each directory in PYTHONPATH
    paths = [p for p in sys.path if os.path.isdir(p)]
    for site_package_dir in paths:
        # For each module
        modules = [os.path.join(site_package_dir, m) for m in os.listdir(site_package_dir)]
        modules = [m for m in modules if os.path.isdir(m)]
        for module_dir in modules:
            if 'meta.json' in os.listdir(module_dir):
                # Ensure the package we're in is a spacy model
                meta_path = os.path.join(module_dir, 'meta.json')
                with open(meta_path) as f:
                    meta = json.load(f)
                if meta.get('parent_package', '') == 'spacy':
                    models.append(module_dir)
    return models


def str2spacy(model):
    if int(spacy.__version__.split('.')[0]) < 3:
        downloaded_models = [os.path.basename(m) for m in list_downloaded_spacy_models()]
        links = list_linked_spacy_models()
    else:
        # As of spacy v3, links do not exist anymore and it is simpler to get a list of
        # downloaded models
        downloaded_models = list(spacy.info()['pipelines'])
        links = []
    filtered_downloaded = [m for m in downloaded_models if m[:2] == model]
    if model in downloaded_models + links:
        # Check whether `model` is the name of a model/link
        return model
    elif filtered_downloaded:
        # Check whether `model` is a lang code and corresponds to a downloaded model
        return filtered_downloaded[0]
    else:
        # Return asked model to have an informative error.
        return model


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

        Spacy model selection: By default this function will load the spacy
        model that is closest to the `language` parameter ('fr' language will
        load the spacy model linked to 'fr' or any 'fr_core_web_*' available
        model). In order to select the model that will be used please provide a
        preloaded model via the `spacy_model` parameter, or link the model you
        wish to use to the corresponding language code
        `python3 -m spacy link spacy_model lang_code`.

        Args:
            text (str): raw text to pre-process.
            max_length (int): maximum number of characters in a single text for
                spacy (for spacy<3 compatibility, as of spacy v3 long texts
                should be splitted in smaller portions), default to
                1,000,000 characters (1mb).
            spacy_model (model): an already loaded spacy model.
        """

        spacy_model = kwargs.get('spacy_model', None)

        if spacy_model is None:
            try:
                spacy_model = spacy.load(str2spacy(self.language),
                                         disable=['ner', 'textcat', 'parser'])
            except OSError:
                logging.warning('No spacy model for \'{}\' language.'.format(self.language))
                logging.warning('Falling back to using english model. There might '
                    'be tokenization and postagging errors. A list of available '
                    'spacy model is available at https://spacy.io/models.'.format(
                        self.language))
                spacy_model = spacy.load(str2spacy('en'),
                                         disable=['ner', 'textcat', 'parser'])
            if int(spacy.__version__.split('.')[0]) < 3:
                sentencizer = spacy_model.create_pipe('sentencizer')
            else:
                sentencizer = 'sentencizer'
            spacy_model.add_pipe(sentencizer)
            if 'max_length' in kwargs and kwargs['max_length']:
                spacy_model.max_length = kwargs['max_length']

        spacy_model = fix_spacy_for_french(spacy_model)
        spacy_doc = spacy_model(text)

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

        doc = Document.from_sentences(
            sentences, input_file=kwargs.get('input_file', None), **kwargs)

        return doc
