#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Readers for the pke module."""

import logging
import spacy

from pke.data_structures import Sentence

# https://spacy.io/usage/linguistic-features#native-tokenizer-additions
from spacy.lang.char_classes import ALPHA, ALPHA_LOWER, ALPHA_UPPER
from spacy.lang.char_classes import CONCAT_QUOTES, LIST_ELLIPSES, LIST_ICONS
from spacy.util import compile_infix_regex

# Modify tokenizer infix patterns
infixes = (
    LIST_ELLIPSES
    + LIST_ICONS
    + [
        r"(?<=[0-9])[+\-\*^](?=[0-9-])",
        r"(?<=[{al}{q}])\.(?=[{au}{q}])".format(
            al=ALPHA_LOWER, au=ALPHA_UPPER, q=CONCAT_QUOTES
        ),
        r"(?<=[{a}]),(?=[{a}])".format(a=ALPHA),
        # ✅ Commented out regex that splits on hyphens between letters:
        # r"(?<=[{a}])(?:{h})(?=[{a}])".format(a=ALPHA, h=HYPHENS),
        r"(?<=[{a}0-9])[:<>=/](?=[{a}])".format(a=ALPHA),
    ]
)

infix_re = compile_infix_regex(infixes)


class Reader(object):
    """Reader default class."""

    def read(self, path):
        raise NotImplementedError


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

    def read(self, text, spacy_model=None):
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
            spacy_model (model): an already loaded spacy model.
        """

        nlp = spacy_model

        if nlp is None:

            # list installed models
            installed_models = [m for m in spacy.util.get_installed_models() if m[:2] == self.language]

            # select first model for the language
            if len(installed_models):
                nlp = spacy.load(installed_models[0], disable=['ner', 'textcat', 'parser'])

            # stop execution is no model is available
            else:
                logging.error('No spacy model for \'{}\' language.'.format(self.language))
                logging.error('A list of available spacy models is available at https://spacy.io/models.')
                return

            # add the sentence splitter
            nlp.add_pipe('sentencizer')

        # Fix for non splitting words with hyphens with spacy taken from
        # https://spacy.io/usage/linguistic-features#native-tokenizer-additions
        nlp.tokenizer.infix_finditer = infix_re.finditer

        # process the document
        spacy_doc = nlp(text)

        sentences = []
        for sentence_id, sentence in enumerate(spacy_doc.sents):
            sentences.append(Sentence(
                words=[token.text for token in sentence],
                pos=[token.pos_ or token.tag_ for token in sentence],
                meta={
                    "lemmas": [token.lemma_ for token in sentence],
                    "char_offsets": [(token.idx, token.idx + len(token.text))
                                     for token in sentence]
                }
            ))
        return sentences


class SpacyDocReader(Reader):
    """Minimal Spacy Doc Reader."""

    def read(self, spacy_doc):
        sentences = []
        for sentence_id, sentence in enumerate(spacy_doc.sents):
            sentences.append(Sentence(
                words=[token.text for token in sentence],
                pos=[token.pos_ or token.tag_ for token in sentence],
                meta={
                    "lemmas": [token.lemma_ for token in sentence],
                    "char_offsets": [(token.idx, token.idx + len(token.text))
                                     for token in sentence]
                }
            ))
        return sentences


class PreprocessedReader(Reader):
    """Reader for preprocessed text."""

    def read(self, list_of_sentence_tuples):
        sentences = []
        for sentence_id, sentence in enumerate(list_of_sentence_tuples):
            words = [word for word, pos_tag in sentence]
            pos_tags = [pos_tag for word, pos_tag in sentence]
            shift = 0
            sentences.append(Sentence(
                words=words,
                pos=pos_tags
            ))
            shift += len(' '.join(words))
        return sentences
