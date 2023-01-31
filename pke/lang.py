# -*- coding: utf-8 -*-

"""Language resources of pke.

Lists of stopwords in different languages.

These lists are taken from spacy.

Langcodes.

"""

import importlib

langcodes = {
       "ar": "arabic",
       "da": "danish",
       "en": "english",
       "es": "spanish",
       "fi": "finnish",
       "fr": "french",
       "de": "german",
       "hu": "hungarian",
       "it": "italian",
       "ja": "japanese",
       "nb": "norwegian",
       "nl": "dutch",
       "pt": "portuguese",
       "ro": "romanian",
       "ru": "russian",
       "sv": "swedish"
}

stopwords = {}
for langcode in langcodes:
    try:
        tmp = importlib.import_module('spacy.lang.{}'.format(langcode))
        stopwords[langcode] = tmp.stop_words.STOP_WORDS
    except ModuleNotFoundError:
        continue
