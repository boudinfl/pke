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
       "du": "dutch",
       "en": "english",
       "fi": "finnish",
       "fr": "french",
       "ge": "german",
       "hu": "hungarian",
       "it": "italian",
       "no": "norwegian",
       "pt": "portuguese",
       "ro": "romanian",
       "ru": "russian",
       "sp": "spanish",
       "sw": "swedish",
       "ja": "japanese"
}

stopwords = {}
for langcode in langcodes:
    try:
        tmp = importlib.import_module('spacy.lang.{}'.format(langcode))
        stopwords[langcode] = tmp.stop_words.STOP_WORDS
    except ModuleNotFoundError:
        continue
