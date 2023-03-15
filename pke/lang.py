# -*- coding: utf-8 -*-

"""Language resources of pke.

Lists of stopwords in different languages.

These lists are taken from spacy.

Langcodes.

"""

import importlib

# This dictionnary holds only languages supported by `pke`.
# Supported languages need a stemmer and a spacy model.

# This dictionnary maps spacy's langcode to stemmer language
#  (ordered by language name).
# The list of languages was obtained using:
#  `nltk.stem.SnowballStemmer.languages`

langcodes = {
    # "ar": "arabic", # no spacy model yet ;)
    "da": "danish",
    "nl": "dutch",
    "en": "english",
    "fi": "finnish",
    "fr": "french",
    "de": "german",
    # "hu": "hungarian", # no spacy model yet ;)
    "it": "italian",
    "nb": "norwegian",
    "pt": "portuguese",
    "ro": "romanian",
    "ru": "russian",
    "es": "spanish",
    "sv": "swedish",
}

stopwords = {}
for langcode in langcodes:
    try:
        tmp = importlib.import_module('spacy.lang.{}'.format(langcode))
        stopwords[langcode] = tmp.stop_words.STOP_WORDS
    except ModuleNotFoundError:
        continue
