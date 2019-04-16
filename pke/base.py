# -*- coding: utf-8 -*-

"""Base classes for the pke module."""

from collections import defaultdict

from pke.data_structures import Candidate, Document
from pke.readers import MinimalCoreNLPReader, RawTextReader

from nltk.stem.snowball import SnowballStemmer
from nltk import RegexpParser
from nltk.corpus import stopwords
from nltk.tag.mapping import map_tag

from string import punctuation
import os
import logging
import codecs

from six import string_types

from builtins import str

ISO_to_language = {'en': 'english', 'pt': 'portuguese', 'fr': 'french',
                   'es': 'spanish', 'it': 'italian', 'nl': 'dutch',
                   'de': 'german'}

escaped_punctuation = {'-lrb-': '(', '-rrb-': ')', '-lsb-': '[', '-rsb-': ']',
                       '-lcb-': '{', '-rcb-': '}'}


class LoadFile(object):
    """The LoadFile class that provides base functions."""

    def __init__(self):
        """Initializer for LoadFile class."""

        self.input_file = None
        """Path to the input file."""

        self.language = None
        """Language of the input file."""

        self.normalization = None
        """Word normalization method."""

        self.sentences = []
        """Sentence container (list of Sentence objects)."""

        self.candidates = defaultdict(Candidate)
        """Keyphrase candidates container (dict of Candidate objects)."""

        self.weights = {}
        """Weight container (can be either word or candidate weights)."""

        self._models = os.path.join(os.path.dirname(__file__), 'models')
        """Root path of the models."""

        self._df_counts = os.path.join(self._models, "df-semeval2010.tsv.gz")
        """Path to the document frequency counts provided in pke."""

        self.stoplist = None
        """List of stopwords."""

    def load_document(self, input, **kwargs):
        """Loads the content of a document/string/stream in a given language.

        Args:
            input (str): input.
            language (str): language of the input, defaults to 'en'.
            encoding (str): encoding of the raw file.
            normalization (str): word normalization method, defaults to
                'stemming'. Other possible values are 'lemmatization' or 'None'
                for using word surface forms instead of stems/lemmas.
        """

        # get the language parameter
        language = kwargs.get('language', 'en')

        # test whether the language is known, otherwise fall back to english
        if language not in ISO_to_language:
            logging.warning(
                "ISO 639 code {} is not supported, switching to 'en'.".format(
                    language))
            language = 'en'

        # initialize document
        doc = Document()

        if isinstance(input, string_types):

            # if input is an input file
            if os.path.isfile(input):

                # an xml file is considered as a CoreNLP document
                if input.endswith('xml'):
                    parser = MinimalCoreNLPReader()
                    doc = parser.read(path=input, **kwargs)
                    doc.is_corenlp_file = True

                # other extensions are considered as raw text
                else:
                    parser = RawTextReader(language=language)
                    encoding = kwargs.get('encoding', 'utf-8')
                    with codecs.open(input, 'r', encoding=encoding) as file:
                        text = file.read()
                    doc = parser.read(text=text, path=input, **kwargs)

            # if input is a string
            else:
                parser = RawTextReader(language=language)
                doc = parser.read(text=input, **kwargs)

        elif getattr(input, 'read', None):
            # check whether it is a compressed CoreNLP document
            name = getattr(input, 'name', None)
            if name and name.endswith('xml'):
                parser = MinimalCoreNLPReader()
                doc = parser.read(path=input, **kwargs)
                doc.is_corenlp_file = True
            else:
                parser = RawTextReader(language=language)
                doc = parser.read(text=input.read(), **kwargs)

        else:
            logging.error('Cannot process {}'.format(type(input)))

        # set the input file
        self.input_file = doc.input_file

        # set the language of the document
        self.language = language

        # set the sentences
        self.sentences = doc.sentences

        # initialize the stoplist
        self.stoplist = stopwords.words(ISO_to_language[self.language])

        # word normalization
        self.normalization = kwargs.get('normalization', 'stemming')
        if self.normalization == 'stemming':
            self.apply_stemming()
        elif self.normalization is None:
            for i, sentence in enumerate(self.sentences):
                self.sentences[i].stems = sentence.words

        # lowercase the normalized words
        for i, sentence in enumerate(self.sentences):
            self.sentences[i].stems = [w.lower() for w in sentence.stems]

        # POS normalization
        if getattr(doc, 'is_corenlp_file', False):
            self.normalize_pos_tags()
            self.unescape_punctuation_marks()

    def apply_stemming(self):
        """Populates the stem containers of sentences."""

        if self.language == 'en':
            # create a new instance of a porter stemmer
            stemmer = SnowballStemmer("porter")
        else:
            # create a new instance of a porter stemmer
            stemmer = SnowballStemmer(ISO_to_language[self.language],
                                      ignore_stopwords=True)

        # iterate throughout the sentences
        for i, sentence in enumerate(self.sentences):
            self.sentences[i].stems = [stemmer.stem(w) for w in sentence.words]

    def normalize_pos_tags(self):
        """Normalizes the PoS tags from udp-penn to UD."""

        if self.language == 'en':
            # iterate throughout the sentences
            for i, sentence in enumerate(self.sentences):
                self.sentences[i].pos = [map_tag('en-ptb', 'universal', tag)
                                         for tag in sentence.pos]

    def unescape_punctuation_marks(self):
        """Replaces the special punctuation marks produced by CoreNLP."""

        for i, sentence in enumerate(self.sentences):
            for j, word in enumerate(sentence.words):
                l_word = word.lower()
                self.sentences[i].words[j] = escaped_punctuation.get(l_word,
                                                                     word)

    def is_redundant(self, candidate, prev, minimum_length=1):
        """Test if one candidate is redundant with respect to a list of already
        selected candidates. A candidate is considered redundant if it is
        included in another candidate that is ranked higher in the list.

        Args:
            candidate (str): the lexical form of the candidate.
            prev (list): the list of already selected candidates (lexical
                forms).
            minimum_length (int): minimum length (in words) of the candidate
                to be considered, defaults to 1.
        """

        # get the tokenized lexical form from the candidate
        candidate = self.candidates[candidate].lexical_form

        # only consider candidate greater than one word
        if len(candidate) < minimum_length:
            return False

        # get the tokenized lexical forms from the selected candidates
        prev = [self.candidates[u].lexical_form for u in prev]

        # loop through the already selected candidates
        for prev_candidate in prev:
            for i in range(len(prev_candidate) - len(candidate) + 1):
                if candidate == prev_candidate[i:i + len(candidate)]:
                    return True
        return False

    def get_n_best(self, n=10, redundancy_removal=False, stemming=False):
        """Returns the n-best candidates given the weights.

        Args:
            n (int): the number of candidates, defaults to 10.
            redundancy_removal (bool): whether redundant keyphrases are
                filtered out from the n-best list, defaults to False.
            stemming (bool): whether to extract stems or surface forms
                (lowercased, first occurring form of candidate), default to
                False.
        """

        # sort candidates by descending weight
        best = sorted(self.weights, key=self.weights.get, reverse=True)

        # remove redundant candidates
        if redundancy_removal:

            # initialize a new container for non redundant candidates
            non_redundant_best = []

            # loop through the best candidates
            for candidate in best:

                # test wether candidate is redundant
                if self.is_redundant(candidate, non_redundant_best):
                    continue

                # add the candidate otherwise
                non_redundant_best.append(candidate)

                # break computation if the n-best are found
                if len(non_redundant_best) >= n:
                    break

            # copy non redundant candidates in best container
            best = non_redundant_best

        # get the list of best candidates as (lexical form, weight) tuples
        n_best = [(u, self.weights[u]) for u in best[:min(n, len(best))]]

        # replace with surface forms if no stemming
        if not stemming:
            n_best = [(' '.join(self.candidates[u].surface_forms[0]).lower(),
                       self.weights[u]) for u in best[:min(n, len(best))]]

        if len(n_best) < n:
            logging.warning(
                'Not enough candidates to choose from '
                '({} requested, {} given)'.format(n, len(n_best)))

        # return the list of best candidates
        return n_best

    def add_candidate(self, words, stems, pos, offset, sentence_id):
        """Add a keyphrase candidate to the candidates container.

        Args:
            words (list): the words (surface form) of the candidate.
            stems (list): the stemmed words of the candidate.
            pos (list): the Part-Of-Speeches of the words in the candidate.
            offset (int): the offset of the first word of the candidate.
            sentence_id (int): the sentence id of the candidate.
        """

        # build the lexical (canonical) form of the candidate using stems
        lexical_form = ' '.join(stems)

        # add/update the surface forms
        self.candidates[lexical_form].surface_forms.append(words)

        # add/update the lexical_form
        self.candidates[lexical_form].lexical_form = stems

        # add/update the POS patterns
        self.candidates[lexical_form].pos_patterns.append(pos)

        # add/update the offsets
        self.candidates[lexical_form].offsets.append(offset)

        # add/update the sentence ids
        self.candidates[lexical_form].sentence_ids.append(sentence_id)

    def ngram_selection(self, n=3):
        """Select all the n-grams and populate the candidate container.

        Args:
            n (int): the n-gram length, defaults to 3.
        """

        # loop through the sentences
        for i, sentence in enumerate(self.sentences):

            # limit the maximum n for short sentence
            skip = min(n, sentence.length)

            # compute the offset shift for the sentence
            shift = sum([s.length for s in self.sentences[0:i]])

            # generate the ngrams
            for j in range(sentence.length):
                for k in range(j + 1, min(j + 1 + skip, sentence.length + 1)):
                    # add the ngram to the candidate container
                    self.add_candidate(words=sentence.words[j:k],
                                       stems=sentence.stems[j:k],
                                       pos=sentence.pos[j:k],
                                       offset=shift + j,
                                       sentence_id=i)

    def longest_pos_sequence_selection(self, valid_pos=None):
        self.longest_sequence_selection(
            key=lambda s: s.pos, valid_values=valid_pos)

    def longest_keyword_sequence_selection(self, keywords):
        self.longest_sequence_selection(
            key=lambda s: s.stems, valid_values=keywords)

    def longest_sequence_selection(self, key, valid_values):
        """Select the longest sequences of given POS tags as candidates.

        Args:
            key (func) : function that given a sentence return an iterable
            valid_values (set): the set of valid values, defaults to None.
        """

        # loop through the sentences
        for i, sentence in enumerate(self.sentences):

            # compute the offset shift for the sentence
            shift = sum([s.length for s in self.sentences[0:i]])

            # container for the sequence (defined as list of offsets)
            seq = []

            # loop through the tokens
            for j, value in enumerate(key(self.sentences[i])):

                # add candidate offset in sequence and continue if not last word
                if value in valid_values:
                    seq.append(j)
                    if j < (sentence.length - 1):
                        continue

                # add sequence as candidate if non empty
                if seq:

                    # add the ngram to the candidate container
                    self.add_candidate(words=sentence.words[seq[0]:seq[-1] + 1],
                                       stems=sentence.stems[seq[0]:seq[-1] + 1],
                                       pos=sentence.pos[seq[0]:seq[-1] + 1],
                                       offset=shift + seq[0],
                                       sentence_id=i)

                # flush sequence container
                seq = []

    def grammar_selection(self, grammar=None):
        """Select candidates using nltk RegexpParser with a grammar defining
        noun phrases (NP).

        Args:
            grammar (str): grammar defining POS patterns of NPs.
        """

        # initialize default grammar if none provided
        if grammar is None:
            grammar = r"""
                NBAR:
                    {<NOUN|PROPN|ADJ>*<NOUN|PROPN>} 
                    
                NP:
                    {<NBAR>}
                    {<NBAR><ADP><NBAR>}
            """

        # initialize chunker
        chunker = RegexpParser(grammar)

        # loop through the sentences
        for i, sentence in enumerate(self.sentences):

            # compute the offset shift for the sentence
            shift = sum([s.length for s in self.sentences[0:i]])

            # convert sentence as list of (offset, pos) tuples
            tuples = [(str(j), sentence.pos[j]) for j in range(sentence.length)]

            # parse sentence
            tree = chunker.parse(tuples)

            # find candidates
            for subtree in tree.subtrees():
                if subtree.label() == 'NP':
                    leaves = subtree.leaves()

                    # get the first and last offset of the current candidate
                    first = int(leaves[0][0])
                    last = int(leaves[-1][0])

                    # add the NP to the candidate container
                    self.add_candidate(words=sentence.words[first:last + 1],
                                       stems=sentence.stems[first:last + 1],
                                       pos=sentence.pos[first:last + 1],
                                       offset=shift + first,
                                       sentence_id=i)

    @staticmethod
    def _is_alphanum(word, valid_punctuation_marks='-'):
        """Check if a word is valid, i.e. it contains only alpha-numeric
        characters and valid punctuation marks.

        Args:
            word (string): a word.
            valid_punctuation_marks (str): punctuation marks that are valid
                    for a candidate, defaults to '-'.
        """
        for punct in valid_punctuation_marks.split():
            word = word.replace(punct, '')
        return word.isalnum()

    def candidate_filtering(self,
                            stoplist=None,
                            minimum_length=3,
                            minimum_word_size=2,
                            valid_punctuation_marks='-',
                            maximum_word_number=5,
                            only_alphanum=True,
                            pos_blacklist=None):
        """Filter the candidates containing strings from the stoplist. Only
        keep the candidates containing alpha-numeric characters (if the
        non_latin_filter is set to True) and those length exceeds a given
        number of characters.
            
        Args:
            stoplist (list): list of strings, defaults to None.
            minimum_length (int): minimum number of characters for a
                candidate, defaults to 3.
            minimum_word_size (int): minimum number of characters for a
                token to be considered as a valid word, defaults to 2.
            valid_punctuation_marks (str): punctuation marks that are valid
                for a candidate, defaults to '-'.
            maximum_word_number (int): maximum length in words of the
                candidate, defaults to 5.
            only_alphanum (bool): filter candidates containing non (latin)
                alpha-numeric characters, defaults to True.
            pos_blacklist (list): list of unwanted Part-Of-Speeches in
                candidates, defaults to [].
        """

        if stoplist is None:
            stoplist = []

        if pos_blacklist is None:
            pos_blacklist = []

        # loop through the candidates
        for k in list(self.candidates):

            # get the candidate
            v = self.candidates[k]

            # get the words from the first occurring surface form
            words = [u.lower() for u in v.surface_forms[0]]

            # discard if words are in the stoplist
            if set(words).intersection(stoplist):
                del self.candidates[k]

            # discard if tags are in the pos_blacklist
            elif set(v.pos_patterns[0]).intersection(pos_blacklist):
                del self.candidates[k]

            # discard if containing tokens composed of only punctuation
            elif any([set(u).issubset(set(punctuation)) for u in words]):
                del self.candidates[k]

            # discard candidates composed of 1-2 characters
            elif len(''.join(words)) < minimum_length:
                del self.candidates[k]

            # discard candidates containing small words (1-character)
            elif min([len(u) for u in words]) < minimum_word_size:
                del self.candidates[k]

            # discard candidates composed of more than 5 words
            elif len(v.lexical_form) > maximum_word_number:
                del self.candidates[k]

            # discard if not containing only alpha-numeric characters
            if only_alphanum and k in self.candidates:
                if not all([self._is_alphanum(w, valid_punctuation_marks)
                            for w in words]):
                    del self.candidates[k]

