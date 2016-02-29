# -*- coding: utf-8 -*-

""" Base classes for the pke module. """

from .readers import MinimalCoreNLPParser
from collections import defaultdict
from nltk.stem.snowball import SnowballStemmer
from string import letters, digits
import re

class Sentence(object):
    """ The sentence data structure. """

    def __init__(self, words):

        self.words = words
        """ tokens as a list. """

        self.pos = []
        """ Part-Of-Speeches as a list. """

        self.stems = []
        """ stems as a list. """

        self.length = len(words)
        """ length of the sentence. """


class Candidate(object):
    """ The keyphrase candidate data structure. """

    def __init__(self):

        self.surface_forms = []
        """ the surface forms of the candidate. """

        self.offsets = []
        """ the offsets of the surface forms. """

        self.pos_patterns = []
        """ the Part-Of-Speech patterns of the candidate. """

        self.lexical_form = []
        """ the lexical form of the candidate. """


class LoadFile(object):
    """ The LoadFile class that provides base functions. """

    def __init__(self, input_file, language='english'):
        """ Initializer for LoadFile class.

            Args:
                input_file (str): the path of the input file.
        """

        self.input_file = input_file
        """ The path of the input file. """

        self.language = language
        """ The language of the input file. """

        self.sentences = []
        """ The sentence container (list of Sentence). """

        self.candidates = defaultdict(Candidate)
        """ The candidate container (dict of Candidate). """

        self.weights = {}
        """ The weight container (can be either word or candidate weights). """


    def read_corenlp_document(self, use_lemmas=True, language="porter"):
        """ Read the input file in CoreNLP XML format and populate the sentence
            list.

            Args:
                use_lemmas (bool): weither lemmas from stanford corenlp are used
                    instead of stems (computed by nltk), defaults to True.
                language (str): the language of the stemming (if used), defaults
                    to porter.
        """

        # parse the document using the Minimal CoreNLP parser
        parse = MinimalCoreNLPParser(self.input_file)

        # populate the sentence container
        for sentence in parse.sentences:

            new_sentence = Sentence(words=sentence["words"])
            new_sentence.pos = sentence["POS"]

            if use_lemmas:
                new_sentence.stems = [t.lower() for t in sentence["lemmas"]]
            else:
                for word in new_sentence.words:
                    new_sentence.stems.append(
                        SnowballStemmer(language).stem(word.lower())
                    )

            self.sentences.append(new_sentence)


    def get_n_best(self, n=10, redundancy_removal=False):
        """ Returns the n-best candidates given the weights. 

            Args:
                n (int): the number of candidates, defaults to 10.
                redundancy_removal (bool): whether redundant keyphrases are 
                    filtered out from the n-best list, defaults to False:
        """

        best = sorted(self.weights, key=self.weights.get, reverse=True)

        if redundancy_removal:
            l = []
            for i in range(len(best)):
                is_redundant = False
                for c in best[:i]:
                    if len(self.candidates[best[i]].lexical_form) > 1 and\
                        re.search('(^|\s)'+best[i]+'($|\s)', c):
                        is_redundant = True
                if not is_redundant:
                    l.append(best[i])
                if len(l) >= n:
                    break
            best = l

        return [(u, self.weights[u]) for u in best[:min(n, len(best))]]


    def ngram_selection(self, n=3):
        """ Select all the n-grams and populate the candidate container.

            Args:
                n (int): the n-gram length, defaults to 3.
        """

        for i, sentence in enumerate(self.sentences):

            skip = min(n, sentence.length)
            shift = sum([s.length for s in self.sentences[0:i]])

            for j in range(sentence.length):
                for k in range(j+1, min(j+1+skip, sentence.length+1)):

                    surface_form = sentence.words[j:k]
                    norm_form = sentence.stems[j:k]
                    key = ' '.join(norm_form)
                    pos_pattern = sentence.pos[j:k]

                    self.candidates[key].surface_forms.append(surface_form)
                    self.candidates[key].lexical_form = norm_form
                    self.candidates[key].offsets.append(shift+j)
                    self.candidates[key].pos_patterns.append(pos_pattern)


    def sequence_selection(self, pos=None):
        """ Select all the n-grams and populate the candidate container.

            Args:
                pos (list): the set of valid POS tags, defaults to None.
        """

        for i, sentence in enumerate(self.sentences):

            shift = sum([s.length for s in self.sentences[0:i]])
            seq = []

            for j in range(sentence.length):

                # add candidate offset in sequence and continue if not last word
                if sentence.pos[j] in pos:
                    seq.append(j)
                    if j < (sentence.length - 1):
                        continue

                # add candidate
                if seq:

                    surface_form = sentence.words[seq[0]:seq[-1]+1]
                    norm_form = sentence.stems[seq[0]:seq[-1]+1]
                    key = ' '.join(norm_form)
                    pos_pattern = sentence.pos[seq[0]:seq[-1]+1]

                    self.candidates[key].surface_forms.append(surface_form)
                    self.candidates[key].lexical_form = norm_form
                    self.candidates[key].offsets.append(shift+j)
                    self.candidates[key].pos_patterns.append(pos_pattern)

                # flush sequence container
                seq = []


    def candidate_filtering(self,
                            stoplist=None,
                            mininum_length=3,
                            mininum_word_size=2,
                            valid_punctuation_marks='-'):
        """ Filter the candidates containing strings from the stoplist. Only 
            keep the candidates containing alpha-numeric characters and those 
            length exceeds a given number of characters.

            Args:
                stoplist (list): list of strings, defaults to None.
                mininum_length (int): minimum number of characters for a 
                    candidate, defaults to 3.
                mininum_word_size (int): minimum number of characters for a
                    token to be considered as a valid word, defaults to 2.
                valid_punctuation_marks (str): punctuation marks that are valid
                    for a candidate, defaults to '-'.
        """

        printable = set(letters + digits + valid_punctuation_marks)

        # loop throught the candidates
        for k, v in self.candidates.items():
            
            # get the words from the first occurring surface form
            words = [u.lower() for u in v.surface_forms[0]]

            # discard if words are in the stoplist
            if set(words).intersection(stoplist):
                del self.candidates[k]

            # discard if not containing only alpha-numeric characters
            elif not set(''.join(words)).issubset(printable):
                del self.candidates[k]

            # discard if containing tokens composed of only punctuation
            elif any([set(u).issubset(valid_punctuation_marks) for u in words]):
                del self.candidates[k]

            # discard candidates composed of 1-2 characters
            elif len(''.join(words)) < mininum_length:
                del self.candidates[k]

            # discard candidates containing small words (1-character)
            elif min([len(u) for u in words]) < mininum_word_size:
                del self.candidates[k]