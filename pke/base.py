# -*- coding: utf-8 -*-

""" Base classes for the pke module. """

from .readers import MinimalCoreNLPParser, PreProcessedTextReader, RawTextReader
from collections import defaultdict
from nltk.stem.snowball import SnowballStemmer as Stemmer
from string import letters, digits


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

        self.meta = {}
        """ meta-information of the sentence. """


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

        self.meta = []
        """ meta-information of the candidate. """


class LoadFile(object):
    """ The LoadFile class that provides base functions. """

    def __init__(self, input_file, language='english'):
        """ Initializer for LoadFile class.

            Args:
                input_file (str): the path of the input file.
                language (str): the language of the input file (used for
                    stoplist), defaults to english.
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


    def read_document(self,
                      format='raw',
                      use_lemmas=False,
                      stemmer='porter',
                      sep='/'):
        """ Read the input file in a given format.

            Args:
                format (str): the input format, defaults to raw.
                use_lemmas (bool): whether lemmas from stanford corenlp are used
                    instead of stems (computed by nltk), defaults to False.
                stemmer (str): the stemmer in nltk to use (if used), defaults
                    to porter.
                sep (str): the separator of the tagged word, defaults to /.
        """

        if format == 'raw':
            self.read_raw_document(stemmer=stemmer)
        elif format == 'preprocessed':
            self.read_preprocessed_document(stemmer=stemmer, sep=sep)
        elif format == 'corenlp':
            self.read_corenlp_document(use_lemmas=use_lemmas, stemmer=stemmer)


    def read_corenlp_document(self, use_lemmas=False, stemmer='porter'):
        """ Read the input file in CoreNLP XML format and populate the sentence
            list.

            Args:
                use_lemmas (bool): whether lemmas from stanford corenlp are used
                    instead of stems (computed by nltk), defaults to False.
                stemmer (str): the stemmer in nltk to use (if used), defaults
                    to porter.
        """

        # parse the document using the Minimal CoreNLP parser
        parse = MinimalCoreNLPParser(self.input_file)

        # loop through the parsed sentences
        for i, sentence in enumerate(parse.sentences):

            # add the sentence to the container
            self.sentences.append(Sentence(words=sentence['words']))

            # add the POS
            self.sentences[i].pos = sentence['POS']

            # add the lemmas
            self.sentences[i].stems = sentence['lemmas']

            # flatten with the stems if required
            if not use_lemmas:
                for j, word in enumerate(self.sentences[i].words):
                    self.sentences[i].stems[j] = Stemmer(stemmer).stem(word)

            # lowercase the stems/lemmas
            for j, stem in enumerate(self.sentences[i].stems):
                self.sentences[i].stems[j] = stem.lower()

            # add the meta-information
            for k, v in sentence.iteritems():
                if k not in set(['POS', 'lemmas', 'words']):
                    self.sentences[i].meta[k] = v


    def read_preprocessed_document(self, stemmer='porter', sep='/'):
        """ Read the preprocessed input file and populate the sentence list.

            Args:
                stemmer (str): the stemmer in nltk to use, defaults to porter.
                sep (str): the separator of the tagged word, defaults to /.
        """

        # parse the document using the preprocessed text parser
        parse = PreProcessedTextReader(self.input_file, sep=sep)

        # loop through the parsed sentences
        for i, sentence in enumerate(parse.sentences):

            # add the sentence to the container
            self.sentences.append(Sentence(words=sentence['words']))

            # add the POS
            self.sentences[i].pos = sentence['POS']

            # add the stems
            for j, word in enumerate(self.sentences[i].words):
                self.sentences[i].stems.append(Stemmer(stemmer).stem(word))

            # lowercase the stems/lemmas
            for j, stem in enumerate(self.sentences[i].stems):
                self.sentences[i].stems[j] = stem.lower()


    def read_raw_document(self, stemmer='porter'):
        """ Read the raw input file and populate the sentence list.

            Args:
                stemmer (str): the stemmer in nltk to use, defaults to porter.
        """

        # parse the document using the preprocessed text parser
        parse = RawTextReader(self.input_file)

        # loop through the parsed sentences
        for i, sentence in enumerate(parse.sentences):

            # add the sentence to the container
            self.sentences.append(Sentence(words=sentence['words']))

            # add the POS
            self.sentences[i].pos = sentence['POS']

            # add the stems
            for j, word in enumerate(self.sentences[i].words):
                self.sentences[i].stems.append(Stemmer(stemmer).stem(word))

            # lowercase the stems/lemmas
            for j, stem in enumerate(self.sentences[i].stems):
                self.sentences[i].stems[j] = stem.lower()


    def is_redundant(self, candidate, prev, mininum_length=1):
        """ Test if one candidate is redundant with respect to a list of already
            selected candidates. A candidate is considered redundant if it is
            included in another candidate that is ranked higher in the list.

            Args:
                candidate (str): the lexical form of the candidate.
                prev (list): the list of already selected candidates (lexical
                    forms).
                mininum_length (int): minimum length (in words) of the candidate
                    to be considered, defaults to 1.
        """

        # get the tokenized lexical form from the candidate
        candidate = self.candidates[candidate].lexical_form

        # only consider candidate greater than one word
        if len(candidate) < mininum_length:
            return False

        # get the tokenized lexical forms from the selected candidates
        prev = [self.candidates[u].lexical_form for u in prev]

        # loop through the already selected candidates
        for prev_candidate in  prev:
            for i in range(len(prev_candidate)-len(candidate)+1):
                if candidate == prev_candidate[i:i+len(candidate)]:
                    return True
        return False


    def get_n_best(self, n=10, redundancy_removal=False):
        """ Returns the n-best candidates given the weights.

            Args:
                n (int): the number of candidates, defaults to 10.
                redundancy_removal (bool): whether redundant keyphrases are
                    filtered out from the n-best list, defaults to False.
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

        # return the list of best candidates as (lexical form, weight) tuples
        return [(u, self.weights[u]) for u in best[:min(n, len(best))]]


    def add_candidate(self, words, stems, pos, offset, meta):
        """ Add a keyphrase candidate to the candidates container.

            Args:
                words (list): the words (surface form) of the candidate.
                stems (list): the stemmed words of the candidate.
                pos (list): the Part-Of-Speeches of the words in the candidate.
                offset (int): the offset of the first word of the candidate.
                meta (dict): the meta-information from the sentence.
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

        # add/update the meta information
        self.candidates[lexical_form].meta.append(meta)


    def ngram_selection(self, n=3):
        """ Select all the n-grams and populate the candidate container.

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
                for k in range(j+1, min(j+1+skip, sentence.length+1)):

                    # add the ngram to the candidate container
                    self.add_candidate(words=sentence.words[j:k],
                                       stems=sentence.stems[j:k],
                                       pos=sentence.pos[j:k],
                                       offset=shift+j,
                                       meta=sentence.meta)


    def longest_pos_sequence_selection(self, valid_pos=None):
        """ Select the longest sequences of given POS tags as candidates.

            Args:
                valid_pos (set): the set of valid POS tags, defaults to None.
        """

        # loop through the sentences
        for i, sentence in enumerate(self.sentences):

            # compute the offset shift for the sentence
            shift = sum([s.length for s in self.sentences[0:i]])

            # container for the sequence (defined as list of offsets)
            seq = []

            # loop through the tokens
            for j, pos in enumerate(self.sentences[i].pos):

                # add candidate offset in sequence and continue if not last word
                if pos in valid_pos:
                    seq.append(j)
                    if j < (sentence.length - 1):
                        continue

                # add sequence as candidate if non empty
                if seq:

                    # add the ngram to the candidate container
                    self.add_candidate(words=sentence.words[seq[0]:seq[-1]+1],
                                       stems=sentence.stems[seq[0]:seq[-1]+1],
                                       pos=sentence.pos[seq[0]:seq[-1]+1],
                                       offset=shift+j,
                                       meta=sentence.meta)

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

        # add accents for other languages
        if self.language == 'french':
            printable.update(set(u'éèêëïàçùö'))

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
