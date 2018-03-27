# -*- coding: utf-8 -*-

""" Statistical keyphrase extraction models. """

from __future__ import absolute_import
from __future__ import division

import string
import math

from pke.base import LoadFile
from pke.utils import load_document_frequency_file
from nltk.corpus import stopwords


class TfIdf(LoadFile):
    """ TF*IDF keyphrase extraction model. """

    def candidate_selection(self, n=3, stoplist=None):
        """ Select 1-3 grams as keyphrase candidates.

            Args:
                n (int): the length of the n-grams, defaults to 3.
                stoplist (list): the stoplist for filtering candidates, defaults
                    to None. Words that are punctuation marks from
                    string.punctuation are not allowed.
        """

        # select ngrams from 1 to 3 grams
        self.ngram_selection(n=n)

        # initialize empty list if stoplist is not provided
        if stoplist is None:
            stoplist = []

        # filter candidates containing punctuation marks
        self.candidate_filtering(stoplist=list(string.punctuation) +
                                 ['-lrb-', '-rrb-', '-lcb-', '-rcb-', '-lsb-',
                                  '-rsb-'] +
                                  stoplist)


    def candidate_weighting(self, df=None):
        """ Candidate weighting function using document frequencies.

            Args:
                df (dict): document frequencies, the number of documents should
                    be specified using the "--NB_DOC--" key.
        """

        # initialize default document frequency counts if none provided
        if df is None:
            df = load_document_frequency_file(self._df_counts, delimiter='\t')

        # initialize the number of documents as --NB_DOC-- + 1 (current)
        N = 1 + df.get('--NB_DOC--', 0)

        # loop throught the candidates
        for k, v in self.candidates.items():

            # get candidate document frequency
            candidate_df = 1 + df.get(k, 0)

            # compute the idf score
            idf = math.log(N / candidate_df, 2)

            # add the idf score to the weights container
            self.weights[k] = len(v.surface_forms) * idf


class KPMiner(LoadFile):
    """ KP-Miner keyphrase extraction model.

        This model was published and described in:

          * Samhaa R. El-Beltagy and Ahmed Rafea, KP-Miner: Participation in
            SemEval-2, *Proceedings of the 5th International Workshop on
            Semantic Evaluation*, pages 190-193, 2010.
    """

    def candidate_selection(self, lasf=3, cutoff=400, stoplist=None):
        """ The candidate selection as described in the KP-Miner paper.

            Args:
                lasf (int): least allowable seen frequency, defaults to 3.
                cutoff (int): the number of words after which candidates are
                    filtered out, defaults to 400.
                stoplist (list): the stoplist for filtering candidates, defaults
                    to the nltk stoplist. Words that are punctuation marks from
                    string.punctuation are not allowed.
        """

        # select ngrams from 1 to 5 grams
        self.ngram_selection(n=5)

        # initialize stoplist list if not provided
        if stoplist is None:
            stoplist = stopwords.words(self.language)

        # filter candidates containing stopwords or punctuation marks
        self.candidate_filtering(stoplist=list(string.punctuation) +
                                 ['-lrb-', '-rrb-', '-lcb-', '-rcb-', '-lsb-',
                                  '-rsb-'] +
                                  stoplist)

        # further filter candidates using lasf and cutoff
        for k, v in self.candidates.items():

            # delete if first candidate offset is greater than cutoff
            if v.offsets[0] > cutoff:
                del self.candidates[k]

            # delete if frequency is lower than lasf
            elif len(v.surface_forms) < lasf:
                del self.candidates[k]


    def candidate_weighting(self, df=None, sigma=3.0, alpha=2.3):
        """ Candidate weight calculation as described in the KP-Miner paper.

            w = tf * idf * B * P_f

            with:
                B = N_d / (P_d * alpha) and B = min(sigma, B)
                N_d = the number of all candidate terms
                P_d = number of candidates whose length exceeds one
                P_f = 1

            Args:
                df (dict): document frequencies, the number of documents should
                    be specified using the "--NB_DOC--" key.
                sigma (int): parameter for boosting factor, defaults to 3.0.
                alpha (int): parameter for boosting factor, defaults to 2.3.
        """

        # initialize default document frequency counts if none provided
        if df is None:
            df = load_document_frequency_file(self._df_counts, delimiter='\t')

        # initialize the number of documents as --NB_DOC-- + 1 (current)
        N = 1 + df.get('--NB_DOC--', 0)

        # compute the number of candidates whose length exceeds one
        P_d = sum([len(v.surface_forms) for v in self.candidates.values()
                   if len(v.lexical_form) > 1])

        # compute the number of all candidate terms
        N_d = sum([len(v.surface_forms) for v in self.candidates.values()])

        # compute the boosting factor
        B = min(N_d / (P_d*alpha), sigma)

        # loop throught the candidates
        for k, v in self.candidates.items():

            # get candidate document frequency
            candidate_df = 1

            # get the df for unigram only
            if len(v.lexical_form) == 1:
                candidate_df += df.get(k, 0)

            # compute the idf score
            idf = math.log(N / candidate_df, 2)

            self.weights[k] = len(v.surface_forms) * B * idf

