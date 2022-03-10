# -*- coding: utf-8 -*-
# Author: Florian Boudin
# Date: 09-10-2018

"""KP-Miner keyphrase extraction model.

Statistical approach to keyphrase extraction described in:

* Samhaa R. El-Beltagy and Ahmed Rafea.
  KP-Miner: Participation in SemEval-2.
  *Proceedings of SemEval*, pages 190-193, 2010.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import string
import logging

from pke.base import LoadFile
from pke.utils import load_document_frequency_file


class KPMiner(LoadFile):
    """KP-Miner keyphrase extraction model.

    Parameterized example::

        import pke

        # 1. create a KPMiner extractor.
        extractor = pke.unsupervised.KPMiner()

        # 2. load the content of the document.
        extractor.load_document(input='path/to/input',
                                language='en',
                                normalization=None)


        # 3. select {1-5}-grams that do not contain punctuation marks or
        #    stopwords as keyphrase candidates. Set the least allowable seen
        #    frequency to 5 and the number of words after which candidates are
        #    filtered out to 200.
        lasf = 5
        cutoff = 200
        extractor.candidate_selection(lasf=lasf, cutoff=cutoff)

        # 4. weight the candidates using KPMiner weighting function.
        df = pke.load_document_frequency_file(input_file='path/to/df.tsv.gz')
        alpha = 2.3
        sigma = 3.0
        extractor.candidate_weighting(df=df, alpha=alpha, sigma=sigma)

        # 5. get the 10-highest scored candidates as keyphrases
        keyphrases = extractor.get_n_best(n=10)
    """

    def candidate_selection(self, lasf=3, cutoff=400):
        """The candidate selection as described in the KP-Miner paper.

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

        # filter candidates containing stopwords
        self.candidate_filtering()

        # further filter candidates using lasf and cutoff
        # Python 2/3 compatible
        for k in list(self.candidates):

            # get the candidate
            v = self.candidates[k]

            # delete if first candidate offset is greater than cutoff
            if v.offsets[0] > cutoff:
                del self.candidates[k]

            # delete if frequency is lower than lasf
            elif len(v.surface_forms) < lasf:
                del self.candidates[k]

    def candidate_weighting(self, df=None, sigma=3.0, alpha=2.3):
        """Candidate weight calculation as described in the KP-Miner paper.

        Note:
            w = tf * idf * B * P_f
            with

              * B = N_d / (P_d * alpha) and B = min(sigma, B)
              * N_d = the number of all candidate terms
              * P_d = number of candidates whose length exceeds one
              * P_f = 1

        Args:
            df (dict): document frequencies, the number of documents should
                be specified using the "--NB_DOC--" key.
            sigma (int): parameter for boosting factor, defaults to 3.0.
            alpha (int): parameter for boosting factor, defaults to 2.3.
        """

        # initialize default document frequency counts if none provided
        if df is None:
            logging.warning('LoadFile._df_counts is hard coded to {}'.format(
                self._df_counts))
            df = load_document_frequency_file(self._df_counts, delimiter='\t')

        # initialize the number of documents as --NB_DOC-- + 1 (current)
        N = 1 + df.get('--NB_DOC--', 0)

        # compute the number of candidates whose length exceeds one
        P_d = sum([len(v.surface_forms) for v in self.candidates.values()
                   if len(v.lexical_form) > 1])

        # fall back to 1 if all candidates are words
        P_d = max(1, P_d)

        # compute the number of all candidate terms
        N_d = sum([len(v.surface_forms) for v in self.candidates.values()])

        # compute the boosting factor
        B = min(N_d / (P_d * alpha), sigma)

        # loop throught the candidates
        for k, v in self.candidates.items():

            # get candidate document frequency
            candidate_df = 1

            # get the df for unigram only
            if len(v.lexical_form) == 1:
                candidate_df += df.get(k, 0)

            # compute the idf score
            idf = math.log(N / candidate_df, 2)

            if len(v.lexical_form) == 1:
                # If single word candidate do not apply boosting factor
                self.weights[k] = len(v.surface_forms) * idf
            else:
                self.weights[k] = len(v.surface_forms) * B * idf
