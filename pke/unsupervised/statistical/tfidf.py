# -*- coding: utf-8 -*-
# Author: Florian Boudin
# Date: 09-10-2018

"""TF-IDF keyphrase extraction model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import string
import logging

from pke.base import LoadFile
from pke.utils import load_document_frequency_file


class TfIdf(LoadFile):
    """TF*IDF keyphrase extraction model.

    Parameterized example::

        import string
        import pke

        # 1. create a TfIdf extractor.
        extractor = pke.unsupervised.TfIdf()

        # 2. load the content of the document.
        extractor.load_document(input='path/to/input',
                                language='en',
                                normalization=None)

        # 3. select {1-3}-grams not containing punctuation marks as candidates.
        extractor.candidate_selection(n=3,
                                      stoplist=list(string.punctuation))

        # 4. weight the candidates using a `tf` x `idf`
        df = pke.load_document_frequency_file(input_file='path/to/df.tsv.gz')
        extractor.candidate_weighting(df=df)

        # 5. get the 10-highest scored candidates as keyphrases
        keyphrases = extractor.get_n_best(n=10)
    """

    def candidate_selection(self, n=3, stoplist=None, **kwargs):
        """Select 1-3 grams as keyphrase candidates.

        Args:
            n (int): the length of the n-grams, defaults to 3.
            stoplist (list): the stoplist for filtering candidates, defaults to
                `None`. Words that are punctuation marks from
                `string.punctuation` are not allowed.

        """

        # select ngrams from 1 to 3 grams
        self.ngram_selection(n=n)

        # initialize empty list if stoplist is not provided
        if stoplist is None:
            stoplist = list(string.punctuation)

        # filter candidates containing punctuation marks
        self.candidate_filtering(stoplist=stoplist)

    def candidate_weighting(self, df=None):
        """Candidate weighting function using document frequencies.

        Args:
            df (dict): document frequencies, the number of documents should be
                specified using the "--NB_DOC--" key.
        """

        # initialize default document frequency counts if none provided
        if df is None:
            logging.warning('LoadFile._df_counts is hard coded to {}'.format(
                self._df_counts))
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
