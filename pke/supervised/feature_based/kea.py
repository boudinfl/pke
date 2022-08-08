# -*- coding: utf-8 -*-
# Author: Florian Boudin
# Date: 09-10-2018

"""Kea supervised keyphrase extraction model.

Kea is a supervised model for keyphrase extraction that uses two features,
namely TF x IDF and first occurrence, to classify keyphrase candidates as
keyphrase or not. The model is described in:

* Ian Witten, Gordon Paynter, Eibe Frank, Carl Gutwin and Craig Nevill-Mannin.
  KEA: Practical Automatic Keyphrase Extraction.
  *Proceedings of the 4th ACM Conference on Digital Libraries*, pages 254–255,
  1999.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import logging

import numpy as np
from joblib import dump as dump_model
from sklearn.naive_bayes import MultinomialNB

from pke.supervised.api import SupervisedLoadFile
from pke.utils import load_document_frequency_file


class Kea(SupervisedLoadFile):
    """Kea keyphrase extraction model.

    Parameterized example::

        import pke

        # 1. create a Kea extractor.
        extractor = pke.supervised.Kea()

        # 2. load the content of the document.
        stoplist = pke.lang.stopwords.get('en')
        extractor.load_document(input='path/to/input',
                                language='en',
                                stoplist=stoplist,
                                normalization=None)

        # 3. select 1-3 grams that do not start or end with a stopword as
        #    candidates. Candidates that contain punctuation marks as words
        #    are discarded.
        extractor.candidate_selection()

        # 4. classify candidates as keyphrase or not keyphrase.
        df = pke.load_document_frequency_file(input_file='path/to/df.tsv.gz')
        model_file = 'path/to/kea_model'
        extractor.candidate_weighting(model_file=model_file,
                                      df=df)

        # 5. get the 10-highest scored candidates as keyphrases
        keyphrases = extractor.get_n_best(n=10)
    """

    def __init__(self):
        """Redefining initializer for Kea."""

        super(Kea, self).__init__()

    def candidate_selection(self):
        """Select 1-3 grams of `normalized` words as keyphrase candidates.
        Candidates that start or end with a stopword are discarded. Candidates
        that contain punctuation marks (from `string.punctuation`) as words are
        filtered out.
        """

        # select ngrams from 1 to 3 grams
        self.ngram_selection(n=3)

        # filter candidates
        self.candidate_filtering()
        # TODO: is filtering only candidate with punctuation mandatory ?
        #self.candidate_filtering(list(string.punctuation))

        # filter candidates that start or end with a stopword
        for k in list(self.candidates):

            # get the candidate
            v = self.candidates[k]

            # delete if candidate contains a stopword in first/last position
            words = [u.lower() for u in v.surface_forms[0]]
            if words[0] in self.stoplist or words[-1] in self.stoplist:
                del self.candidates[k]

    def feature_extraction(self, df=None, training=False):
        """Extract features for each keyphrase candidate. Features are the
        tf*idf of the candidate and its first occurrence relative to the
        document.

        Args:
            df (dict): document frequencies, the number of documents should be
                specified using the "--NB_DOC--" key.
            training (bool): indicates whether features are computed for the
                training set for computing IDF weights, defaults to false.
        """

        # initialize default document frequency counts if none provided
        if df is None:
            logging.warning('LoadFile._df_counts is hard coded to {}'.format(
                self._df_counts))
            df = load_document_frequency_file(self._df_counts, delimiter='\t')

        # initialize the number of documents as --NB_DOC--
        N = df.get('--NB_DOC--', 0) + 1
        if training:
            N -= 1

        # find the maximum offset
        maximum_offset = float(sum([s.length for s in self.sentences]))

        for k, v in self.candidates.items():

            # get candidate document frequency
            candidate_df = 1 + df.get(k, 0)

            # hack for handling training documents
            if training and candidate_df > 1:
                candidate_df -= 1

            # compute the tf*idf of the candidate
            idf = math.log(N / candidate_df, 2)

            # add the features to the instance container
            self.instances[k] = np.array([len(v.surface_forms) * idf,
                                          v.offsets[0] / maximum_offset])

        # scale features
        self.feature_scaling()

    def candidate_weighting(self, model_file=None, df=None):
        """Extract features and classify candidates.

        Args:
            model_file (str): path to the model file.
            df (dict): document frequencies, the number of documents should
                    be specified using the "--NB_DOC--" key.
        """
        if not self.candidates:
            return

        self.feature_extraction(df=df)
        self.classify_candidates(model=model_file)

    @staticmethod
    def train(training_instances, training_classes, model_file):
        """ Train a Naive Bayes classifier and store the model in a file.

            Args:
                training_instances (list): list of features.
                training_classes (list): list of binary values.
                model_file (str): the model output file.
        """

        clf = MultinomialNB()
        clf.fit(training_instances, training_classes)
        dump_model(clf, model_file)
