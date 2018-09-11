# -*- coding: utf-8 -*-
# Author: Florian Boudin
# Date: 09-10-2018

"""Kea keyphrase extraction model.

Supervised approach to keyphrase extraction described in:

* Ian Witten, Gordon Paynter, Eibe Frank, Carl Gutwin and Craig Nevill-Mannin.
  KEA: Practical Automatic Keyphrase Extraction.
  *Proceedings of the 4th ACM Conference on Digital Libraries*, pages 254–255,
  1999.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from pke.supervised.api import SupervisedLoadFile
from pke.utils import load_document_frequency_file

import math
import string
import numpy as np

from nltk.corpus import stopwords

from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib


class Kea(SupervisedLoadFile):
    """Kea keyphrase extraction model.

    Parameterized example::

        import pke
        from nltk.corpus import stopwords

        # 1. create a Kea extractor.
        extractor = pke.supervised.Kea(input_file='path/to/input.xml')

        # 2. load the content of the document.
        extractor.read_document(format='corenlp')

        # 3. select 1-3 grams that do not start or end with a stopword as
        #    candidates.
        stoplist = stopwords.words('english')
        extractor.candidate_selection(stoplist=stoplist)

        # 4. classify candidates as keyphrase or not keyphrase.
        df = pke.load_document_frequency_file(input_file='path/to/df.tsv.gz')
        model_file = 'path/to/kea_model'
        extractor.candidate_weighting(self, model_file=model_file, df=df)

        # 5. get the 10-highest scored candidates as keyphrases
        keyphrases = extractor.get_n_best(n=10)

    """

    def __init__(self, input_file=None, language='english'):
        """Redefining initializer for Kea.

        Args:
            input_file (str): path to the input file, defaults to None.
            language (str): language of the document, used for stopwords list,
                default to 'english'.

        """

        super(Kea, self).__init__(input_file=input_file, language=language)


    def candidate_selection(self, stoplist=None):
        """Select 1-3 grams as keyphrase candidates. Candidates that start or
        end with a stopword are discarded.

        Args:
            stoplist (list): the stoplist for filtering candidates, defaults
                to the nltk stoplist. Words that are punctuation marks from
                string.punctuation are not allowed.
        """

        # select ngrams from 1 to 3 grams
        self.ngram_selection(n=3)

        # filter candidates containing punctuation marks
        self.candidate_filtering(list(string.punctuation) +
                                 ['-lrb-', '-rrb-', '-lcb-', '-rcb-', '-lsb-',
                                  '-rsb-'])

        # initialize stoplist list if not provided
        if stoplist is None:
            stoplist = stopwords.words(self.language)

        # filter candidates that start or end with a stopword
        # Python 2/3 compatible
        for k in list(self.candidates):

            # get the candidate
            v = self.candidates[k]

            # delete if candidate contains a stopword in first/last position
            words = [u.lower() for u in v.surface_forms[0]]
            if words[0] in stoplist or words[-1] in stoplist:
                del self.candidates[k]


    def feature_extraction(self, df=None, training=False):
        """Extract features (tf*idf, first occurrence and length) for each
        candidate.

        Args:
            df (dict): document frequencies, the number of documents should be
                specified using the "--NB_DOC--" key.
            training (bool): indicates whether features are computed for the
                training set for computing IDF weights, defaults to false.
        """

        # initialize default document frequency counts if none provided
        if df is None:
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
                                          v.offsets[0]/maximum_offset])

        # scale features
        self.feature_scaling()


    def candidate_weighting(self, model_file=None, df=None):
        """Extract features and classify candidates.

        Args:
            model_file (str): path to the model file.
            df (dict): document frequencies, the number of documents should
                    be specified using the "--NB_DOC--" key.
        """
        
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
        joblib.dump(clf, model_file)

