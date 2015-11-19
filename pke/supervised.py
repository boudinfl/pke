# -*- coding: utf-8 -*-

""" Supervised keyphrase extraction models. """

import string
from .base import LoadFile
from nltk.corpus import stopwords
from sklearn.externals import joblib
import numpy as np

class Kea(LoadFile):
    """ Kea keyphrase extraction model. """

    def __init__(self, input_file):
        """ Redefining initializer for Kea. """

        super(Kea, self).__init__(input_file)

        self.instances = {}
        """ The instances container. """


    def candidate_selection(self):
        """ Select 1-3 grams as keyphrase candidates. Candidates that start or 
            end with a stopword are discarded.
        """

        # select ngrams from 1 to 3 grams
        self.ngram_selection(n=3)

        # filter candidates containing punctuation marks
        self.candidate_filtering(list(string.punctuation) +
                                 ['-lrb-', '-rrb-', '-lcb-', '-rcb-', '-lsb-',
                                  '-rsb-'])

        # initialize the stoplist
        stoplist = stopwords.words(self.language)

        # filter candidates that start or end with a stopword
        for k, v in self.candidates.items():

            # delete if first candidate offset is greater than cutoff
            if v.lexical_form[0] in stoplist or v.lexical_form[-1] in stoplist:
                del self.candidates[k]


    def feature_extraction(self, idf=None):
        """ Extract features (tf*idf, first occurrence and length) for each 
            candidate.

            Args:
                idf (dict): idf weights dictionary.
        """

        # find the maximum idf weight
        default_idf = max(idf.values())

        # find the maximum offset
        maximum_offset = float(sum([s.length for s in self.sentences]))

        for k, v in self.candidates.iteritems():

            current_idf = default_idf
            if k in idf:
                current_idf = idf[k]

            self.instances[k] = np.array([len(v.surface_forms) * current_idf,
                                 v.offsets[0]/maximum_offset,
                                 len(v.lexical_form)])

    def classify_candidates(self, model):
        """ Classify the candidates as keyphrase or not keyphrase.

            Args:
                model (str): the path to load the model.
        """

        # load the model
        clf = joblib.load(model)

        # get matrix of instances
        candidates = self.instances.keys()
        X = [self.instances[u] for u in candidates]

        # classify candidates
        y = clf.predict_proba(X)

        for i, candidate in enumerate(candidates):
            self.weights[candidate] = y[i][1]
       
















