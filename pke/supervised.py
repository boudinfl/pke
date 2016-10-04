# -*- coding: utf-8 -*-

""" Supervised keyphrase extraction models. """

from __future__ import division

import re
import math
import string

from .base import LoadFile

import numpy as np

from nltk.corpus import stopwords

import pickle

from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
# from sklearn.utils import shuffle
# from sklearn.linear_model import LogisticRegression


class SupervisedLoadFile(LoadFile):
    """ The SupervisedLoadFile class that provides extra base functions for
        supervised models. """

    def __init__(self, input_file=None, language='english'):
        """ Redefining initializer. """

        super(SupervisedLoadFile, self).__init__(input_file=input_file,
                                                 language=language)

        self.instances = {}
        """ The instances container. """


    def feature_scaling(self):
        """ Scale features to [0,1]. """

        candidates = self.instances.keys()
        X = [self.instances[u] for u in candidates]
        X = MinMaxScaler().fit_transform(X)
        for i, candidate in enumerate(candidates):
            self.instances[candidate] = X[i]


    def classify_candidates(self, model):
        """ Classify the candidates as keyphrase or not keyphrase.

            Args:
                model (str): the path to load the model in pickle format.
        """

        # load the model
        with open(model, 'rb') as f:
            clf = pickle.load(f)

        # get matrix of instances
        candidates = self.instances.keys()
        X = [self.instances[u] for u in candidates]

        # classify candidates
        y = clf.predict_proba(X)

        for i, candidate in enumerate(candidates):
            self.weights[candidate] = y[i][1]


class Kea(SupervisedLoadFile):
    """ Kea keyphrase extraction model. """

    def __init__(self, input_file=None, language='english'):
        """ Redefining initializer for Kea. """

        super(Kea, self).__init__(input_file=input_file, language=language)


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

            # delete if candidate contains a stopword in first/last position
            words = [u.lower() for u in v.surface_forms[0]]
            if words[0] in stoplist or words[-1] in stoplist:
                del self.candidates[k]


    def feature_extraction(self, df=None, training=False):
        """ Extract features (tf*idf, first occurrence and length) for each
            candidate.

            Args:
                df (dict): document frequencies, the number of documents should
                    be specified using the "--NB_DOC--" key.
                training (bool): indicates whether features are computed for the
                    training set for computing IDF weights, defaults to false.
        """

        # initialize the number of documents as --NB_DOC--
        N = df.get('--NB_DOC--', 0) + 1
        if training:
            N -= 1

        # find the maximum offset
        maximum_offset = float(sum([s.length for s in self.sentences]))

        for k, v in self.candidates.iteritems():

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
        with open(model_file, 'wb') as f:
            pickle.dump(clf, f)


class WINGNUS(SupervisedLoadFile):
    """ WINGNUS keyphrase extraction model. """


    def __init__(self, input_file=None, language='english'):
        """ Redefining initializer for WINGNUS. """

        super(WINGNUS, self).__init__(input_file=input_file, language=language)


    def candidate_selection(self,
                            NP='^((JJ|NN) ){,2}NN$',
                            NP_IN_NP='^((JJ|NN) )?NN IN ((JJ|NN) )?NN$'):
        """ Select noun phrases (NP) and NP containing a preprositional phrase
            (NP IN NP) as keyphrase candidates.

            Args:
                NP (str): the pattern for noun phrases, defaults to
                    '^((JJ|NN) ){,2}NN$'.
                simplex_NP (str): the pattern for filtering simplex noun
                    phrases, defaults to '^((JJ|NN) )?NN IN ((JJ|NN) )?NN$'.
        """

        # select ngrams from 1 to 4 grams
        self.ngram_selection(n=4)

        # filter candidates containing punctuation marks
        self.candidate_filtering(stoplist=list(string.punctuation) +
                                 ['-lrb-', '-rrb-', '-lcb-', '-rcb-', '-lsb-',
                                  '-rsb-'])

        # filter non-simplex noun phrases
        for k, v in self.candidates.items():

            # valid surface forms container
            valid_surface_forms = []

            # loop through the surface forms
            for i in range(len(v.pos_patterns)):
                pattern = ' '.join([u[:2] for u in v.pos_patterns[i]])
                if re.search(NP, pattern) or re.search(NP_IN_NP, pattern):
                    valid_surface_forms.append(i)

            # delete candidate if not valid
            if not valid_surface_forms:
                del self.candidates[k]

            # otherwise update the candidate data
            else:
                self.candidates[k].surface_forms = [v.surface_forms[i] for i
                                                    in valid_surface_forms]
                self.candidates[k].offsets = [v.offsets[i] for i
                                              in valid_surface_forms]
                self.candidates[k].pos_patterns = [v.pos_patterns[i] for i
                                                   in valid_surface_forms]


    def feature_extraction(self, df=None, training=False, features_set=None):
        """ Extract features for each candidate.

            Args:
                df (dict): document frequencies, the number of documents should
                    be specified using the "--NB_DOC--" key.
                training (bool): indicates whether features are computed for the
                    training set for computing IDF weights, defaults to false.
                features_set (list): the set of features to use, defaults to
                    [1, 4, 6].
        """

        # define the default features_set
        if features_set is None:
            features_set = [1, 4, 6]

        # initialize the number of documents as --NB_DOC--
        N = df.get('--NB_DOC--', 0) + 1
        if training:
            N -= 1

        # find the maximum offset
        maximum_offset = float(sum([s.length for s in self.sentences]))

        for k, v in self.candidates.iteritems():

            # initialize features array
            feature_array = []

            # get candidate document frequency
            candidate_df = 1 + df.get(k, 0)

            # hack for handling training documents
            if training and candidate_df > 1:
                candidate_df -= 1

            # compute the tf*idf of the candidate
            idf = math.log(N / candidate_df, 2)

            # [F1] TF*IDF
            feature_array.append(len(v.surface_forms) * idf)

            # [F2] -> TF
            feature_array.append(len(v.surface_forms))

            # [F3] -> term frequency of substrings
            tf_of_substrings = 0
            stoplist = stopwords.words(self.language)
            for i in range(len(v.lexical_form)):
                for j in range(i, min(len(v.lexical_form), i+3)):
                    sub_words = v.lexical_form[i:j+1]
                    sub_string = ' '.join(sub_words)

                    # skip if substring is fullstring
                    if sub_string == ' '.join(v.lexical_form):
                        continue

                    # skip if substring contains a stopword
                    if set(sub_words).intersection(stoplist):
                        continue

                    # check whether the substring occurs "as it"
                    if self.candidates.has_key(sub_string):

                        # loop throught substring offsets
                        for offset_1 in self.candidates[sub_string].offsets:
                            is_included = False
                            for offset_2 in v.offsets:
                                if offset_1 >= offset_2 and \
                                   offset_1 <= offset_2 + len(v.lexical_form):
                                    is_included = True
                            if not is_included:
                                tf_of_substrings += 1

            feature_array.append(tf_of_substrings)

            # [F4] -> relative first occurrence
            feature_array.append(v.offsets[0]/maximum_offset)

            # [F5] -> relative last occurrence
            feature_array.append(v.offsets[-1]/maximum_offset)

            # [F6] -> length of phrases in words
            feature_array.append(len(v.lexical_form))

            # [F7] -> typeface
            feature_array.append(0)

            # extract information from candidate meta information
            sections = [u['section'] for u in v.meta if u.has_key('section')]
            types = [u['type'] for u in v.meta if u.has_key('type')]

            # [F8] -> Is in title
            feature_array.append('title' in sections)

            # [F9] -> TitleOverlap
            feature_array.append(0)

            # [F10] -> Header
            feature_array.append('sectionHeader' in types or
                                 'subsectionHeader' in types or
                                 'subsubsectionHeader' in types)

            # [F11] -> abstract
            feature_array.append('abstract' in sections)

            # [F12] -> introduction
            feature_array.append('introduction' in sections)

            # [F13] -> related work
            feature_array.append('related work' in sections)

            # [F14] -> conclusions
            feature_array.append('conclusions' in sections)

            # [F15] -> HeaderF
            feature_array.append(types.count('sectionHeader')+
                                 types.count('subsectionHeader')+
                                 types.count('subsubsectionHeader'))

            # [F11] -> abstractF
            feature_array.append(sections.count('abstract'))

            # [F12] -> introductionF
            feature_array.append(sections.count('introduction'))

            # [F13] -> related workF
            feature_array.append(sections.count('related work'))

            # [F14] -> conclusionsF
            feature_array.append(sections.count('conclusions'))

            # add the features to the instance container
            self.instances[k] = np.array([feature_array[i-1] for i \
                                          in features_set])

        # scale features
        self.feature_scaling()


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
        with open(model_file, 'wb') as f:
            pickle.dump(clf, f)


class SEERLAB(SupervisedLoadFile):
    """ SEERLAB keyphrase extraction model. """

    def __init__(self, input_file=None, language='english'):
        """ Redefining initializer for SEERLAB. """

        super(SEERLAB, self).__init__(input_file=input_file, language=language)


    def candidate_selection(self,
                            dblp_candidates=None,
                            mf_unigrams=30,
                            mf_non_unigrams=30):
        """ Select keyphrase candidates.

            Args:
                dblp_candidates (list): valid candidates according to the list
                    of candidates extracted from the dblp titles.
                mf_unigrams (int): the number of most frequent unigrams to
                    include in the candidates, defaults to 30.
                mf_non_unigrams (int): the number of most frequent non-unigrams
                    to include in the candidates, defaults to 30.
        """

        # select ngrams from 1 to 4 grams
        self.ngram_selection(n=4)

        # filter candidates containing stopwords or punctuation marks
        self.candidate_filtering(stoplist=stopwords.words(self.language) +
                                 list(string.punctuation) +
                                 ['-lrb-', '-rrb-', '-lcb-', '-rcb-', '-lsb-',
                                  '-rsb-'])

        # build the sets of unigrams, non-unigrams and acronyms
        unigrams = list()
        non_unigrams = list()
        acronyms = list()

        # loop through the candidates
        for k, v in self.candidates.items():

            # adding unigram
            if len(v.lexical_form) == 1:
                unigrams.append((len(v.surface_forms), k))

            # adding non unigram
            else:
                non_unigrams.append((len(v.surface_forms), k))

            # adding acronym
            form = ' '.join(v.surface_forms[0])
            if form.isupper() and len(form) > 1:
                acronyms.append(k)

        # first populate valid candidates with acronyms
        valid_candidates = set(acronyms)

        # add the most frequent unigrams
        valid_candidates.update(set([elem[0] for elem in \
            sorted(unigrams, reverse=True)[:min(len(unigrams), mf_unigrams)]]))

        # add the most frequent non unigrams
        valid_candidates.update(set([elem[1] for elem in \
            sorted(non_unigrams, reverse=True)[:min(len(non_unigrams),
                                                    mf_non_unigrams)]]))

        # filter candidates according the the most frequent sets
        for k, v in self.candidates.items():
            if k not in valid_candidates:
                del self.candidates[k]

        # loop through sentences to extract candidates occuring in dblp
        for i, sentence in enumerate(self.sentences):

            skip = min(4, sentence.length)
            shift = sum([s.length for s in self.sentences[0:i]])
            j = 0

            while j < sentence.length:
                for k in range(min(j+skip, sentence.length+1), j, -1):

                    surface_form = sentence.words[j:k]
                    norm_form = sentence.stems[j:k]
                    pos_pattern = sentence.pos[j:k]
                    key = ' '.join(norm_form)

                    if key in dblp_candidates and key not in self.candidates:

                        self.candidates[key].surface_forms.append(surface_form)
                        self.candidates[key].lexical_form = norm_form
                        self.candidates[key].offsets.append(shift+j)
                        self.candidates[key].pos_patterns.append(pos_pattern)

                        j = k -1
                        break
                j += 1


    def feature_extraction(self, df=None, training=False):
        """ Extract features (tf*idf, first occurrence and length) for each
            candidate.

            Args:
                df (dict): document frequencies, the number of documents should
                    be specified using the "--NB_DOC--" key.
                training (bool): indicates whether features are computed for the
                    training set for computing IDF weights, defaults to false.
        """

        # initialize the number of documents as --NB_DOC--
        N = df.get('--NB_DOC--', 0) + 1
        if training:
            N -= 1

        for k, v in self.candidates.iteritems():

            # get candidate document frequency
            candidate_df = 1 + df.get(k, 0)

            # hack for handling training documents
            if training and candidate_df != 1:
                candidate_df -= 1

            # compute the tf*idf of the candidate
            idf = math.log(N / candidate_df, 2)

            # test if candidate is an acronym
            is_acronym = 0
            for surface_form in v.surface_forms:
                form = ' '.join(surface_form)
                if form.isupper() and len(form) > 1:
                    is_acronym = 1

            # compute frequency in title (defined as first sentence)
            # max_offset = self.sentences[0].length
            # tf_title = len([u for u in v.offsets if u <= max_offset])

            # add the features to the instance container
            self.instances[k] = np.array([len(v.lexical_form),               # N
                                          is_acronym,                     # ACRO
                                          len(v.surface_forms),         # TF_doc
                                          candidate_df,                     # DF
                                          len(v.surface_forms) * idf])   # TFIDF


        # scale features
        # self.feature_scaling()


    @staticmethod
    def train(training_instances, training_classes, model_file):
        """ Train a Random Forest classifier and store the model in a file.

            Args:
                training_instances (list): list of features.
                training_classes (list): list of binary values.
                model_file (str): the model output file.
        """

        clf = RandomForestClassifier(n_estimators=200,
                                     max_features=3,
                                     class_weight='balanced')

        # Down sampling the instances to 1:7

        # decompose instances into positives/negatives
        # positives = []
        # negatives = []
        # for i in range(len(training_instances)):
        #     if training_classes[i] == 1:
        #         positives.append(training_instances[i])
        #     else:
        #         negatives.append(training_instances[i])


        # np.random.shuffle(negatives)

        # training_instances = negatives[:min(len(positives)*7, len(negatives))]
        # training_classes = [0]*len(training_instances)
        # training_instances.extend(positives)
        # training_classes.extend([1]*len(positives))

        # X, y = shuffle(training_instances, training_classes, random_state=0)

        # fit the data
        clf.fit(training_instances, training_classes)
        # clf.fit(X, y)
        with open(model_file, 'wb') as f:
            pickle.dump(clf, f)

        # print clf.feature_importances_
        # print selector.support_
        # print selector.ranking_
