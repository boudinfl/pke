# -*- coding: utf-8 -*-
# Author: Florian Boudin
# Date: 09-10-2018

"""Kea keyphrase extraction model.

Supervised approach to keyphrase extraction described in:

* Thuy Dung Nguyen and Minh-Thang Luong.
  WINGNUS: Keyphrase Extraction Utilizing Document Logical Structure.
  *Proceedings of SemEval*, pages 166–169, 2010.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import re
import string

import numpy as np
from sklearn.externals import joblib
from sklearn.naive_bayes import MultinomialNB

from pke.supervised.api import SupervisedLoadFile
from pke.utils import load_document_frequency_file


class WINGNUS(SupervisedLoadFile):
    """WINGNUS keyphrase extraction model.

    Parameterized example::

        import pke

        # 1. create a WINGNUS extractor.
        extractor = pke.supervised.WINGNUS(input_file='path/to/input.xml')

        # 2. load the content of the document.
        extractor.read_document(format='corenlp')

        # 3. select simplex noun phrases as candidates.
        extractor.candidate_selection()

        # 4. classify candidates as keyphrase or not keyphrase.
        df = pke.load_document_frequency_file(input_file='path/to/df.tsv.gz')
        model_file = 'path/to/wingnus_model'
        extractor.candidate_weighting(self, model_file=model_file, df=df)

        # 5. get the 10-highest scored candidates as keyphrases
        keyphrases = extractor.get_n_best(n=10)

    """

    def __init__(self):
        """Redefining initializer for WINGNUS.
        """

        super(WINGNUS, self).__init__()

    def candidate_selection(self,
                            NP='^((JJ|NN) ){,2}NN$',
                            NP_IN_NP='^((JJ|NN) )?NN IN ((JJ|NN) )?NN$',
                            **kwargs):
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
        self.candidate_filtering(
            stoplist=list(string.punctuation) +
                     ['-lrb-', '-rrb-', '-lcb-', '-rcb-', '-lsb-', '-rsb-'])

        # filter non-simplex noun phrases
        # Python 2/3 compatible
        for k in list(self.candidates):

            # get the candidate
            v = self.candidates[k]

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
        """Extract features for each candidate.

        Args:
            df (dict): document frequencies, the number of documents should be
                specified using the "--NB_DOC--" key.
            training (bool): indicates whether features are computed for the
                training set for computing IDF weights, defaults to false.
            features_set (list): the set of features to use, defaults to
                [1, 4, 6].

        """

        # define the default features_set
        if features_set is None:
            features_set = [1, 4, 6]

        # initialize default document frequency counts if none provided
        if df is None:
            df = load_document_frequency_file(self._df_counts, delimiter='\t')

        # initialize the number of documents as --NB_DOC--
        N = df.get('--NB_DOC--', 0) + 1
        if training:
            N -= 1

        # find the maximum offset
        maximum_offset = float(sum([s.length for s in self.sentences]))

        # loop through the candidates
        for k, v in self.candidates.items():

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
            stoplist = self.stoplist
            for i in range(len(v.lexical_form)):
                for j in range(i, min(len(v.lexical_form), i + 3)):
                    sub_words = v.lexical_form[i:j + 1]
                    sub_string = ' '.join(sub_words)

                    # skip if substring is fullstring
                    if sub_string == ' '.join(v.lexical_form):
                        continue

                    # skip if substring contains a stopword
                    if set(sub_words).intersection(stoplist):
                        continue

                    # check whether the substring occurs "as it"
                    if sub_string in self.candidates:

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
            feature_array.append(v.offsets[0] / maximum_offset)

            # [F5] -> relative last occurrence
            feature_array.append(v.offsets[-1] / maximum_offset)

            # [F6] -> length of phrases in words
            feature_array.append(len(v.lexical_form))

            # [F7] -> typeface
            feature_array.append(0)

            # extract information from sentence meta information
            meta = [self.sentences[sid].meta for sid in v.sentence_ids]

            # extract meta information of candidate
            sections = [u['section'] for u in meta if 'section' in u]
            types = [u['type'] for u in meta if 'type' in u]

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
            feature_array.append(types.count('sectionHeader') +
                                 types.count('subsectionHeader') +
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
            self.instances[k] = np.array([feature_array[i - 1] for i
                                          in features_set])

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
        # with open(model_file, 'wb') as f:
        #     pickle.dump(clf, f)
