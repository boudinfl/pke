# -*- coding: utf-8 -*-

"""Statistical keyphrase extraction models."""

from __future__ import absolute_import
from __future__ import division

import string
import math
import numpy
import re

from itertools import combinations
from collections import defaultdict

from pke.base import LoadFile
from pke.utils import load_document_frequency_file
from nltk.corpus import stopwords
from nltk.metrics import edit_distance


class TfIdf(LoadFile):
    """TF*IDF keyphrase extraction model.

    Parameterized example::

        import string
        import pke

        # 1. create a TfIdf extractor.
        extractor = pke.unsupervised.TfIdf(input_file='path/to/input.xml')

        # 2. load the content of the document.
        extractor.read_document(format='corenlp')

        # 3. select {1-3}-grams not containing punctuation marks as candidates.
        n = 3
        stoplist = list(string.punctuation)
        stoplist += ['-lrb-', '-rrb-', '-lcb-', '-rcb-', '-lsb-', '-rsb-']
        extractor.candidate_selection(n=n, stoplist=stoplist)

        # 4. weight the candidates using a `tf` x `idf`
        df = pke.load_document_frequency_file(input_file='path/to/df.tsv.gz')
        extractor.candidate_weighting(df=df)

        # 5. get the 10-highest scored candidates as keyphrases
        keyphrases = extractor.get_n_best(n=10)

    """

    def candidate_selection(self, n=3, stoplist=None):
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
            stoplist = []

        # filter candidates containing punctuation marks
        self.candidate_filtering(stoplist=list(string.punctuation) +
                                 ['-lrb-', '-rrb-', '-lcb-', '-rcb-', '-lsb-',
                                  '-rsb-'] +
                                  stoplist)


    def candidate_weighting(self, df=None):
        """Candidate weighting function using document frequencies.

        Args:
            df (dict): document frequencies, the number of documents should be
                specified using the "--NB_DOC--" key.
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
    """KP-Miner keyphrase extraction model.

    This model was published and described in:

      * Samhaa R. El-Beltagy and Ahmed Rafea, KP-Miner: Participation in
        SemEval-2, *Proceedings of the 5th International Workshop on
        Semantic Evaluation*, pages 190-193, 2010.

    Parameterized example::

        import pke

        # 1. create a KPMiner extractor. 
        extractor = pke.unsupervised.KPMiner(input_file='path/to/input.xml',
                                             language='english')

        # 2. load the content of the document.
        extractor.read_document(format='corenlp')

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


class YAKE(LoadFile):
    """YAKE keyphrase extraction model.

    This model was published and described in:

      * Ricardo Campos, Vítor Mangaravite, Arian Pasquali, Alípio Mário Jorge,
        Célia Nunes and Adam Jatowt.
        YAKE! Collection-Independent Automatic Keyword Extractor.
        *Proceedings of ECIR 2018.*, pages 806-810.

    Parameterized example::

        import pke
        from nltk.corpus import stopwords

        # 1. create a YAKE extractor.
        extractor = pke.unsupervised.YAKE(input_file='path/to/input.xml')

        # 2. load the content of the document.
        extractor.read_document(format='corenlp')

        # 3. select {1-3}-grams not containing punctuation marks and not
        #    beginning/ending with a stopword as candidates.
        stoplist = stopwords.words('english')
        extractor.candidate_selection(n=3, stoplist=stoplist)

        # 4. weight the candidates usinh YAKE weighting scheme, a window (in
             words) for computing left/right contexts can be specified.
        window = 3
        extractor.candidate_weighting(window=window)

        # 5. get the 10-highest scored candidates as keyphrases.
        #    redundant keyphrases are removed from the output using levenshtein
        #    distance and a threshold.
        threshold = 0.8
        keyphrases = extractor.get_n_best(n=10, threshold=threshold)

    """

    def __init__(self, input_file=None, language='english'):
        """ Redefining initializer for YAKE. """

        super(YAKE, self).__init__(input_file=input_file, language=language)

        self.words = defaultdict(set)
        """ A word container. """


    def candidate_selection(self, n=3, stoplist=None):
        """Select 1-3 grams as keyphrase candidates. Candidates beginning or
        ending with a stopword are filtered out. Words that are punctuation
        marks from `string.punctuation` are not allowed.

        Args:
            n (int): the n-gram length, defaults to 3.
            stoplist (list): the stoplist for filtering candidates, defaults to
                the nltk stoplist.
        """

        # punctuation marks
        punctuation = list(string.punctuation)
        punctuation += ['-lrb-', '-rrb-', '-lcb-', '-rcb-', '-lsb-', '-rsb-']

        # initialize stoplist list if not provided
        self.stoplist = stoplist
        if stoplist is None:
            self.stoplist = stopwords.words(self.language)

        # loop through the sentences
        for i, sentence in enumerate(self.sentences):

            # limit the maximum n for short sentence
            skip = min(n, sentence.length)

            # compute the offset shift for the sentence
            shift = sum([s.length for s in self.sentences[0:i]])

            # generate the ngrams
            for j in range(sentence.length):
                for k in range(j+1, min(j+1+skip, sentence.length+1)):

                    words = sentence.words[j:k]
                    lowercase_words = [w.lower() for w in words]
                    stems = sentence.stems[j:k]
                    pos = sentence.pos[j:k]
                    lexical_form = ' '.join(lowercase_words)

                    # add the lowercased words
                    for offset, w in enumerate(lowercase_words):
                        if re.search('\w', w):
                            self.words[w].add((shift+j+offset, shift, i,
                                               words[offset]))

                    # skip candidate if it contains punctuation marks
                    if len( set(lowercase_words) & set(punctuation) ):
                        continue

                    # skip candidate if it starts/ends with a stopword
                    elif lowercase_words[0] in self.stoplist or\
                         lowercase_words[-1] in self.stoplist:
                        continue

                    # add the ngram to the candidate container
                    self.candidates[lexical_form].surface_forms.append(words)
                    self.candidates[lexical_form].lexical_form = stems
                    self.candidates[lexical_form].pos_patterns.append(pos)
                    self.candidates[lexical_form].offsets.append(shift+j)
                    self.candidates[lexical_form].sentence_ids.append(i)

        # filter candidates containing punctuation marks
        self.candidate_filtering()


    def candidate_weighting(self, window=3):
        """Candidate weight calculation as described in the YAKE paper.

        Args:
            window (int): the size in words of the sliding window used for
                computing the co-occurrence matrix, defaults to 2.
        """

        # get a container for 'individual' word weights
        word_weights = defaultdict(float)

        # get a static word index
        words = list(self.words)

        # get frequency statistics
        frequencies = [len(self.words[t]) for t in self.words]
        valid_frequencies = [ len(self.words[t]) for t in self.words\
                              if t not in self.stoplist ]
        mean_freq = numpy.mean(valid_frequencies)
        std_freq = numpy.std(valid_frequencies)
        max_freq = max(frequencies)

        # compute Left/Right co-occurrence contexts
        WL = defaultdict(list)
        WR = defaultdict(list)
        for i, j in combinations(range(len(words)), 2):
            for t_i in self.words[words[i]]:
                for t_j in self.words[words[j]]:
                    # skip if they do not occur within the same sentence
                    if t_i[2] != t_j[2]:
                        continue

                    if math.fabs(t_i[0]-t_j[0]) < window:
                        if t_i[0] > t_j[0]:
                            WL[words[i]].append(words[j])
                            WR[words[j]].append(words[i])
                        else:
                            WL[words[j]].append(words[i])
                            WR[words[i]].append(words[j])            

        features = defaultdict(dict)

        # Loop through the words to compute features and weights
        for word in words:

            # compute the term frequency
            features[word]['TF'] = len(self.words[word])

            # compute the uppercase/acronym term frequencies
            features[word]['TF_A'] = 0
            features[word]['TF_U'] = 0
            for (offset, shift, sent_id, surface_form) in self.words[word]:
                if surface_form.isupper() and len(word) > 1:
                    features[word]['TF_A'] += 1
                elif surface_form[0].isupper() and offset != shift:
                    features[word]['TF_U'] += 1

            # compute the casing feature
            features[word]['CASING'] = max(features[word]['TF_A'],
                                           features[word]['TF_U'])
            features[word]['CASING'] /= 1.0 + math.log(features[word]['TF'])

            # compute the position feature
            sentence_ids = list(set([t[2] for t in self.words[word]]))
            features[word]['POS'] = math.log(3+numpy.median(sentence_ids))
            features[word]['POS'] = math.log(features[word]['POS'])

            # compute the frequency feature
            features[word]['FREQ'] = features[word]['TF']
            features[word]['FREQ'] /= (mean_freq + std_freq)

            # compute the word relatedness feature
            features[word]['WL'] = 0.0
            if len(WL[word]):
                features[word]['WL'] = len(set(WL[word])) / len(WL[word])
            features[word]['PL'] = len(set(WL[word])) / max_freq

            features[word]['WR'] = 0.0
            if len(WR[word]):
                features[word]['WR'] = len(set(WR[word])) / len(WR[word])
            features[word]['PR'] = len(set(WR[word])) / max_freq

            features[word]['REL'] = 0.5 + ((features[word]['WL'] * \
                      (features[word]['TF'] / max_freq)) + features[word]['PL'])
            features[word]['REL'] += 0.5 + ((features[word]['WR'] * \
                      (features[word]['TF'] / max_freq)) + features[word]['PR'])

            # compute the DifSentence feature
            features[word]['DIF'] = len(set(sentence_ids)) / len(self.sentences)

            # compute the word weight from its features
            word_weights[word] = features[word]['REL'] * features[word]['POS']
            word_weights[word] /= ( features[word]['CASING'] +\
                               (features[word]['FREQ']/features[word]['REL']) +\
                               (features[word]['DIF']/features[word]['REL']) )

        # loop throught the candidates to compute their weights
        for k, v in self.candidates.items():
            candidate_weights = [ word_weights[word.lower()] for word in\
                                  v.surface_forms[0] ]

            self.weights[k] = numpy.prod(candidate_weights)
            self.weights[k] /= len(v.offsets) * (1 + sum(candidate_weights))


    def is_redundant(self, candidate, prev, threshold=0.8):
        """ Test if one candidate is redundant with respect to a list of already
            selected candidates. A candidate is considered redundant if its
            levenshtein distance, with another candidate that is ranked higher
            in the list, is greater than a threshold.

            Args:
                candidate (str): the lexical form of the candidate.
                prev (list): the list of already selected candidates (lexical
                    forms).
                threshold (float): the threshold used when computing the
                    levenshtein distance, defaults to 0.8.
        """

        # loop through the already selected candidates
        for prev_candidate in prev:
            dist = edit_distance(candidate, prev_candidate)
            dist /= max(len(candidate), len(prev_candidate))
            if (1.0 - dist) > threshold:
                return True
        return False


    def get_n_best(self, n=10, redundancy_removal=True, stemming=True,
                   threshold=0.8):
        """ Returns the n-best candidates given the weights.

            Args:
                n (int): the number of candidates, defaults to 10.
                redundancy_removal (bool): whether redundant keyphrases are
                    filtered out from the n-best list using levenshtein
                    distance, defaults to True.
                stemming (bool): whether to extract stems or surface forms
                    (lowercased, first occurring form of candidate), default to
                    stems.
                threshold (float): the threshold used when computing the
                    levenshtein distance, defaults to 0.8.
        """

        # sort candidates by ascending weight
        best = sorted(self.weights, key=self.weights.get, reverse=False)

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

        # get the list of best candidates as (lexical form, weight) tuples
        n_best = [(u, self.weights[u]) for u in best[:min(n, len(best))]]

        # replace with surface forms is no stemming
        if stemming:
            n_best = [(' '.join(self.candidates[u].lexical_form).lower(),
                       self.weights[u]) for u in best[:min(n, len(best))]]

        # return the list of best candidates
        return n_best

