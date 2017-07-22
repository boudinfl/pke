# -*- coding: utf-8 -*-

""" Unsupervised keyphrase extraction models. """

from __future__ import division
from __future__ import absolute_import

import string
import networkx as nx
import numpy as np
import math

from .base import LoadFile
from .utils import load_document_frequency_file

from itertools import combinations
from collections import defaultdict

from nltk.corpus import stopwords

from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
from sklearn.cluster import spectral_clustering


class TfIdf(LoadFile):
    """ TF*IDF keyphrase extraction model. """

    def candidate_selection(self, stoplist=None):
        """ Select 1-3 grams as keyphrase candidates. """

        # select ngrams from 1 to 3 grams
        self.ngram_selection(n=3)

        # filter candidates containing stopwords or punctuation marks
        if stoplist == None:
            stoplist = stoplist=stopwords.words(self.language) 
        self.candidate_filtering(stoplist=stoplist +
                                 list(string.punctuation) +
                                 ['-lrb-', '-rrb-', '-lcb-', '-rcb-', '-lsb-',
                                  '-rsb-'])

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
        """

        # select ngrams from 1 to 5 grams
        self.ngram_selection(n=5)

        # filter candidates containing stopwords or punctuation marks
        if stoplist == None:
            stoplist = stoplist=stopwords.words(self.language) 
        self.candidate_filtering(stoplist=stoplist +
                                 list(string.punctuation) +
                                 ['-lrb-', '-rrb-', '-lcb-', '-rcb-', '-lsb-',
                                  '-rsb-'])

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


class SingleRank(LoadFile):
    """ The SingleRank keyphrase extraction model.

        This model was published and described in:

          * Xiaojun Wan and Jianguo Xiao, CollabRank: Towards a Collaborative
            Approach to Single-Document Keyphrase Extraction, *Proceedings of
            the 22nd International Conference on Computational Linguistics
            (Coling 2008)*, pages 969-976, 2008.
    """

    def __init__(self, input_file, language='english'):
        """ Redefining initializer for SingleRank. """

        super(SingleRank, self).__init__(input_file=input_file,
                                         language=language)

        self.graph = nx.Graph()
        """ The word graph. """


    def build_word_graph(self, window=10, pos=None):
        """ Build the word graph from the document.

            Args:
                window (int): the window within the sentence for connecting two
                    words in the graph, defaults to 10.
                pos (set): the set of valid pos for words to be considered as
                    nodes in the graph, defaults to (NN, NNS, NNP, NNPS, JJ,
                    JJR, JJS).
        """

        # define default pos tags set
        if pos is None:
            pos = set(['NN', 'NNS', 'NNP', 'NNPS', 'JJ', 'JJR', 'JJS'])

        # flatten document and initialize nodes 
        sequence = []

        for sentence in self.sentences:
            for j, node in enumerate(sentence.stems):
                if sentence.pos[j] in pos:
                    self.graph.add_node(node)
                sequence.append((node, sentence.pos[j]))

        # loop through sequence to build the edges in the graph
        for j, node_1 in enumerate(sequence):
            for k in range(j+1, min(j+window, len(sequence))):
                node_2 = sequence[k]
                if node_1[1] in pos and node_2[1] in pos:
                    if not self.graph.has_edge(node_1[0], node_2[0]):
                        self.graph.add_edge(node_1[0], node_2[0], weight=0)
                    self.graph[node_1[0]][node_2[0]]['weight'] += 1.0


    def candidate_selection(self, pos=None, stoplist=None):
        """ The candidate selection as described in the SingleRank paper.

            Args:
                pos (set): the set of valid POS tags, defaults to (NN, NNS,
                    NNP, NNPS, JJ, JJR, JJS).
        """

        # define default pos tags set
        if pos is None:
            pos = set(['NN', 'NNS', 'NNP', 'NNPS', 'JJ', 'JJR', 'JJS'])

        # select sequence of adjectives and nouns
        self.longest_pos_sequence_selection(valid_pos=pos)

        # filter candidates containing stopwords or punctuation marks
        if stoplist == None:
            stoplist = stoplist=stopwords.words(self.language) 
        self.candidate_filtering(stoplist=stoplist +
                                 list(string.punctuation) +
                                 ['-lrb-', '-rrb-', '-lcb-', '-rcb-', '-lsb-',
                                  '-rsb-'])

    def candidate_weighting(self, window=10, pos=None, normalized=False):
        """ Candidate weight calculation using random walk.

            Args:
                window (int): the window within the sentence for connecting two
                    words in the graph, defaults to 10.
                pos (set): the set of valid pos for words to be considered as
                    nodes in the graph, defaults to (NN, NNS, NNP, NNPS, JJ,
                    JJR, JJS).
                normalized (False): normalize keyphrase score by their length,
                    defaults to False
        """

        # define default pos tags set
        if pos is None:
            pos = set(['NN', 'NNS', 'NNP', 'NNPS', 'JJ', 'JJR', 'JJS'])

        # build the word graph
        self.build_word_graph(window=window, pos=pos)

        # compute the word scores using random walk
        w = nx.pagerank_scipy(self.graph)

        # loop through the candidates
        for k in self.candidates.keys():
            tokens = self.candidates[k].lexical_form
            self.weights[k] = sum([w[t] for t in tokens])
            if normalized:
                self.weights[k] /= len(tokens)


class TopicRank(LoadFile):
    """ The TopicRank keyphrase extraction model.

        This model was published and described in:

          * Adrien Bougouin, Florian Boudin and Béatrice Daille, TopicRank:
            Graph-Based Topic Ranking for Keyphrase Extraction, *Proceedings of
            the Sixth International Joint Conference on Natural Language
            Processing*, pages 543-551, 2013.
    """

    def __init__(self, input_file, language='english'):
        """ Redefining initializer for TopicRank. """

        super(TopicRank, self).__init__(input_file=input_file,
                                        language=language)

        self.graph = nx.Graph()
        """ The topic graph. """

        self.topics = []
        """ The topic container. """


    def candidate_selection(self, pos=None, stoplist=None):
        """ The candidate selection as described in the SingleRank paper.

            Args:
                pos (set): the set of valid POS tags, defaults to (NN, NNS,
                    NNP, NNPS, JJ, JJR, JJS).
        """

        # define default pos tags set
        if pos is None:
            pos = set(['NN', 'NNS', 'NNP', 'NNPS', 'JJ', 'JJR', 'JJS'])

        # select sequence of adjectives and nouns
        self.longest_pos_sequence_selection(valid_pos=pos)

        # filter candidates containing stopwords or punctuation marks
        if stoplist == None:
            stoplist = stoplist=stopwords.words(self.language) 
        self.candidate_filtering(stoplist=stoplist +
                                 list(string.punctuation) +
                                 ['-lrb-', '-rrb-', '-lcb-', '-rcb-', '-lsb-',
                                  '-rsb-'])


    def vectorize_candidates(self):
        """ Vectorize the keyphrase candidates.

            Returns:
                C (list): the list of candidates.
                X (matrix): vectorized representation of the candidates.
        """

        # build the vocabulary, i.e. setting the vector dimensions
        dim = set([])
        for k, v in self.candidates.iteritems():
            for w in v.lexical_form:
                dim.add(w)
        dim = list(dim)

        # vectorize the candidates
        C = self.candidates.keys()
        X = np.zeros((len(C), len(dim)))
        for i, k in enumerate(C):
            for w in self.candidates[k].lexical_form:
                X[i, dim.index(w)] += 1

        return C, X


    def topic_clustering(self, threshold=0.74, method='average'):
        """ Clustering candidates into topics.

            Args:
                threshold (float): the minimum similarity for clustering,
                    defaults to 0.74, i.e. more than 1/4 of stem overlap
                    similarity.
                method (str): the linkage method, defaults to average.
        """

        # vectorize the candidates
        candidates, X = self.vectorize_candidates()

        # compute the distance matrix
        Y = pdist(X, 'jaccard')

        # compute the clusters
        Z = linkage(Y, method=method)

        # form flat clusters
        clusters = fcluster(Z, t=threshold, criterion='distance')

        # for each cluster id
        for cluster_id in range(1, max(clusters)+1):
            self.topics.append([candidates[j] for j in range(len(clusters))
                                if clusters[j] == cluster_id])


    def build_topic_graph(self):
        """ Build the topic graph. """

        # adding the nodes to the graph
        self.graph.add_nodes_from(range(len(self.topics)))

        # loop through the topics to connect the nodes
        for i, j in combinations(range(len(self.topics)), 2):
            self.graph.add_edge(i, j, weight=0)
            for c_i in self.topics[i]:
                for c_j in self.topics[j]:
                    for p_i in self.candidates[c_i].offsets:
                        for p_j in self.candidates[c_j].offsets:
                            gap = abs(p_i - p_j)
                            if p_i < p_j:
                                gap -= len(self.candidates[c_i].lexical_form)-1
                            if p_j < p_i:
                                gap -= len(self.candidates[c_j].lexical_form)-1
                            self.graph[i][j]['weight'] += 1.0 / gap


    def candidate_weighting(self, threshold=0.74, method='average',
                            heuristic=None):
        """ Candidate weight calculation using random walk.

            Args:
                threshold (float): the minimum similarity for clustering,
                    defaults to 0.74.
                method (str): the linkage method, defaults to average.
                heuristic (str): the heuristic for selecting the best candidate
                    for each topic, defaults to first occurring candidate. Other
                    options are 'frequent' (most frequent candidate, position
                    is used for ties).
        """

        # cluster the candidates
        self.topic_clustering(threshold=threshold, method=method)

        # build the topic graph
        self.build_topic_graph()

        # compute the word scores using random walk
        w = nx.pagerank_scipy(self.graph)

        # loop throught the topics
        for i, topic in enumerate(self.topics):

            # get the offsets of the topic candidates
            offsets = [self.candidates[t].offsets[0] for t in topic]

            # get first candidate from topic
            if heuristic == 'frequent':

                # get frequencies for each candidate within the topic
                freq = [len(self.candidates[t].surface_forms) for t in topic]

                # get the indexes of the most frequent candidates
                indexes = [j for j, f in enumerate(freq) if f == max(freq)]

                # offsets of the indexes
                indexes_offsets = [offsets[j] for j in indexes]
                most_frequent = indexes_offsets.index(min(indexes_offsets))
                self.weights[topic[most_frequent]] = w[i]

            else:
                first = offsets.index(min(offsets))
                self.weights[topic[first]] = w[i]

