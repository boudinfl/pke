# -*- coding: utf-8 -*-

""" Unsupervised keyphrase extraction models. """

import string
import networkx as nx
import numpy as np
import math
from .base import LoadFile
from itertools import combinations
from nltk.corpus import stopwords
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist


class TfIdf(LoadFile):
    """ TF*IDF keyphrase extraction model. """

    def candidate_selection(self):
        """ Select 1-3 grams as keyphrase candidates. """

        # select ngrams from 1 to 3 grams
        self.ngram_selection(n=3)

        # filter candidates containing punctuation marks
        self.candidate_filtering(stoplist=list(string.punctuation) +
                                 ['-lrb-', '-rrb-', '-lcb-', '-rcb-', '-lsb-',
                                  '-rsb-'])


    def candidate_weighting(self, df=None, N=144):
        """ Candidate weighting function using document frequencies.
            Args:
                df (dict): document frequencies.
                N (int): the number of documents for computing IDF, defaults to
                    144 as in the SemEval dataset.
        """

        # loop throught the candidates
        for k, v in self.candidates.items():

            # get candidate document frequency
            candidate_df = 1 + df.get(k, 0)

            # compute the idf score
            idf = math.log(float(N+1) / float(candidate_df), 2)

            # add the idf score to the weights container
            self.weights[k] = len(v.surface_forms) * idf


class KPMiner(LoadFile):
    """ KP-Miner keyphrase extraction model. 

        This model was published and described in:

          * Samhaa R. El-Beltagy and Ahmed Rafea, KP-Miner: Participation in 
            SemEval-2, *Proceedings of the 5th International Workshop on 
            Semantic Evaluation*, pages 190-193, 2010.
    """

    def candidate_selection(self, lasf=3, cutoff=400):
        """ The candidate selection as described in the KP-Miner paper.

            Args:
                lasf (int): least allowable seen frequency, defaults to 3.
                cutoff (int): the number of words after which candidates are
                    filtered out, defaults to 400.
        """

        # select ngrams from 1 to 5 grams
        self.ngram_selection(n=5)

        # filter candidates containing stopwords or punctuation marks
        self.candidate_filtering(stoplist=stopwords.words(self.language) +
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


    def candidate_weighting(self, df=None, N=144, sigma=3.0, alpha=2.3):
        """ Candidate weight calculation as described in the KP-Miner paper.

            w = tf * idf * B * P_f

            with:
                B = N_d / (P_d * alpha) and B = min(sigma, B)
                N_d = the number of all candidate terms
                P_d = number of candidates whose length exceeds one
                P_f = 1

            Args:
                df (dict): document frequencies.
                N (int): the number of documents for computing IDF, defaults to
                    144 as in the SemEval dataset.
                sigma (int): parameter for boosting factor, defaults to 3.0.
                alpha (int): parameter for boosting factor, defaults to 2.3.
        """

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
            idf = math.log(float(N+1) / float(candidate_df), 2)

            self.weights[k] = len(v.surface_forms) * B * idf


class SingleRank(LoadFile):
    """ The SingleRank keyphrase extraction model.

        This model was published and described in:

          * Xiaojun Wan and Jianguo Xiao, CollabRank: Towards a Collaborative 
            Approach to Single-Document Keyphrase Extraction, *Proceedings of 
            the 22nd International Conference on Computational Linguistics 
            (Coling 2008)*, pages 969-976, 2008.
    """

    def __init__(self, input_file):
        """ Redefining initializer for SingleRank. """

        super(SingleRank, self).__init__(input_file)

        self.graph = nx.Graph()
        """ The word graph. """


    def build_word_graph(self, window=10, pos=None):
        """ Build the word graph from the document. """

        # loop through the sentences to build the graph
        for sentence in self.sentences:

            # add the nodes to the graph
            for j, node in enumerate(sentence.stems):
                if sentence.pos[j][:2] in pos:
                    self.graph.add_node(node)

            # add the edges between the nodes
            for j, node_1 in enumerate(sentence.stems):
                for k in range(j+1, min(j+window, sentence.length)):
                    node_2 = sentence.stems[k]
                    if not self.graph.has_edge(node_1, node_2):
                        self.graph.add_edge(node_1, node_2, weight=0)
                    self.graph[node_1][node_2]['weight'] += 1.0


    def candidate_selection(self):
        """ The candidate selection as described in the SingleRank paper. """

        # select sequence of adjectives and nouns
        self.sequence_selection(pos=['NN', 'NNS', 'NNP', 'NNPS',
                                     'JJ', 'JJR', 'JJS'])

        # filter candidates containing stopwords or punctuation marks
        self.candidate_filtering(stoplist=stopwords.words(self.language) +
                                 list(string.punctuation) +
                                 ['-lrb-', '-rrb-', '-lcb-', '-rcb-', '-lsb-',
                                  '-rsb-'])


    def candidate_weighting(self):
        """ Candidate weight calculation using random walk. """

        # build the word graph
        self.build_word_graph(window=10, pos=['JJ', 'NN'])

        # compute the word scores using random walk
        w = nx.pagerank_scipy(self.graph)

        # loop throught the candidates
        for k in self.candidates.keys():
            tokens = self.candidates[k].lexical_form
            self.weights[k] = sum([w[t] for t in tokens]) / len(tokens)


class TopicRank(LoadFile):
    """ The TopicRank keyphrase extraction model. 

        This model was published and described in:

          * Adrien Bougouin, Florian Boudin and Béatrice Daille, TopicRank: 
            Graph-Based Topic Ranking for Keyphrase Extraction, *Proceedings of
            the Sixth International Joint Conference on Natural Language
            Processing*, pages 543-551, 2013.
    """

    def __init__(self, input_file):
        """ Redefining initializer for TopicRank. """

        super(TopicRank, self).__init__(input_file)

        self.graph = nx.Graph()
        """ The topic graph. """

        self.topics = []
        """ The topic container. """


    def candidate_selection(self):
        """ The candidate selection as described in the TopicRank paper. """

        # select sequence of adjectives and nouns
        self.sequence_selection(pos=['NN', 'NNS', 'NNP', 'NNPS',
                                     'JJ', 'JJR', 'JJS'])

        # filter candidates containing stopwords or punctuation marks
        self.candidate_filtering(stoplist=stopwords.words(self.language) +
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


    def topic_clustering(self, threshold=0.25, method='average'):
        """ Clustering candidates into topics.

            Args:
                threshold (int): the minimum similarity for clustering, defaults
                    to 0.25.
                method (str): the linkage method, defaults to average.
        """

        # vectorize the candidates
        candidates, X = self.vectorize_candidates()

        # compute the distance matrix
        Y = pdist(X, 'jaccard')

        # compute the clusters
        Z = linkage(Y, method=method)

        # form flat clusters
        clusters = fcluster(Z, t=1.0-threshold, criterion='distance')

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
                            self.graph[i][j]['weight'] += 1.0 / abs(p_i-p_j)


    def candidate_weighting(self):
        """ Candidate weight calculation using random walk. """

        # cluster the candidates
        self.topic_clustering()

        # build the topic graph
        self.build_topic_graph()

        # compute the word scores using random walk
        w = nx.pagerank_scipy(self.graph)

        # loop throught the topics
        for i, topic in enumerate(self.topics):

            # get first occuring candidate from topic
            offsets = [self.candidates[t].offsets[0] for t in topic]
            first = offsets.index(min(offsets))
            self.weights[topic[first]] = w[i]



