# -*- coding: utf-8 -*-
# Author: Florian Boudin
# Date: 09-10-2018

"""TopicRank keyphrase extraction model.

Graph-based ranking approach to keyphrase extraction described in:

* Adrien Bougouin, Florian Boudin and Béatrice Daille.
  TopicRank: Graph-Based Topic Ranking for Keyphrase Extraction.
  *In proceedings of IJCNLP*, pages 543-551, 2013.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import string
from itertools import combinations

import networkx as nx
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist

from pke.base import LoadFile


class TopicRank(LoadFile):
    """TopicRank keyphrase extraction model.

    Parameterized example::

        import pke
        import string
        from nltk.corpus import stopwords

        # 1. create a TopicRank extractor.
        extractor = pke.unsupervised.TopicRank()

        # 2. load the content of the document.
       extractor.load_document(input='path/to/input.xml')

        # 3. select the longest sequences of nouns and adjectives, that do
        #    not contain punctuation marks or stopwords as candidates.
        pos = {'NOUN', 'PROPN', 'ADJ'}
        stoplist = list(string.punctuation)
        stoplist += ['-lrb-', '-rrb-', '-lcb-', '-rcb-', '-lsb-', '-rsb-']
        stoplist += stopwords.words('english')
        extractor.candidate_selection(pos=pos, stoplist=stoplist)

        # 4. build topics by grouping candidates with HAC (average linkage,
        #    threshold of 1/4 of shared stems). Weight the topics using random
        #    walk, and select the first occuring candidate from each topic.
        extractor.candidate_weighting(threshold=0.74, method='average')

        # 5. get the 10-highest scored candidates as keyphrases
        keyphrases = extractor.get_n_best(n=10)

    """

    def __init__(self):
        """Redefining initializer for TopicRank.
        """

        super(TopicRank, self).__init__()

        self.graph = nx.Graph()
        """ The topic graph. """

        self.topics = []
        """ The topic container. """

    def candidate_selection(self, pos=None, stoplist=None):
        """Selects longest sequences of nouns and adjectives as keyphrase
        candidates.

        Args:
            pos (set): the set of valid POS tags, defaults to ('NOUN',
                'PROPN', 'ADJ').
            stoplist (list): the stoplist for filtering candidates, defaults to
                the nltk stoplist. Words that are punctuation marks from
                string.punctuation are not allowed.

        """

        # define default pos tags set
        if pos is None:
            pos = {'NOUN', 'PROPN', 'ADJ'}

        # select sequence of adjectives and nouns
        self.longest_pos_sequence_selection(valid_pos=pos)

        # initialize stoplist list if not provided
        if stoplist is None:
            stoplist = self.stoplist

        # filter candidates containing stopwords or punctuation marks
        self.candidate_filtering(stoplist=list(string.punctuation) +
                                          ['-lrb-', '-rrb-', '-lcb-', '-rcb-', '-lsb-', '-rsb-'] +
                                          stoplist)

    def vectorize_candidates(self):
        """Vectorize the keyphrase candidates.

        Returns:
            C (list): the list of candidates.
            X (matrix): vectorized representation of the candidates.

        """

        # build the vocabulary, i.e. setting the vector dimensions
        dim = set([])
        # for k, v in self.candidates.iteritems():
        # iterate Python 2/3 compatible
        for (k, v) in self.candidates.items():
            for w in v.lexical_form:
                dim.add(w)
        dim = list(dim)

        # vectorize the candidates Python 2/3 + sort for random issues
        C = list(self.candidates)  # .keys()
        C.sort()

        X = np.zeros((len(C), len(dim)))
        for i, k in enumerate(C):
            for w in self.candidates[k].lexical_form:
                X[i, dim.index(w)] += 1

        return C, X

    def topic_clustering(self, threshold=0.74, method='average'):
        """Clustering candidates into topics.

        Args:
            threshold (float): the minimum similarity for clustering, defaults
                to 0.74, i.e. more than 1/4 of stem overlap similarity.
            method (str): the linkage method, defaults to average.

        """

        # handle document with only one candidate
        if len(self.candidates) == 1:
            self.topics.append([list(self.candidates)[0]])
            return

        # vectorize the candidates
        candidates, X = self.vectorize_candidates()

        # compute the distance matrix
        Y = pdist(X, 'jaccard')

        # compute the clusters
        Z = linkage(Y, method=method)

        # form flat clusters
        clusters = fcluster(Z, t=threshold, criterion='distance')

        # for each topic identifier
        for cluster_id in range(1, max(clusters) + 1):
            self.topics.append([candidates[j] for j in range(len(clusters))
                                if clusters[j] == cluster_id])

    def build_topic_graph(self):
        """Build topic graph."""

        # adding the nodes to the graph
        self.graph.add_nodes_from(range(len(self.topics)))

        # loop through the topics to connect the nodes
        for i, j in combinations(range(len(self.topics)), 2):
            self.graph.add_edge(i, j, weight=0.0)
            for c_i in self.topics[i]:
                for c_j in self.topics[j]:
                    for p_i in self.candidates[c_i].offsets:
                        for p_j in self.candidates[c_j].offsets:
                            gap = abs(p_i - p_j)
                            if p_i < p_j:
                                gap -= len(self.candidates[c_i].lexical_form) - 1
                            if p_j < p_i:
                                gap -= len(self.candidates[c_j].lexical_form) - 1
                            self.graph[i][j]['weight'] += 1.0 / gap

    def candidate_weighting(self,
                            threshold=0.74,
                            method='average',
                            heuristic=None):
        """Candidate ranking using random walk.

        Args:
            threshold (float): the minimum similarity for clustering, defaults
                to 0.74.
            method (str): the linkage method, defaults to average.
            heuristic (str): the heuristic for selecting the best candidate for
                each topic, defaults to first occurring candidate. Other options
                are 'frequent' (most frequent candidate, position is used for
                ties).

        """
        if not self.candidates:
            return

        # cluster the candidates
        self.topic_clustering(threshold=threshold, method=method)

        # build the topic graph
        self.build_topic_graph()

        # compute the word scores using random walk
        w = nx.pagerank_scipy(self.graph, alpha=0.85, weight='weight')

        # loop through the topics
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
                # Choosing the first occuring most frequent candidate
                most_frequent = offsets.index(min(indexes_offsets))
                self.weights[topic[most_frequent]] = w[i]

            else:
                first = offsets.index(min(offsets))
                self.weights[topic[first]] = w[i]
