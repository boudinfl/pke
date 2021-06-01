# -*- coding: utf-8 -*-
# Author: Florian Boudin
# Date: 09-10-2018

"""TopicCoRank supervised keyphrase extraction model.


TopicCoRank is a supervised graph-based ranking approach to keyphrase
extraction that operates over a unified graph that connects two graphs: the
former represents the document and the latter captures how keyphrases are
associated with each other in the training data. The model is described in:

* Adrien Bougouin, Florian Boudin, and Beatrice Daille.
  Keyphrase annotation with graph co-ranking
  *Proceedings of the COLINGs*, pages 2945–2955, 2016.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from pke.unsupervised import TopicRank
from pke.utils import load_references

from itertools import combinations
from collections import defaultdict
import logging
import networkx as nx
import math


class TopicCoRank(TopicRank):
    """TopicCoRank keyphrase extraction model.

    Parameterized example::

        import pke
        import string
        from nltk.corpus import stopwords

        # 1. create a TopicCoRank extractor.
        extractor = pke.unsupervised.TopicCoRank()

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
        """Redefining initializer for TopicCoRank."""

        super(TopicCoRank, self).__init__()

        self.domain_to_integer = {}

        self.topic_to_integer = {}

    def build_topic_graph(self):
        """Re-define the topic graph construction method.

        Build the topic graph by connecting topics if their candidates
        co-occur in the same sentence. Edges are weighted by the number of
        oc-occurrences.
        """

        # adding the nodes to the graph
        self.graph.add_nodes_from(range(len(self.topics)), src="topic")

        # loop through the topics to connect the nodes
        for i, j in combinations(range(len(self.topics)), 2):

            # for each candidate in topic i
            for c_i in self.topics[i]:

                # for each candidate in topic j
                for c_j in self.topics[j]:

                    weight = len(
                        set(self.candidates[c_i].sentence_ids).intersection(
                            self.candidates[c_j].sentence_ids))

                    if weight > 0:
                        if not self.graph.has_edge(i, j):
                            self.graph.add_edge(i, j, weight=0, type="in")
                        self.graph[i][j]['weight'] += weight

    def unify_with_domain_graph(self, input_file, excluded_file=None):
        """Unify the domain graph, built from a reference file, with the topic
        graph, built from a document.

        Args:
            input_file (str): path to the reference file.
            excluded_file (str): file to exclude (for leave-one-out
                cross-validation), defaults to None.
        """

        if input_file.endswith('.json'):
            references = load_references(input_file=input_file,
                                         language=self.language)
        else:
            logging.warning("{} is not a reference file".format(input_file))
            pass

        # remove excluded file if needed
        if excluded_file is not None:
            if excluded_file not in references:
                logging.warning("{} is not in reference".format(excluded_file))
            else:
                logging.info("{} removed from reference".format(excluded_file))
                del references[excluded_file]

        # initialize the topic_to_integer map
        for i, topic in enumerate(self.topics):
            for candidate in topic:
                self.topic_to_integer[candidate] = i

        offset = len(self.topics)

        # loop through the doc_ids
        for doc_id in references:

            # for each pair of gold keyphrases
            for gold_1, gold_2 in combinations(references[doc_id], 2):

                # adding nodes to the graph
                if gold_1 not in self.domain_to_integer:
                    self.domain_to_integer[gold_1] = offset
                    self.graph.add_node(offset, src="domain", candidate=gold_1)

                    # checking for out edges with topics
                    if gold_1 in self.topic_to_integer:
                        self.graph.add_edge(self.domain_to_integer[gold_1],
                                            self.topic_to_integer[gold_1],
                                            weight=0, type="out")

                    offset += 1

                if gold_2 not in self.domain_to_integer:
                    self.domain_to_integer[gold_2] = offset
                    self.graph.add_node(offset, src="domain", candidate=gold_2)

                    # checking for out edges with topics
                    if gold_2 in self.topic_to_integer:
                        self.graph.add_edge(self.domain_to_integer[gold_2],
                                            self.topic_to_integer[gold_2],
                                            weight=0, type="out")

                    offset += 1

                node_1 = self.domain_to_integer[gold_1]
                node_2 = self.domain_to_integer[gold_2]

                # add/update the edge
                if not self.graph.has_edge(node_1, node_2):
                    self.graph.add_edge(node_1, node_2, weight=0, type="in")
                self.graph[node_1][node_2]['weight'] += 1

    def candidate_weighting(self,
                            input_file=None,
                            excluded_file=None,
                            lambda_t=0.1,
                            lambda_k=0.5,
                            nb_iter=100,
                            convergence_threshold=0.001):
        """Weight candidates using the co-ranking formulae.

        Args:
            input_file (str): path to the reference file.
            excluded_file (str): file to exclude (for leave-one-out
                cross-validation), defaults to None.
            lambda_t(float): lambda for topics used in the co-ranking formulae,
                defaults to 0.1.
            lambda_k(float): lambda for keyphrases used in the co-ranking
                formulae, defaults to 0.5.
            nb_iter (int): maximum number of iterations, defaults to 100.
            convergence_threshold (float): early stop threshold, defaults to
                0.001.
        """

        # compute topics
        self.topic_clustering()

        # build graph
        self.build_topic_graph()

        # unify with domain graph
        self.unify_with_domain_graph(input_file=input_file,
                                     excluded_file=excluded_file)

        logging.info("resulting graph is {} nodes".format(
                                                    len(self.graph.nodes())))

        weights = [1.0] * len(self.graph.nodes)

        # pre-compute the inner/outer normalizations
        inner_norms = [0.0] * len(self.graph.nodes)
        outer_norms = [0.0] * len(self.graph.nodes)

        for j in self.graph.nodes():
            inner_norm = 0
            outer_norm = 0
            for k in self.graph.neighbors(j):
                if self.graph[j][k]['type'] == "in":
                    inner_norm += self.graph[j][k]["weight"]
                else:
                    outer_norm += 1
            inner_norms[j] = inner_norm
            outer_norms[j] = outer_norm

        # ranking nodes in the graph using co-ranking
        converged = False
        while nb_iter > 0 and not converged:

            converged = True

            #logging.info("{} iter left".format(nb_iter))

            # save the weights
            w = weights.copy()

            for i in self.graph.nodes():

                # compute inner/outer recommendations
                r_in = 0.0
                r_out = 0.0
                for j in self.graph.neighbors(i):

                    # inner recommendation
                    if self.graph[i][j]['type'] == "in":
                        r_in += (self.graph[i][j]["weight"] * w[j]) / \
                                inner_norms[j]

                    # outer recommendation
                    else:
                        r_out += w[j] / outer_norms[j]

                # compute the new weight
                if self.graph.nodes[i]["src"] == "topic":
                    weights[i] = (1 - lambda_t) * r_out
                    weights[i] += lambda_t * r_in
                else:
                    weights[i] = (1 - lambda_k) * r_out
                    weights[i] += lambda_k * r_in

                # check for non convergence
                if math.fabs(weights[i] - w[i]) > convergence_threshold:
                    converged = False

            nb_iter -= 1

        # get the final ranking
        for i in self.graph.nodes():

            # if it is a topic candidate
            if self.graph.nodes[i]["src"] == "topic":

                # get the candidates from the topic
                topic = self.topics[i]

                # get the offsets of the topic candidates
                offsets = [self.candidates[t].offsets[0] for t in topic]

                first = offsets.index(min(offsets))
                self.weights[topic[first]] = weights[i]

            # otherwise it is a keyphrase from the domain
            else:

                gold = self.graph.nodes[i]["candidate"]

                # check if it is acceptable, i.e. if it is directly or
                # transitively connected to a topic
                connected = False
                for j in self.graph.neighbors(i):
                    if self.graph.nodes[j]["src"] == "topic":
                        connected = True
                        break
                    for k in self.graph.neighbors(j):
                        if self.graph.nodes[k]["src"] == "topic":
                            connected = True
                            break
                    if connected:
                        break

                if connected:
                    if gold in self.weights:
                        self.weights[gold] = max(self.weights[gold], weights[i])
                    else:
                        self.weights[gold] = weights[i]
