# -*- coding: utf-8 -*-
# Author: Florian Boudin
# Date: 09-11-2018

"""PositionRank keyphrase extraction model.

Graph-based ranking approach to keyphrase extraction described in:

* Corina Florescu and Cornelia Caragea.
  PositionRank: An Unsupervised Approach to Keyphrase Extraction from Scholarly
  Documents.
  *In proceedings of ACL*, pages 1105-1115, 2017.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from pke.unsupervised import SingleRank

import networkx as nx
from collections import defaultdict


class PositionRank(SingleRank):
    """PositionRank keyphrase extraction model. 

    Parameterized example::

        import pke

        # 1. create a PositionRank extractor.
        extractor = pke.unsupervised.PositionRank(input_file='path/to/input')

        # 2. load the content of the document.
        extractor.read_document(format='corenlp')

        # 3. select the noun phrases up to 3 words as keyphrase candidates.
        grammar = "NP: {<JJ.*>*<NN.*>+}"
        extractor.candidate_selection(grammar=grammar, maximum_word_number=3)

        # 4. weight the candidates using the sum of their word's scores that are
        #    computed using random walk biaised with the position of the words
        #    in the document. In the graph, nodes are words (nouns and
        #    adjectives only) that are connected if they occur in a window of
        #    10 words.
        pos = set(['NN', 'NNS', 'NNP', 'NNPS', 'JJ', 'JJR', 'JJS'])
        extractor.candidate_weighting(window=10, pos=pos)

        # 5. get the 10-highest scored candidates as keyphrases
        keyphrases = extractor.get_n_best(n=10)

    """

    def __init__(self):
        """Redefining initializer for PositionRank.
        """

        super(PositionRank, self).__init__()

        self.positions = defaultdict(float)

    def candidate_selection(self, grammar=None, maximum_word_number=3, **kwargs):
        """Candidate selection heuristic.

        Keyphrase candidates are noun phrases that match the regular expression
        (adjective)*(noun)+, of length up to three.

        Args:
            grammar (str): grammar defining POS patterns of NPs, defaults to 
                "NP: {<JJ.*>*<NN.*>+}".
            maximum_word_number (int): the maximum number of words allowed for
                keyphrase candidates, defaults to 3.

        """

        # define default NP grammar if none provided
        if grammar is None:
            grammar = "NP:{<JJ.*>*<NN.*>+}"

        # select sequence of adjectives and nouns
        self.grammar_selection(grammar=grammar)

        # filter candidates greater than 3 words
        for k in list(self.candidates):
            v = self.candidates[k]
            if len(v.lexical_form) > maximum_word_number:
                del self.candidates[k]

    def build_word_graph(self, window=10, pos=None):
        """Build the word graph from the document.

        Args:
            window (int): the window within the sentence for connecting two
                words in the graph, defaults to 10.
            pos (set): the set of valid pos for words to be considered as nodes
                in the graph, defaults to (NN, NNS, NNP, NNPS, JJ, JJR, JJS).
        """

        # define default pos tags set
        if pos is None:
            pos = {'NN', 'NNS', 'NNP', 'NNPS', 'JJ', 'JJR', 'JJS'}

        # flatten document and initialize nodes 
        sequence = []

        for sentence in self.sentences:
            for j, node in enumerate(sentence.stems):
                if sentence.pos[j] in pos:
                    self.graph.add_node(node)
                sequence.append((node, sentence.pos[j]))

                # compute the inverse position priors
                self.positions[node] += 1 / (len(sequence)+1)

        # loop through sequence to build the edges in the graph
        for j, node_1 in enumerate(sequence):
            for k in range(j+1, min(j+window, len(sequence))):
                node_2 = sequence[k]
                if node_1[1] in pos and node_2[1] in pos \
                   and node_1[0] != node_2[0]:
                    if not self.graph.has_edge(node_1[0], node_2[0]):
                        self.graph.add_edge(node_1[0], node_2[0], weight=0)
                    self.graph[node_1[0]][node_2[0]]['weight'] += 1

    def candidate_weighting(self, window=10, pos=None, normalized=False):
        """Candidate weight calculation using random walk.

        Args:
            window (int): the window within the sentence for connecting two
                words in the graph, defaults to 10.
            pos (set): the set of valid pos for words to be considered as nodes
                in the graph, defaults to (NN, NNS, NNP, NNPS, JJ, JJR, JJS).
            normalized (False): normalize keyphrase score by their length,
                defaults to False.
        """

        # define default pos tags set
        if pos is None:
            pos = {'NN', 'NNS', 'NNP', 'NNPS', 'JJ', 'JJR', 'JJS'}

        # build the word graph
        self.build_word_graph(window=window, pos=pos)

        # normalize cummulated inverse positions
        norm = sum(self.positions.values())
        for word in self.positions:
            self.positions[word] /= norm

        # compute the word scores using biaised random walk
        w = nx.pagerank(G=self.graph,
                        alpha=0.85,
                        personalization=self.positions,
                        max_iter=100,
                        weight='weight')

        # loop through the candidates
        for k in self.candidates.keys():
            tokens = self.candidates[k].lexical_form
            self.weights[k] = sum([w[t] for t in tokens])
            if normalized:
                self.weights[k] /= len(tokens)

