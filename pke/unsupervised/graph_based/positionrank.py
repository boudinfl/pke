# -*- coding: utf-8 -*-
# Author: Florian Boudin
# Date: 09-11-2018

"""PositionRank keyphrase extraction model.

PositionRank is an unsupervised model for keyphrase extraction from scholarly
documents that incorporates information from all positions of a word's
occurrences into a biased PageRank. The model is described in:

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

        # define the valid Part-of-Speeches to occur in the graph
        pos = {'NOUN', 'PROPN', 'ADJ'}

        # define the grammar for selecting the keyphrase candidates
        grammar = "NP: {<ADJ>*<NOUN|PROPN>+}"

        # 1. create a PositionRank extractor.
        extractor = pke.unsupervised.PositionRank()

        # 2. load the content of the document.
        extractor.load_document(input='path/to/input',
                                language='en',
                                normalization=None)

        # 3. select the noun phrases up to 3 words as keyphrase candidates.
        extractor.candidate_selection(grammar=grammar,
                                      maximum_word_number=3)

        # 4. weight the candidates using the sum of their word's scores that are
        #    computed using random walk biaised with the position of the words
        #    in the document. In the graph, nodes are words (nouns and
        #    adjectives only) that are connected if they occur in a window of
        #    10 words.
        extractor.candidate_weighting(window=10,
                                      pos=pos)

        # 5. get the 10-highest scored candidates as keyphrases
        keyphrases = extractor.get_n_best(n=10)

    """

    def __init__(self):
        """Redefining initializer for PositionRank."""

        super(PositionRank, self).__init__()

        self.positions = defaultdict(float)
        """Container the sums of word's inverse positions."""

    def candidate_selection(self,
                            grammar=None,
                            maximum_word_number=3):
        """Candidate selection heuristic using a syntactic PoS pattern for
        noun phrase extraction.

        Keyphrase candidates are noun phrases that match the regular expression
        (adjective)*(noun)+, of length up to three.

        Args:
            grammar (str): grammar defining POS patterns of NPs, defaults to 
                "NP: {<ADJ>*<NOUN|PROPN>+}".
            maximum_word_number (int): the maximum number of words allowed for
                keyphrase candidates, defaults to 3.
        """

        if grammar is None:
            grammar = "NP:{<ADJ>*<NOUN|PROPN>+}"

        # select sequence of adjectives and nouns
        self.grammar_selection(grammar=grammar)

        # filter candidates greater than 3 words
        for k in list(self.candidates):
            v = self.candidates[k]
            if len(v.lexical_form) > maximum_word_number:
                del self.candidates[k]

    def build_word_graph(self, window=10, pos=None):
        """Build the graph representation of the document.

        In the graph, nodes are words that passes a Part-of-Speech filter. Two
        nodes are connected if the words corresponding to these nodes co-occur
        within a `window` of contiguous tokens. The weight of an edge is
        computed based on the co-occurrence count of the two words within a
        `window` of successive tokens.

        Args:
            window (int): the window within the sentence for connecting two
                words in the graph, defaults to 10.
            pos (set): the set of valid pos for words to be considered as nodes
                in the graph, defaults to ('NOUN', 'PROPN', 'ADJ').
        """

        if pos is None:
            pos = {'NOUN', 'PROPN', 'ADJ'}

        # flatten document as a sequence of only valid (word, position) tuples
        text = []
        for i, sentence in enumerate(self.sentences):
            shift = sum([s.length for s in self.sentences[0:i]])
            for j, word in enumerate(sentence.stems):
                if sentence.pos[j] in pos:
                    text.append((word, shift+j))

        # add nodes to the graph
        self.graph.add_nodes_from([word for (word, position) in text])

        # add edges to the graph
        for i, (node1, position1) in enumerate(text):
            j = i+1
            while j < len(text) and (text[j][1] - position1) < window:
                node2, position2 = text[j]
                if node1 != node2:
                    if not self.graph.has_edge(node1, node2):
                        self.graph.add_edge(node1, node2, weight=0)
                    self.graph[node1][node2]['weight'] += 1
                j = j + 1

        # compute the sums of the word's inverse positions
        for word, position in text:
            self.positions[word] += 1 / (position + 1)

    def candidate_weighting(self, window=10, pos=None, normalized=False):
        """Candidate weight calculation using a biased PageRank.

        Args:
            window (int): the window within the sentence for connecting two
                words in the graph, defaults to 10.
            pos (set): the set of valid pos for words to be considered as nodes
                in the graph, defaults to ('NOUN', 'PROPN', 'ADJ').
            normalized (False): normalize keyphrase score by their length,
                defaults to False.
        """

        if pos is None:
            pos = {'NOUN', 'PROPN', 'ADJ'}

        # build the word graph
        self.build_word_graph(window=window,
                              pos=pos)

        # normalize cumulated inverse positions
        norm = sum(self.positions.values())
        for word in self.positions:
            self.positions[word] /= norm

        # compute the word scores using biased random walk
        w = nx.pagerank(G=self.graph,
                        alpha=0.85,
                        tol=0.0001,
                        personalization=self.positions,
                        weight='weight')

        # loop through the candidates
        for k in self.candidates.keys():
            tokens = self.candidates[k].lexical_form
            self.weights[k] = sum([w.get(t, 0.0) for t in tokens])
            if normalized:
                self.weights[k] /= len(tokens)

