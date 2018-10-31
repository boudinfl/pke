# -*- coding: utf-8 -*-
# Authors: Ygor Gallina, Florian Boudin
# Date: 10-18-2018

"""TextRank keyphrase extraction model.

Graph-based ranking approach to keyphrase extraction described in:

* Rada Mihalcea and Paul Tarau.
  TextRank: Bringing Order into Texts
  *In Proceedings of EMNLP*, 2004.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import string

import networkx as nx

from pke.base import LoadFile


class TextRank(LoadFile):
    """TextRank keyphrase extraction model.
    This model is essentially the same as SingleRank as its core is the
    PageRank algorithm. In this model the edge's weight is not used.

    Parameterized example::

        import pke
        import string
        from nltk.corpus import stopwords

        # 1. create a TextRank extractor.
        extractor = pke.unsupervised.TextRank()

        # 2. load the content of the document.
        extractor.load_document(input='path/to/input.xml')

        # 3. select the the longest sequences of nouns and adjectives, that do
        #    not contain punctuation marks or stopwords as candidates.
        pos = {'NOUN', 'PROPN', 'ADJ'}
        stoplist = list(string.punctuation)
        stoplist += ['-lrb-', '-rrb-', '-lcb-', '-rcb-', '-lsb-', '-rsb-']
        stoplist += stopwords.words('english')
        extractor.candidate_selection(pos=pos, stoplist=stoplist)

        # 4. weight the candidates using the sum of their word's scores
        #    that are computed using random walk. In the graph, nodes are words
        #    (nouns and adjectives only) that are connected if they occur in a
        #    window of 2 words.
        #    OR
        #    use option `top` to extract candidates according
        #    to (Mihalcea and Tarau, 2004, 3.1)
        extractor.candidate_weighting(window=2, pos=pos)

        # 5. get the 10-highest scored candidates as keyphrases
        keyphrases = extractor.get_n_best(n=10)

    """

    def __init__(self):
        """Redefining initializer for TextRank.
        """

        super(TextRank, self).__init__()

        self.graph = nx.Graph()
        """ The word graph. """
        self.pagerank_output = None

    def candidate_selection(self, pos=None, stoplist=None):
        """ The candidate selection as described in the TextRank paper.

            Args:
                pos (set): the set of valid POS tags, defaults to ('NOUN',
                    'PROPN', 'ADJ').
                stoplist (list): the stoplist for filtering candidates, defaults
                    to the nltk stoplist. Words that are punctuation marks from
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
        self.candidate_filtering(
            stoplist=list(string.punctuation) +
            ['-lrb-', '-rrb-', '-lcb-', '-rcb-', '-lsb-', '-rsb-'] +
            stoplist)

    def build_word_graph(self, window=2, pos=None):
        """Build the word graph from the document.

        Args:
            window (int): the window within the sentence for connecting two
                words in the graph, defaults to 2.
            pos (set): the set of valid pos for words to be considered as nodes
                in the graph, defaults to ('NOUN', 'PROPN', 'ADJ').

        """

        # define default pos tags set
        if pos is None:
            pos = {'NOUN', 'PROPN', 'ADJ'}

        # flatten document and initialize nodes
        sequence = []

        for sentence in self.sentences:
            for j, node in enumerate(sentence.stems):
                if sentence.pos[j] in pos:
                    self.graph.add_node(node)
                    sequence.append((node, sentence.pos[j]))
                # the coocurence window skips stopwords
                # "... systems of linear ...", it is unclear about keeping
                # unvalid pos and is not sentence aware
                # "... numbers. Criteria ..." (cf. Figure 2)
                # the graph building is not very detailed

        # loop through sequence to build the edges in the graph
        for j, node_1 in enumerate(sequence):
            for k in range(j + 1, min(j + window, len(sequence))):
                node_2 = sequence[k]
                if node_1[1] in pos and \
                   node_2[1] in pos and \
                   node_1[0] != node_2[0]:
                    if not self.graph.has_edge(node_1[0], node_2[0]):
                        self.graph.add_edge(node_1[0], node_2[0])

    def candidate_weighting(self, window=2, pos=None,
                            normalized=False, top=None):
        """Candidate ranking using random walk.

        Args:
            window (int): the window within the sentence for connecting two
                words in the graph, defaults to 2.
            pos (set): the set of valid pos for words to be considered as nodes
                in the graph, defaults to ('NOUN', 'PROPN', 'ADJ').
            normalized (False): normalize keyphrase score by their length,
                defaults to False.
            top (float): percent of top vertices to keep for keyterm
                generation as described in (Mihalcea and Tarau, 2004, 3.1)

        """

        # define default pos tags set
        if pos is None:
            pos = {'NOUN', 'PROPN', 'ADJ'}

        # build the word graph
        self.build_word_graph(window=window, pos=pos)

        # compute the word scores using random walk
        self.pagerank_output = nx.pagerank_scipy(
            self.graph, alpha=0.85, tol=0.0001, weight=None)

        w = self.pagerank_output

        if top is not None:
            # keeping only the top keywords
            to_keep = math.floor(len(self.graph.node) * top)
            w = sorted(self.pagerank_output.items(),
                       key=lambda x: x[1],
                       reverse=True)
            w = w[:to_keep]  # filtering
            w = {k: v for k, v in w}

            # post-processing : creating key terms
            self.longest_keyword_sequence_selection(w.keys())

        # Weigh the candidates
        for k in self.candidates.keys():
            tokens = self.candidates[k].lexical_form
            self.weights[k] = sum([w[t] for t in tokens])
            if normalized:
                self.weights[k] /= len(tokens)
