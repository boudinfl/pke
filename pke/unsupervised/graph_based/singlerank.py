# -*- coding: utf-8 -*-
# Author: Florian Boudin
# Date: 09-11-2018

"""SingleRank keyphrase extraction model.

Graph-based ranking approach to keyphrase extraction described in:

* Xiaojun Wan and Jianguo Xiao.
  CollabRank: Towards a Collaborative Approach to Single-Document Keyphrase
  Extraction.
  *In proceedings of the COLING*, pages 969-976, 2008.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import string

import networkx as nx

from pke.base import LoadFile


class SingleRank(LoadFile):
    """SingleRank keyphrase extraction model.

    Parameterized example::

        import pke
        import string
        from nltk.corpus import stopwords

        # 1. create a SingleRank extractor.
        extractor = pke.unsupervised.SingleRank(input_file='path/to/input.xml')

        # 2. load the content of the document.
        extractor.read_document(format='corenlp')

        # 3. select the the longest sequences of nouns and adjectives, that do
        #    not contain punctuation marks or stopwords as candidates.
        pos = set(['NN', 'NNS', 'NNP', 'NNPS', 'JJ', 'JJR', 'JJS'])
        stoplist = list(string.punctuation)
        stoplist += ['-lrb-', '-rrb-', '-lcb-', '-rcb-', '-lsb-', '-rsb-']
        stoplist += stopwords.words('english')
        extractor.candidate_selection(pos=pos, stoplist=stoplist)

        # 4. weight the candidates using the sum of their word's scores that are
        #    computed using random walk. In the graph, nodes are words (nouns
        #    and adjectives only) that are connected if they occur in a window
        #    of 10 words.
        extractor.candidate_weighting(window=10, pos=pos)

        # 5. get the 10-highest scored candidates as keyphrases
        keyphrases = extractor.get_n_best(n=10)

    """

    def __init__(self):
        """Redefining initializer for SingleRank.
        """

        super(SingleRank, self).__init__()

        self.graph = nx.Graph()
        """ The word graph. """

    def candidate_selection(self, pos=None, stoplist=None):
        """ The candidate selection as described in the SingleRank paper.

            Args:
                pos (set): the set of valid POS tags, defaults to (NN, NNS,
                    NNP, NNPS, JJ, JJR, JJS).
                stoplist (list): the stoplist for filtering candidates, defaults
                    to the nltk stoplist. Words that are punctuation marks from
                    string.punctuation are not allowed.
        """

        # define default pos tags set
        if pos is None:
            pos = {'NN', 'NNS', 'NNP', 'NNPS', 'JJ', 'JJR', 'JJS'}

        # select sequence of adjectives and nouns
        self.longest_pos_sequence_selection(valid_pos=pos)

        # initialize stoplist list if not provided
        if stoplist is None:
            stoplist = self.stoplist

        # filter candidates containing stopwords or punctuation marks
        self.candidate_filtering(stoplist=list(string.punctuation) +
                                          ['-lrb-', '-rrb-', '-lcb-', '-rcb-', '-lsb-', '-rsb-'] +
                                          stoplist)

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

        # loop through sequence to build the edges in the graph
        for j, node_1 in enumerate(sequence):
            for k in range(j + 1, min(j + window, len(sequence))):
                node_2 = sequence[k]
                if node_1[1] in pos and node_2[1] in pos \
                        and node_1[0] != node_2[0]:
                    if not self.graph.has_edge(node_1[0], node_2[0]):
                        self.graph.add_edge(node_1[0], node_2[0], weight=0)
                    self.graph[node_1[0]][node_2[0]]['weight'] += 1.0

    def candidate_weighting(self, window=10, pos=None, normalized=False):
        """Candidate ranking using random walk.

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

        # compute the word scores using random walk
        w = nx.pagerank_scipy(self.graph, alpha=0.85, weight='weight')

        # loop through the candidates
        for k in self.candidates.keys():
            tokens = self.candidates[k].lexical_form
            self.weights[k] = sum([w[t] for t in tokens])
            if normalized:
                self.weights[k] /= len(tokens)
