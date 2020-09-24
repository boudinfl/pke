#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Florian Boudin & Timothée Poulain
# Date: 10-02-2018

"""CollabRank Towards a Collaborative Approach to Single-Document
Keyphrase model
Graph-based ranking approach to keyphrase extraction described in:
* Xiaojun Wan and Jianguo Xiao
Proceedings of the 22nd International Conference on Computational Linguistics (Coling 2008), pages 969–976
"""


from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
from __future__ import absolute_import

from pke.unsupervised import SingleRank
from pke.base import LoadFile

import networkx as nx
import logging


class CollabRank(SingleRank):
    """CollabRank keyphrase extraction model.

    Parameterized example::

        import pke
        import string
        from nltk.corpus import stopwords

        # 1. create an CollabRank extractor.
        extractor = pke.unsupervised.CollabRank()

        # 2. load the content of the document.
        extractor.load_document(input='path/to/input.xml')

        # 3. select the the longest sequences of nouns and adjectives, that do
        #    not contain punctuation marks or stopwords as candidates.
        pos = {'NOUN', 'PROPN', 'ADJ'}
        stoplist = list(string.punctuation)
        stoplist += ['-lrb-', '-rrb-', '-lcb-', '-rcb-', '-lsb-', '-rsb-']
        stoplist += stopwords.words('english')
        extractor.candidate_selection(pos=pos, stoplist=stoplist)

        # 4. weight the candidates using the sum of their word's scores that are
        #    computed using random walk. In the graph, nodes are words (nouns
        #    and adjectives only) that are connected if they occur in a window
        #    of 10 words. A set of extra documents should be provided to expand
        #    the graph.
        collab_documents = [('path/to/input1.xml', similarity1),
                              ('path/to/input2.xml', similarity2)]
        extractor.candidate_weighting(window=10,
                                      pos=pos,
                                      collab_documents=collab_documents,
                                      format='corenlp')

        # 5. get the 10-highest scored candidates as keyphrases
        keyphrases = extractor.get_n_best(n=10)

    """

    def __init__(self):
        """ Redefining initializer for CollabRank. """

        super(CollabRank, self).__init__()

    def collab_word_graph(self,
                          input_file,
                          similarity,
                          window=10,
                          pos=None):
        """Expands the word graph using the given document.

        Args:
            input_file (str): path to the input file.
            similarity (float): similarity for weighting edges.
            window (int): the window within the sentence for connecting two
                words in the graph, defaults to 10.
            pos (set): the set of valid pos for words to be considered as nodes
                in the graph, defaults to ('NOUN', 'PROPN', 'ADJ').
        """

        # define default pos tags set
        if pos is None:
            pos = {'NOUN', 'PROPN', 'ADJ'}

        # initialize document loader
        doc = LoadFile()
        print(input_file)
        # read document
        doc.load_document(input=input_file,
                          language=self.language,
                          normalization=self.normalization)

        # flatten document and initialize nodes
        sequence = []

        for sentence in doc.sentences:
            for j, node in enumerate(sentence.stems):
                if node not in self.graph and sentence.pos[j] in pos:
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
                    self.graph[node_1[0]][node_2[0]]['weight'] += float(similarity)

    def candidate_weighting(self,
                            window=10,
                            pos=None,
                            collab_documents=None,
                            normalized=False):
        """Candidate ranking using random walk.

        Args:
            window (int): the window within the sentence for connecting two
                words in the graph, defaults to 10.
            pos (set): the set of valid pos for words to be considered as nodes
                in the graph, defaults to ('NOUN', 'PROPN', 'ADJ').
            collab_documents (list): the set of documents to expand the graph,
                should be a list of tuples (input_path, similarity). Defaults to
                empty list, i.e. no expansion.
            normalized (False): normalize keyphrase score by their length,
                defaults to False.
        """

        # define default pos tags set
        if pos is None:
            pos = {'NOUN', 'PROPN', 'ADJ'}

        if collab_documents is None:
            collab_documents = []
            logging.warning('No cluster documents provided for CollabRank.')

        # build the word graph
        self.build_word_graph(window=window, pos=pos)

        # expand the word graph
        for input_file, similarity in collab_documents:
            self.collab_word_graph(input_file=input_file,
                                   similarity=similarity,
                                   window=window,
                                   pos=pos)

        # compute the word scores using random walk
        w = nx.pagerank_scipy(self.graph, alpha=0.85, weight='weight')

        # loop through the candidates
        for k in self.candidates.keys():
            tokens = self.candidates[k].lexical_form
            self.weights[k] = sum([w[t] for t in tokens])
            if normalized:
                self.weights[k] /= len(tokens)
