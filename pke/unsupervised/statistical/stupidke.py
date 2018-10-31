# -*- coding: utf-8 -*-
# Author: ygor Gallina
# Date: 19-10-2018

"""StupidKE keyphrase extraction model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from pke.unsupervised import SingleRank


class StupidKE(SingleRank):
    """StupidKE keyphrase extraction model.

    Parameterized example::

        import string
        import pke

        # 1. create a StupidKE extractor.
        extractor = pke.unsupervised.StupidKE(input_file='path/to/input.xml')

        # 2. load the content of the document.
        extractor.read_document(format='corenlp')

        # 3. select the the longest sequences of nouns and adjectives, that do
        #    not contain punctuation marks or stopwords as candidates.
        pos = {'NOUN', 'PROPN', 'ADJ'}
        stoplist = list(string.punctuation)
        stoplist += ['-lrb-', '-rrb-', '-lcb-', '-rcb-', '-lsb-', '-rsb-']
        stoplist += stopwords.words('english')
        extractor.candidate_selection(pos=pos, stoplist=stoplist)

        # 4. weight the candidates
        extractor.candidate_weighting(df=df)

        # 5. get the 10-highest scored candidates as keyphrases
        keyphrases = extractor.get_n_best(n=10)

    """

    def candidate_weighting(self, **kwargs):
        """Candidate weighting function using position.
        :param **kwargs:
        """

        # Weigh the candidates
        for k in self.candidates.keys():
            # the '-' ensures that the first item will have the higher weight
            self.weights[k] = -min(self.candidates[k].offsets)
