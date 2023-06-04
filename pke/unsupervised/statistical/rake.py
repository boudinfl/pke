from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import re
from collections import defaultdict

import numpy
from nltk.metrics import edit_distance

from pke.base import LoadFile

class RAKE(LoadFile):
    
    def generate_candidate_keywords(self):

        

        self.candidates.clear()
        for i, s in enumerate(self.sentences):
            # pseudocode: start at beginning, continue until stopword, add candidate, go to next stopword and repeat until sentence ends
            raw = " ".join(s.words)
            tmp = re.sub(self.stoplist, '|', raw.strip())
            phrases = tmp.split('|')
            phrase_list = []
            for c in phrases:
                c = c.strip().lower()
                if c != "":
                    phrase_list.append(c)
            
            self.add_candidate(words=phrase_list,
                                       stems=sentence.stems[j:k],
                                       pos=sentence.pos[j:k],
                                       offset=shift + j,
                                       sentence_id=i)

        return