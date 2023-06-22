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
    
    def __init__(self):
        """Redefining initializer for RAKE.
        """

        super(RAKE, self).__init__()

        self.word_scores = {}


    def generate_candidate_keywords(self):
        # TODO: pairs across stopwords
        self.candidates.clear()

        for i, s in enumerate(self.sentences):
            # pseudocode: start at beginning, continue until stopword, add candidate, go to next stopword and repeat until sentence ends
            j = 0
            shift = sum([s.length for s in self.sentences[0:i]])
            for k in range(len(s.words)):
                if s.words[j].lower() in self.stoplist or not self._is_alphanum(s.words[j]):
                    j = j + 1
                elif (k == len(s.words) - 1 or s.words[k].lower() in self.stoplist or not self._is_alphanum(s.words[k])) and k - j > 0: # either we are at the last word in the sentence or the word is a stopword
                    self.add_candidate(words=s.words[j:k],
                                        stems=s.stems[j:k],
                                        pos=s.pos[j:k],
                                        offset=shift + j,
                                        sentence_id=i)
                    j = k + 1 # skip stopword for next iteration
        return
    

    def calculate_word_scores(self):
        word_frequency = {}
        word_degree = {}
        for c in self.candidates.values():
            word_list_degree = len(c.lexical_form) - 1
            for phrase in c.surface_forms:
                for word in phrase:
                    word = word.lower()
                    word_frequency.setdefault(word, 0)
                    word_frequency[word] += 1
                    word_degree.setdefault(word, 0)
                    word_degree[word] += word_list_degree # one degree for every occurrence of the word
        for item in word_frequency:
            word_degree[item] = word_degree[item] + word_frequency[item]
        
        # Calculate Word scores = deg(w)/frew(w)
        for item in word_frequency:
            self.word_scores.setdefault(item, 0)
            self.word_scores[item] = word_degree[item] / (word_frequency[item] * 1.0)  #orig.
        #word_score[item] = word_frequency[item]/(word_degree[item] * 1.0) #exp.
        return
    

    def generate_candidate_keyword_scores(self):
        for c in self.candidates.values():
            keyword_candidate = []
            for w in c.surface_forms[0]:
                keyword_candidate.append(w.lower())
            keyword_candidate = " ".join(keyword_candidate)
            self.weights.setdefault(keyword_candidate, 0)
            for phrase in c.surface_forms:
                candidate_score = 0
                for word in phrase:
                    candidate_score += self.word_scores[word.lower()]
                self.weights[keyword_candidate] = candidate_score
        return
