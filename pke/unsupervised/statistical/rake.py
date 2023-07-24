# -*- coding: utf-8 -*-

"""RAKE keyphrase extraction model.

Statistical approach to keyphrase extraction described in:

* Stuart Rose, Dave Engel, Nick Cramer, and Wendy Cowley.
  Automatic Keyword Extraction from Individual Documents
  *Text Mining: Applications and Theory*, 2010.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import re
from collections import defaultdict

import numpy
from nltk.metrics import edit_distance

from pke.base import LoadFile
from pke.data_structures import Candidate

class RAKE(LoadFile):
    """YAKE keyphrase extraction model.

    Parameterized example::

        import pke
        from pke.lang import stopwords

        # 1. create a RAKE extractor.
        extractor = pke.unsupervised.YAKE()

        # 2. load the content of the document.
        stoplist = stopwords.get('english')
        extractor.load_document(input='path/to/input',
                                language='en',
                                stoplist=stoplist,
                                normalization=None)


        # 3. generate candidates by splitting the document at every stopword
        # or non-alphanumeric character
        extractor.generate_candidate_keywords()

        # 4. create a score for each word, then weight candidates baseed on
        # the scores of the words used, as specified in RAKE
        extractor.calculate_word_scores()
        extractor.generate_candidate_keyword_scores()

        # 5. get the 10-highest scored candidates as keyphrases.
        #    redundant keyphrases are removed from the output using levenshtein
        #    distance and a threshold.
        threshold = 0.8
        keyphrases = extractor.get_n_best(n=10, threshold=threshold)
    """


    def __init__(self):
        """Redefining initializer for RAKE.
        """

        super(RAKE, self).__init__()

        self.word_scores = {}
        """Container for word scores"""

        self.adjoining_words = defaultdict(Candidate)
        """Container for special stopword adjoined candidates"""


    def generate_candidate_keywords(self, adjoining = False):
        """Create candidates by extracting all consecutive non-stopword non-punctuation words
        Args:
            adjoining (bool)
                Specify if non-stopwords with the same stopword between them showing up 
                repeatedly should be classified as candidates (page 8 of RAKE paper)
        """
        self.candidates.clear()

        for i, s in enumerate(self.sentences):
            # start at beginning, continue until stopword, add candidate, go to next stopword and repeat until sentence ends
            j = 0
            shift = sum([s.length for s in self.sentences[0:i]])
            for k in range(len(s.words)):
                # index of first word in candidate should not be a stopword
                if s.words[j].lower() in self.stoplist or not self._is_alphanum(s.words[j]):
                    j = j + 1
                # if we are at the last word in the sentence or the word is a stopword/punctuation mark, add to candidates
                elif (k == len(s.words) - 1 or s.words[k].lower() in self.stoplist or not self._is_alphanum(s.words[k])) and k - j > 0: 
                    self.add_candidate(words=s.words[j:k],
                                        stems=s.stems[j:k],
                                        pos=s.pos[j:k],
                                        offset=shift + j,
                                        sentence_id=i)
                    j = k + 1 # skip stopword for next iteration

        """Count repeatedly stopword adjoined keywords
        CODE IS UNTESTED!!! Could not find an implementation of the adjoining feature to test results against
        USE AT OWN RISK! """
        # adjoining pseudocode: iterate and look for stopwords, store word-stopword-word in dictionary.
        # if twice appearing, save to member variable for later. 
        # Separate member variable needs to be kept separate from candidate dictionary to prevent double counting in word score calculation
        # when calculating candidate keyword scores, then add adjoining keywords to candidates container and weights container
        # candidates container necessary for get_n_best
        if adjoining:
            adjoining_once = {} # keep a container of adjoining keywords that only appear once.
            # container is adjoining stopword tuple mapped to candidate object
            for i, s in enumerate(self.sentences):
                # pseudocode: start at beginning, continue until stopword, add candidate, go to next stopword and repeat until sentence ends
                shift = sum([s.length for s in self.sentences[0:i]])
                for k in range(len(s.words)):
                    # if at start or end of sentence, there is not a word in front or behind, so skip
                    if k == 0 or k == len(s.words) - 1:
                        continue

                    left_word = s.words[k - 1].lower() # lowercase for comparisons
                    stop_word = s.words[k].lower()
                    right_word = s.words[k + 1].lower()
                    # if the middle word is a stopword and the prior and following words are not 
                    if stop_word in self.stoplist and left_word not in self.stoplist and right_word not in self.stoplist:
                        # if this is the first time an adjoining stopword is found, add to one time container
                        if (left_word, stop_word, right_word) not in adjoining_once:
                            c = Candidate()
                            c.surface_forms.append(s.words[k-1:k+1])
                            c.lexical_form = s.stems[k-1:k+1]
                            c.pos_patterns.append(s.pos[k-1:k+1])
                            c.offsets.append(shift + k - 1)
                            c.sentence_ids.append(i)
                            adjoining_once[(left_word, stop_word, right_word)] = c
                        # if this is the second time it has been found, update candidate in first time dictionary, then add to separate adjoining keyword dictionary
                        else:
                            adjoining_once[(left_word, stop_word, right_word)].surface_forms.append(s.words[k-1:k+1])
                            adjoining_once[(left_word, stop_word, right_word)].pos_patterns.append(s.pos[k-1:k+1])
                            adjoining_once[(left_word, stop_word, right_word)].offsets.append(shift + k - 1)
                            adjoining_once[(left_word, stop_word, right_word)].sentence_ids.append(i)
                            
                            lexical_form = ' '.join(s.stems[k-1:k+1])
                            #adjoining keyword dictionary same mapping as regular
                            self.adjoining_words[lexical_form] = adjoining_once[(left_word, stop_word, right_word)]
        return
    

    def calculate_word_scores(self, use_stems = False):
        """Create a dictionary of word scores.
        Calculating by taking the number of keywords the word appears with (including itself) in candidates and dividing by the frequency of occurrence of each word.
        Args:
            use_stems (bool)
                Specify if word scores should be calculated using stems
        """
        word_frequency = {} # container to store frequency of word occurrence
        word_degree = {} # container to store number of other keywords a word appears with
        
        # STEMS UNTESTED: Cannot find tests to compare against, use at own risk!
        if use_stems:
            for c in self.candidates.values():
                word_list_degree = len(c.lexical_form) - 1
                for word in c.lexical_form:
                    word_frequency.setdefault(word, 0)
                    word_frequency[word] += len(c.surface_forms) # +1 to frequency for every appearance on the surface
                    word_degree.setdefault(word, 0)
                    word_degree[word] += word_list_degree * len(c.surface_forms) # degree added per instance of surface form
        else:
            for c in self.candidates.values():
                word_list_degree = len(c.lexical_form) - 1
                for phrase in c.surface_forms:
                    for word in phrase:
                        word = word.lower()
                        word_frequency.setdefault(word, 0)
                        word_frequency[word] += 1
                        word_degree.setdefault(word, 0)
                        word_degree[word] += word_list_degree # one degree for every occurrence of the word

        # add in occurrences of the word itself to degree
        for item in word_frequency:
            word_degree[item] = word_degree[item] + word_frequency[item]
        
        # Calculate Word scores = deg(w)/frew(w)
        for item in word_frequency:
            self.word_scores.setdefault(item, 0)
            self.word_scores[item] = word_degree[item] / (word_frequency[item] * 1.0)  #orig.
        #word_score[item] = word_frequency[item]/(word_degree[item] * 1.0) #exp.
        return
    

    def generate_candidate_keyword_scores(self, use_stems = False):
        """Score each candidate
        Scoring is done by adding up word scores of all words that appear in each candidate
        Args:
            use_stems (bool)
                Specify if candidate scores should be calculated using stems
        """
        for c in self.candidates.values():
            # STEMS UNTESTED: Cannot find tests to compare against, use at own risk!
            if use_stems:
                phrases = c.lexical_form
            else:
                phrases = c.surface_forms
            for phrase in phrases:
                candidate_score = 0
                for word in phrase:
                    candidate_score += self.word_scores[word.lower()]
                self.weights[" ".join(c.lexical_form)] = candidate_score # candidates use lexical so get_n_best works

        # Handles adjoining words. If adjoining wasn't used, dictionary will be empty so code will be skipped.
        # ADJOINING UNTESTED, USE AT OWN RISK!!
        for c in self.adjoining_words.values():
            # add adjoining into candidates (for looking up with get_n_best)
            self.candidates[" ".join(c.lexical_form)] = c

            # STEMS UNTESTED: Cannot find tests to compare against, use at own risk!
            if use_stems:
                phrases = c.lexical_form
            else:
                phrases = c.surface_forms
            for phrase in phrases:
                candidate_score = 0
                # we know there are three terms with middle term as stopword
                # so just add first term and last terms word scores to candidate weight
                candidate_score += self.word_scores[phrase[0].lower()]
                candidate_score += self.word_scores[phrase[-1].lower()]
                self.weights[" ".join(c.lexical_form)] = candidate_score
        return
