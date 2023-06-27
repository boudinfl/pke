from __future__ import absolute_import
import os
import sys

# this code allows testing using whatever code is on machine, not on repo
sys.path.append('pke')
from unsupervised.statistical.rake import RAKE
sys.path.pop()


from sample import sample_list
valid_pos = {'NOUN', 'PROPN', 'ADJ'}

# Following test cases ran against https://github.com/aneesha/RAKE/blob/master/rake.py results

stop_words = []
for line in open("tests/data/SmartStoplist.txt"):
    if line.strip()[0:1] != "#":
        for word in line.split():  # in case more than one per line
            stop_words.append(word)

extractor = RAKE()
extractor.load_document(input=sample_list, stoplist=stop_words)


def test_rake_candidate_selection():
    extractor.generate_candidate_keywords()
    assert len(extractor.candidates) == 18
    tmp = []
    for c in extractor.candidates.values():
        for i in c.surface_forms:
            tmp.append(" ".join(i).lower())
    tmp = [*set(tmp)]
    assert sorted(tmp) == sorted(['inverse problems', 'model', 'process', 'investigated', 'proved', 
                                  'proposed', 'efficiency', 'considered', 'compressible ion exchanger', 
                                  'mathematical model', 'numerical solution methods', 'allowing', 
                                  'proposed methods', 'unique solvability', 'ion exchanger compression', 
                                  'numerical experiment', 'ion exchange', 'demonstrated'])


def test_rake_word_scores():
    extractor.calculate_word_scores()
    assert extractor.word_scores == {'inverse': 2.0, 'problems': 2.0, 'mathematical': 2.0, 'model': 1.6666666666666667, 
                                     'ion': 2.4, 'exchange': 2.0, 'compressible': 3.0, 'exchanger': 3.0, 
                                     'considered': 1.0, 'allowing': 1.0, 'compression': 3.0, 'process': 1.0, 
                                     'investigated': 1.0, 'unique': 2.0, 'solvability': 2.0, 'proved': 1.0, 
                                     'numerical': 2.5, 'solution': 3.0, 'methods': 2.5, 'proposed': 1.5, 
                                     'efficiency': 1.0, 'demonstrated': 1.0, 'experiment': 2.0}


def test_rake_candidate_scores():
    extractor.generate_candidate_keyword_scores()
    assert extractor.get_n_best(6) == [('compressible ion exchanger', 8.4), ('ion exchanger compression', 8.4), 
                                      ('numerical solution methods', 8.0), ('numerical experiment', 4.5), 
                                      ('ion exchange', 4.4), ('inverse problems', 4.0)]


if __name__ == '__main__':
    test_rake_candidate_selection()
    test_rake_word_scores()
    test_rake_candidate_scores()