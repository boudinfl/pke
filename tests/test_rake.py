from __future__ import absolute_import
import os
import sys
sys.path.append('pke')
print(sys.path)
from unsupervised.statistical.rake import RAKE
sys.path.pop()


from sample import sample_list
valid_pos = {'NOUN', 'PROPN', 'ADJ'}

# TODO: Download a verified RAKE implementation like https://github.com/aneesha/RAKE/blob/master/rake.py and verify same results

stop_words = []
for line in open("tests/data/SmartStoplist.txt"):
    if line.strip()[0:1] != "#":
        for word in line.split():  # in case more than one per line
            stop_words.append(word)


def test_rake_candidate_selection():
    extractor = RAKE()
    extractor.load_document(input=sample_list, stoplist=stop_words)
    extractor.generate_candidate_keywords()
    print(extractor.candidates)
    print(len(extractor.candidates))
    assert len(extractor.candidates) == 22


if __name__ == '__main__':
    test_rake_candidate_selection()