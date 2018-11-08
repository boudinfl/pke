#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pke
import codecs

model = pke.unsupervised.TopicRank

xml_test_file = 'tests/data/1939.xml'
raw_test_file = 'tests/data/1939.txt'


def test_reading():

    # loading XML input
    extr1 = model()
    extr1.load_document(xml_test_file)

    # loading txt input
    extr2 = model()
    extr2.load_document(raw_test_file)

    # loading from string
    extr3 = model()
    with codecs.open(raw_test_file, 'r', 'utf-8') as f:
        text = f.read()
    extr3.load_document(text)

    # loading from stream
    extr4 = model()
    with codecs.open(raw_test_file, 'r', 'utf-8') as f:
        extr4.load_document(f)

    assert len(extr1.sentences) == 4 and \
           extr2.sentences == extr3.sentences == extr4.sentences


if __name__ == '__main__':
    test_reading()