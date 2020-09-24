#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pke
import codecs

model = pke.unsupervised.TopicRank

data_path = os.path.join('tests', 'data')
xml_test_file = data_path + os.sep + '1939.xml'
raw_test_file = data_path + os.sep + '1939.txt'


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

    assert len(extr1.sentences) == 4 and extr2.sentences == extr3.sentences


def test_french_model():
    extr = model()
    extr.load_document('est-ce')
    assert '' not in extr.sentences[0].pos

if __name__ == '__main__':
    test_reading()