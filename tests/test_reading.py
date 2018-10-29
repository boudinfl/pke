#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pke

model = pke.unsupervised.StupidKE

test_en_raw_file = 'examples/C-1.txt'
test_pt_file = 'examples/2000_10_09-13_00_00-JornaldaTarde-8-topic-seg.txt-Nr1.xml'
test_pt_raw_file = 'examples/2000_10_09-13_00_00-JornaldaTarde-8-topic-seg.txt-Nr1.txt'


def test_reading_en():
    extract2 = model()
    with open(test_en_raw_file, 'r') as f:
        extract2.load_document(f, lang='en')

        extract3 = model()
    with open(test_en_raw_file, 'r') as f:
        text = f.read()
        extract3.load_document(text, lang='en')

    assert extract2.sentences == extract3.sentences


def test_reading_pt():
    extract1 = model()
    extract1.load_document(test_pt_file, lang='pt')

    extract2 = model()
    with open(test_pt_raw_file, 'r', encoding='iso-8859-1') as f:
        extract2.load_document(f, lang='pt')

    extract3 = model()
    with open(test_pt_raw_file, 'r', encoding='iso-8859-1') as f:
        text = f.read()
    extract3.load_document(text, lang='pt')

    assert len(extract1.sentences) == len(extract2.sentences) and \
            extract2.sentences == extract3.sentences


if __name__ == '__main__':
    test_reading_en()
    test_reading_pt()