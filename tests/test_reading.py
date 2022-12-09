# -*- coding: utf-8 -*-

import pke

from .sample import sample, sample_doc, sample_list


def test_reading():

    # loading from string
    extractor1 = pke.base.LoadFile()
    extractor1.load_document(sample)

    # loading from string
    extractor2 = pke.base.LoadFile()
    extractor2.load_document(sample_doc)

    # loading from preprocessed text
    extractor3 = pke.base.LoadFile()
    extractor3.load_document(sample_list)

    assert len(extractor1.sentences) == 4 and extractor1.sentences == extractor2.sentences and \
           extractor2.sentences == extractor3.sentences and extractor1.sentences[0] == extractor2.sentences[0] and \
           extractor2.sentences[0] == extractor3.sentences[0]


if __name__ == '__main__':
    test_reading()
