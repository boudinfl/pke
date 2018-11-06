#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pke.unsupervised import (
    TopicRank, SingleRank,
    MultipartiteRank, PositionRank,
    TopicalPageRank, ExpandRank,
    TextRank, TfIdf, KPMiner,
    YAKE, FirstPhrases
)
from pke.supervised import Kea, WINGNUS

test_file = 'examples/C-1.xml'


def test_unsupervised_run():
    def test(model, file):
        extractor = model()
        extractor.load_document(file)
        extractor.candidate_selection()
        extractor.candidate_weighting()

    models = [
        TopicRank, SingleRank,
        MultipartiteRank, PositionRank,
        TopicalPageRank, ExpandRank,
        TextRank, TfIdf, KPMiner,
        YAKE, FirstPhrases
    ]
    for m in models:
        print("testing {}".format(m))
        test(m, test_file)


def test_supervised_run():
    def test(model, file):
        extractor = model()
        extractor.load_document(file)
        extractor.candidate_selection()
        extractor.candidate_weighting()

    models = [
        Kea, WINGNUS
    ]
    for m in models:
        print("testing {}".format(m))
        test(m, test_file)


if __name__ == '__main__':
    test_unsupervised_run()
    test_supervised_run()