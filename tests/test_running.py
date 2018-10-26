#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pke.unsupervised import (
    TopicRank, SingleRank,
    MultipartiteRank, PositionRank,
    TopicalPageRank, ExpandRank,
    TextRank, TfIdf, KPMiner,
    YAKE, StupidKE
)
from pke.supervised import Kea, WINGNUS

valid_pos = {'NN', 'NNS', 'NNP', 'NNPS', 'JJ', 'JJR', 'JJS'}
test_file = 'examples/C-1.xml'


def test_unsupervised_run():
    def test(model, file):
        extractor = model()
        extractor.load_document(file)
        extractor.candidate_selection(pos=valid_pos)
        extractor.candidate_weighting()

    models = [
        TopicRank, SingleRank,
        MultipartiteRank, PositionRank,
        TopicalPageRank, ExpandRank,
        TextRank, TfIdf, KPMiner,
        YAKE, StupidKE
    ]
    for m in models:
        test(m, test_file)


def test_supervised_run():
    def test(model, file):
        extractor = model()
        extractor.load_document(file)
        extractor.candidate_selection(pos=valid_pos)
        extractor.candidate_weighting()

    models = [
        Kea, WINGNUS
    ]
    for m in models:
        test(m, test_file)


if __name__ == '__main__':
    test_unsupervised_run()
    test_supervised_run()