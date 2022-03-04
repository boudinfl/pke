#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pke

data_path = os.path.join('tests', 'data')


def create_df(corpus, tmp_path, name='corpus_df.gz'):
    df_file = tmp_path / name
    pke.utils.compute_document_frequency(
        corpus, str(df_file), n=1)
    corpus_df = pke.utils.load_document_frequency_file(str(df_file))
    return corpus_df, df_file


def test_load_reference():
    """Various tests for loading a reference file."""

    id = 'C-41'

    g1 = pke.utils.load_references(input_file=data_path + os.sep + 'reference.json',
                                   normalize_reference=True,
                                   language="en",
                                   encoding='utf-8')

    g2 = pke.utils.load_references(input_file=data_path + os.sep + 'reference.stem.json',
                                   normalize_reference=False,
                                   language="en",
                                   encoding='utf-8')

    g3 = pke.utils.load_references(input_file=data_path + os.sep + 'reference.final',
                                   normalize_reference=True,
                                   language="en",
                                   encoding='utf-8')

    g4 = pke.utils.load_references(input_file=data_path + os.sep + 'reference.stem.final',
                                   normalize_reference=False,
                                   language="en",
                                   encoding='utf-8')

    assert set(g1[id]) == set(g2[id]) == set(g3[id]) == set(g4[id])


# TODO: test load_document_frequency_file ? As it is used to test
#     compute_document_frequency

def test_compute_document_frequency(tmp_path):
    from collections import Counter
    # tmp_path is a Path object provided automatically by pytest

    # Create a corpus
    corpus = ['lorem sit amet', 'lorem ipsum']

    # Create expected value
    expected = Counter([t for v in corpus for t in v.split()])
    expected['--NB_DOC--'] = len(corpus)

    # Compute document frequency
    tmp_freq = tmp_path / 'tmp_doc_freq.tsv.gz'
    pke.utils.compute_document_frequency(
        corpus, str(tmp_freq), n=1)

    # Asserting
    df = pke.utils.load_document_frequency_file(str(tmp_freq))
    assert df == expected


def test_compute_lda(tmp_path):
    import gzip
    import pickle

    # Create a corpus
    corpus = ['lorem sit amet', 'lorem ipsum']

    # Create expected value
    expected_dict = set(t for v in corpus for t in v.split())

    # Compute LDA topics
    tmp_lda = tmp_path / 'lda.pickle.gz'
    pke.utils.compute_lda_model(
        corpus, str(tmp_lda), n_topics=2)

    # Asserting
    with gzip.open(tmp_lda, 'rb') as f:
        (dictionary, _, _, _) = pickle.load(f)
    assert sorted(dictionary) == sorted(expected_dict)


def test_train_supervised_model(tmp_path):
    # Create a corpus
    corpus = [('001', 'lorem sit amet'), ('002', 'lorem ipsum')]
    reference = {'001': ['ref1', 'ref2'], '002': ['ref1', 'ref2']}

    tmp_model = tmp_path / 'model.pickle'
    pke.utils.train_supervised_model(
        corpus, reference, str(tmp_model),
        df=None, leave_one_out=False,
        model=pke.supervised.Kea())  # TODO: fix doc for model param


def test_train_supervised_model_leave_one_out(tmp_path):
    # Create a corpus
    corpus = [('001', 'lorem sit amet'), ('002', 'lorem ipsum')]
    reference = {'001': ['ref1', 'ref2'], '002': ['ref1', 'ref2']}

    tmp_model = tmp_path / 'model.pickle'
    pke.utils.train_supervised_model(
        corpus, reference, str(tmp_model),
        df=None, leave_one_out=True,
        model=pke.supervised.Kea())  # TODO: fix doc for model param
