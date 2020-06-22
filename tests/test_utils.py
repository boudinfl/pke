#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pke

data_path = os.path.join('tests', 'data')


def create_corpus(values, tmp_path, name='corpus.gz'):
    corpus_dir = tmp_path / name
    corpus_dir.mkdir()
    for k, v in values.items():
        (corpus_dir / k).write_text(v)
    return corpus_dir


def create_df(corpus_dir, tmp_path, name='corpus_df.gz'):
    corpus_df_file = tmp_path / name
    pke.utils.compute_document_frequency(
        str(corpus_dir), str(corpus_df_file), extension='txt', n=1)
    corpus_df = pke.utils.load_document_frequency_file(str(corpus_df_file))
    return corpus_df, corpus_df_file


def load_pairwise_similarities(path):
    from collections import defaultdict
    import gzip
    import bisect
    """Load the pairwise similarities for ExpandRank."""

    pairwise_sim = defaultdict(list)
    with gzip.open(path, 'rb') as f:
        lines = f.readlines()
        for line in lines:
            cols = line.decode('utf-8').strip().split()
            cols[0] = os.path.basename(cols[0])
            cols[1] = os.path.basename(cols[1])
            # Add (score, file1) to pairwise_sim[file0]
            # while ensuring that duplicate element are next to eahch other ?
            bisect.insort(pairwise_sim[cols[0]], (float(cols[2]), cols[1]))
            bisect.insort(pairwise_sim[cols[1]], (float(cols[2]), cols[0]))
    return pairwise_sim


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
    corpus = {'a.txt': 'lorem sit amet', 'b.txt': 'lorem ipsum'}
    tmp_corpus = create_corpus(corpus, tmp_path)

    # Create expected value
    expected = Counter([t for v in corpus.values() for t in v.split()])
    expected['--NB_DOC--'] = len(corpus)

    # Compute document frequency
    tmp_freq = tmp_path / 'tmp_doc_freq.tsv.gz'
    pke.utils.compute_document_frequency(
        str(tmp_corpus), str(tmp_freq), extension='txt', n=1)

    # Asserting
    df = pke.utils.load_document_frequency_file(str(tmp_freq))
    assert df == expected


def test_compute_lda(tmp_path):
    import gzip
    import pickle

    # Create a corpus
    corpus = {'a.txt': 'lorem sit amet', 'b.txt': 'lorem ipsum'}
    tmp_corpus = create_corpus(corpus, tmp_path)

    # Create expected value
    expected_dict = set(t for v in corpus.values() for t in v.split())

    # Compute LDA topics
    tmp_lda = tmp_path / 'lda.pickle.gz'
    pke.utils.compute_lda_model(
        str(tmp_corpus), str(tmp_lda), n_topics=2, extension='txt')

    # Asserting
    with gzip.open(tmp_lda, 'rb') as f:
            (dictionary, _, _, _) = pickle.load(f)
    assert sorted(dictionary) == sorted(expected_dict)


def test_load_document_as_bos(tmp_path):

    # Create file
    tmp_file = tmp_path / 'tmp_file.txt'
    tmp_file.write_text('lorem ipsum sit amet')

    # Create expected value
    expected = {'lorem': 1, 'ipsum': 1, 'sit': 1, 'amet': 1}

    # Compute bag of stem
    bos = pke.utils.load_document_as_bos(str(tmp_file))

    assert bos == expected


def test_compute_pairwise_sim_one_corpus(tmp_path):
    # TODO: test with/without collection_dir and with/without df
    # Create a corpus
    corpus = {'a.txt': 'lorem sit amet', 'b.txt': 'lorem ipsum'}
    corpus_dir = create_corpus(corpus, tmp_path)
    corpus_df, _ = create_df(corpus_dir, tmp_path)

    # Create expected value
    expected = {k: sorted([k2 for k2 in corpus if k2 != k]) for k in corpus}

    # Compute pairwise similarity
    pairw_file = tmp_path / 'pairwise.gz'
    pke.utils.compute_pairwise_similarity_matrix(
        str(corpus_dir), str(pairw_file), extension='txt',
        collection_dir=None, df=corpus_df)  # TODO: remove and fix "df={}"
    pairw = load_pairwise_similarities(str(pairw_file))

    # Asserting
    # Removing score from Dict[File, (Score, File)]
    pairw = {k: sorted([f for _, f in v]) for k, v in pairw.items()}

    assert pairw == expected


def test_compute_pairwise_sim_two_corpus(tmp_path):
    # TODO: test with/without collection_dir and with/without df
    # Create a corpus
    corpus = {'a.txt': 'lorem sit amet', 'b.txt': 'lorem ipsum'}
    corpus_dir = create_corpus(corpus, tmp_path)
    corpus_df, _ = create_df(corpus_dir, tmp_path)

    # Create a collection
    collection = {'1.txt': 'sit ipsum', '2.txt': 'lorem ipsum sit amet'}
    collection_dir = create_corpus(collection, tmp_path, name='collection')

    # Create expected value
    expected = {k: sorted([k2 for k2 in collection]) for k in corpus}
    expected.update({k: sorted([k2 for k2 in corpus]) for k in collection})

    # Compute pairwise similarity
    tmp_pairw = tmp_path / 'pairwise.gz'
    pke.utils.compute_pairwise_similarity_matrix(
        str(corpus_dir), str(tmp_pairw), extension='txt',
        collection_dir=str(collection_dir), df=corpus_df)
    pairw = load_pairwise_similarities(str(tmp_pairw))

    # Asserting
    # Removing score from Dict[File, (Score, File)]
    pairw = {k: sorted([f for _, f in v]) for k, v in pairw.items()}

    assert pairw == expected


def test_train_supervised_model(tmp_path):
    # TODO : test with/without leave_one_out
    # Create a corpus
    corpus = {'a.txt': 'lorem sit amet', 'b.txt': 'lorem ipsum'}
    tmp_corpus = create_corpus(corpus, tmp_path)

    # Create a reference
    tmp_ref = tmp_path / 'ref.json'
    tmp_ref.write_text(
        '{"a": [["lorem"]], "b": [["lorem ipsum"]]}'
    )

    tmp_model = tmp_path / 'model.pickle'
    pke.utils.train_supervised_model(
        str(tmp_corpus), str(tmp_ref), str(tmp_model),
        extension='txt', df=None, leave_one_out=False,
        model=pke.supervised.Kea())  # TODO: fix doc for model param


def test_train_supervised_model_leave_one_out(tmp_path):
    # TODO : test with/without leave_one_out
    # Create a corpus
    corpus = {'a.txt': 'lorem sit amet', 'b.txt': 'lorem ipsum'}
    tmp_corpus = create_corpus(corpus, tmp_path)

    # Create a reference
    tmp_ref = tmp_path / 'ref.json'
    tmp_ref.write_text(
        '{"a": [["lorem"]], "b": [["ipsum"]]}'
    )

    tmp_model = tmp_path / 'model.pickle'
    pke.utils.train_supervised_model(
        str(tmp_corpus), str(tmp_ref), str(tmp_model),
        extension='txt', df=None, leave_one_out=True,
        model=pke.supervised.Kea())  # TODO: fix doc for model param


if __name__ == '__main__':
    test_load_reference()
