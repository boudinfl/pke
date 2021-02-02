# -*- coding: utf-8 -*-

"""Useful functions for the pke module."""

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import os
import sys
import csv
import math
import glob
import pickle
import gzip
import json
import bisect
import codecs
import logging

from itertools import combinations, product
from collections import defaultdict

from pke.base import LoadFile, get_stopwords, get_stemmer_func

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation


def load_document_frequency_file(input_file,
                                 delimiter='\t'):
    """Load a tsv (tab-separated-values) file containing document frequencies.
    Automatically detects if input file is compressed (gzip) by looking at its
    extension (.gz).

    Args:
        input_file (str): the input file containing document frequencies in
            csv format.
        delimiter (str): the delimiter used for separating term-document
            frequencies tuples, defaults to '\t'.

    Returns:
        dict: a dictionary of the form {term_1: freq}, freq being an integer.
    """

    # initialize the DF dictionary
    frequencies = {}

    # open the input file
    with gzip.open(input_file, 'rt', encoding='utf-8') if input_file.endswith('.gz') else \
            codecs.open(input_file, 'rt', encoding='utf-8') as f:
        # read the csv file
        df_reader = csv.reader(f, delimiter=delimiter)

        # populate the dictionary
        for row in df_reader:
            frequencies[row[0]] = int(row[1])

    # return the populated dictionary
    return frequencies


def compute_document_frequency(input_dir,
                               output_file,
                               extension='xml',
                               language='en',
                               normalization="stemming",
                               stoplist=None,
                               delimiter='\t',
                               n=3,
                               max_length=None,
                               encoding=None):
    """Compute the n-gram document frequencies from a set of input documents. An
    extra row is added to the output file for specifying the number of
    documents from which the document frequencies were computed
    (--NB_DOC-- tab XXX). The output file is compressed using gzip.

    Args:
        input_dir (str): the input directory.
        output_file (str): the output file.
        extension (str): file extension for input documents, defaults to xml.
        language (str): language of the input documents (used for computing the
            n-stem or n-lemma forms), defaults to 'en' (english).
        normalization (str): word normalization method, defaults to 'stemming'.
            Other possible values are 'lemmatization' or 'None' for using word
            surface forms instead of stems/lemmas.
        stoplist (list): the stop words for filtering n-grams, default to None.
        delimiter (str): the delimiter between n-grams and document frequencies,
            defaults to tabulation (\t).
        n (int): the size of the n-grams, defaults to 3.
        encoding (str): encoding of files in input_dir, default to None.
    """

    # document frequency container
    frequencies = defaultdict(int)

    # initialize number of documents
    nb_documents = 0

    # loop through the documents
    for input_file in glob.iglob(input_dir + os.sep + '*.' + extension):

        #logging.info('reading file {}'.format(input_file))

        # initialize load file object
        doc = LoadFile()

        # read the input file
        doc.load_document(input=input_file,
                          language=language,
                          normalization=normalization,
                          max_length=max_length,
                          encoding=encoding)

        # candidate selection
        doc.ngram_selection(n=n)

        # filter candidates containing punctuation marks
        doc.candidate_filtering(stoplist=stoplist)

        # loop through candidates
        for lexical_form in doc.candidates:
            frequencies[lexical_form] += 1

        nb_documents += 1

        if nb_documents % 1000 == 0:
            logging.info("{} docs, memory used: {} mb".format(nb_documents,
                                                           sys.getsizeof(
                                                               frequencies)
                                                           / 1024 / 1024 ))

    # create directories from path if not exists
    if os.path.dirname(output_file):
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # dump the df container
    with gzip.open(output_file, 'wt', encoding='utf-8') as f:

        # add the number of documents as special token
        first_line = '--NB_DOC--' + delimiter + str(nb_documents)
        f.write(first_line + '\n')

        for ngram in frequencies:
            line = ngram + delimiter + str(frequencies[ngram])
            f.write(line + '\n')


def train_supervised_model(input_dir,
                           reference_file,
                           model_file,
                           extension='xml',
                           language='en',
                           normalization="stemming",
                           df=None,
                           model=None,
                           sep_doc_id=':',
                           sep_ref_keyphrases=',',
                           normalize_reference=False,
                           leave_one_out=False,
                           encoding=None,
                           ref_encoding=None):
    """Build a supervised keyphrase extraction model from a set of documents and
    a reference file.

    Args:
        input_dir (str): the input directory.
        reference_file (str): the reference file.
        model_file (str): the model output file.
        extension (str): file extension for input documents, defaults to xml.
        language (str): language of the input documents (used for computing the
            n-stem or n-lemma forms), defaults to 'en' (english).
        normalization (str): word normalization method, defaults to 'stemming'.
            Other possible values are 'lemmatization' or 'None' for using word
            surface forms instead of stems/lemmas.
        df (dict): df weights dictionary.
        model (object): the supervised model to train, defaults to Kea.
        sep_doc_id (str): the separator used for doc_id in reference file,
            defaults to ':'.
        sep_ref_keyphrases (str): the separator used for keyphrases in
            reference file, defaults to ','.
        normalize_reference (bool): whether to normalize the reference
            keyphrases, default to False.
        leave_one_out (bool): whether to use a leave-one-out procedure for
            training, creating one model per input, defaults to False.
        encoding (str): encoding of files in `input_dir`, default to None.
        ref_encoding (str): encoding of `reference_file`, default to None.
    """

    logging.info('building model {} from {}'.format(model, input_dir))

    references = load_references(reference_file,
                                 sep_doc_id=sep_doc_id,
                                 sep_ref_keyphrases=sep_ref_keyphrases,
                                 normalize_reference=normalize_reference,
                                 language=language,
                                 encoding=ref_encoding)
    training_instances = []
    training_classes = []
    masks = {}
    offsets = []
    sizes = []

    # get the input files from the input directory
    for input_file in glob.iglob(input_dir + os.sep + '*.' + extension):

        logging.info('reading file {}'.format(input_file))

        # get the document id from file name
        doc_id = '.'.join(os.path.basename(input_file).split('.')[0:-1])

        # initialize the input file
        model.__init__()

        # load the document
        model.load_document(input=input_file,
                            language=language,
                            normalization=normalization,
                            encoding=encoding)

        # candidate selection
        model.candidate_selection()

        # skipping documents without candidates
        if not len(model.candidates):
            continue

        # extract features
        model.feature_extraction(df=df, training=True)

        # add the first offset for leave-one-out masking
        masks[doc_id] = [len(training_classes)]

        # annotate the reference keyphrases in the instances
        for candidate in model.instances:
            if candidate in references[doc_id]:
                training_classes.append(1)
            else:
                training_classes.append(0)
            training_instances.append(model.instances[candidate])

        # add the last offset for leave-one-out masking
        masks[doc_id].append(len(training_classes))

    if not leave_one_out:
        logging.info('writing model to {}'.format(model_file))
        model.train(training_instances=training_instances,
                    training_classes=training_classes,
                    model_file=model_file)
    else:
        logging.info('leave-one-out training procedure')

        for doc_id in masks:
            logging.info('writing model to {}'.format(doc_id))
            ind = masks[doc_id]
            fold = training_instances[:ind[0]] + training_instances[ind[1]:]
            gold = training_classes[:ind[0]] + training_classes[ind[1]:]
            model.train(training_instances=fold,
                        training_classes=gold,
                        model_file=model_file+"."+doc_id+".pickle")


def load_references(input_file,
                    sep_doc_id=':',
                    sep_ref_keyphrases=',',
                    normalize_reference=False,
                    language="en",
                    encoding=None):
    """Load a reference file. Reference file can be either in json format or in
    the SemEval-2010 official format.

    Args:
        input_file (str): path to the reference file.
        sep_doc_id (str): the separator used for doc_id in reference file,
            defaults to ':'.
        sep_ref_keyphrases (str): the separator used for keyphrases in
            reference file, defaults to ','.
        normalize_reference (bool): whether to normalize the reference
            keyphrases using stemming, default to False.
        language (str): language of the input documents (used for computing the
            stems), defaults to 'en' (english).
        encoding (str): file encoding, default to None.
    """

    logging.info('loading reference keyphrases from {}'.format(input_file))

    references = defaultdict(list)

    # open input file
    with codecs.open(input_file, 'r', encoding) as f:

        # load json data
        if input_file.endswith('.json'):
            references = json.load(f)
            for doc_id in references:
                references[doc_id] = [keyphrase for variants in
                                      references[doc_id] for keyphrase in
                                      variants]
        # or load SemEval-2010 file
        else:
            for line in f:
                cols = line.strip().split(sep_doc_id)
                doc_id = cols[0].strip()
                keyphrases = cols[1].strip().split(sep_ref_keyphrases)
                for v in keyphrases:
                    if '+' in v:
                        for s in v.split('+'):
                            references[doc_id].append(s)
                    else:
                        references[doc_id].append(v)

        # normalize reference if needed
        if normalize_reference:

            # initialize stemmer
            stem = get_stemmer_func(language)

            for doc_id in references:
                for i, keyphrase in enumerate(references[doc_id]):
                    stems = [stem(w) for w in keyphrase.split()]
                    references[doc_id][i] = ' '.join(stems)

    return references


def load_lda_model(input_file):
    """Load a gzip file containing lda model.

    Args:
        input_file (str): the gzip input file containing lda model.

    Returns:
        dictionnary: a dictionary of the form {term_1: freq}, freq being an integer.
        model: an initialized sklearn.decomposition.LatentDirichletAllocation model.
    """
    model = LatentDirichletAllocation()
    with gzip.open(input_file, 'rb') as f:
            (dictionary,
             model.components_,
             model.exp_dirichlet_component_,
             model.doc_topic_prior_) = pickle.load(f)
    return dictionary, model


def compute_lda_model(input_dir,
                      output_file,
                      n_topics=500,
                      extension="xml",
                      language="en",
                      normalization="stemming",
                      max_length=None,
                      encoding=None):
    """Compute a LDA model from a collection of documents. Latent Dirichlet
    Allocation is computed using sklearn module.

    Args:
        input_dir (str): the input directory.
        output_file (str): the output file.
        n_topics (int): number of topics for the LDA model, defaults to 500.
        extension (str): file extension for input documents, defaults to xml.
        language (str): language of the input documents, used for stop_words
            in sklearn CountVectorizer, defaults to 'en'.
        normalization (str): word normalization method, defaults to 'stemming'.
            Other possible values are 'lemmatization' or 'None' for using word
            surface forms instead of stems/lemmas.
        encoding (str): encoding of files in `input_dir`, default to None.
    """

    # texts container
    texts = []

    # loop throught the documents
    for input_file in glob.iglob(input_dir + os.sep + '*.' + extension):

        logging.info('reading file {}'.format(input_file))

        # initialize load file object
        doc = LoadFile()

        # read the input file
        doc.load_document(input=input_file,
                          language=language,
                          normalization=normalization,
                          max_length=max_length,
                          encoding=encoding)

        # container for current document
        text = []

        # loop through sentences
        for sentence in doc.sentences:
            # get the tokens (stems) from the sentence if they are not
            # punctuation marks 
            text.extend([sentence.stems[i] for i in range(sentence.length)
                         if sentence.pos[i] != 'PUNCT' and
                         sentence.pos[i].isalpha()])

        # add the document to the texts container
        texts.append(' '.join(text))

    # vectorize dataset
    # get the stoplist from nltk because CountVectorizer only contains english
    # stopwords atm
    tf_vectorizer = CountVectorizer(
        stop_words=get_stopwords(language))
    tf = tf_vectorizer.fit_transform(texts)

    # extract vocabulary
    vocabulary = tf_vectorizer.get_feature_names()

    # create LDA model and train
    lda_model = LatentDirichletAllocation(n_components=n_topics,
                                          random_state=0,
                                          learning_method='batch')
    lda_model.fit(tf)

    # save all data necessary for later prediction
    saved_model = (vocabulary,
                   lda_model.components_,
                   lda_model.exp_dirichlet_component_,
                   lda_model.doc_topic_prior_)

    # Dump the df container
    logging.info('writing LDA model to {}'.format(output_file))

    # create directories from path if not exists
    if os.path.dirname(output_file):
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # dump the LDA model
    with gzip.open(output_file, 'wb') as fp:
        pickle.dump(saved_model, fp)


def load_document_as_bos(input_file,
                         language="en",
                         normalization="stemming",
                         stoplist=None,
                         encoding=None):
    """Load a document as a bag of words/stems/lemmas.

    Args:
        input_file (str): path to input file.
        language (str): language of the input documents, used for stop_words
            in sklearn CountVectorizer, defaults to 'en'.
        normalization (str): word normalization method, defaults to 'stemming'.
            Other possible values are 'lemmatization' or 'None' for using word
            surface forms instead of stems/lemmas.
        stoplist (list): the stop words for filtering tokens, default to [].
        encoding (str): encoding of `input_file`, default to None.
    """

    # initialize empty stoplist is None provided
    if stoplist is None:
        stoplist = []

    # initialize load file object
    doc = LoadFile()

    # read the input file
    doc.load_document(input=input_file,
                      language=language,
                      normalization=normalization,
                      encoding=encoding)

    # initialize document vector
    vector = defaultdict(int)

    # loop through the sentences and add the stems to the vector
    for i, sentence in enumerate(doc.sentences):
        for j, stem in enumerate(sentence.stems):
            if stem in stoplist:
                continue
            vector[stem] += 1

    return vector


def load_pairwise_similarities(path):
    """Load the pairwise similarities for ExpandRank."""

    pairwise_sim = defaultdict(list)
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            cols = line.strip().split()
            cols[0] = os.path.basename(cols[0])
            cols[1] = os.path.basename(cols[1])
            # Add (score, file1) to pairwise_sim[file0]
            # while ensuring that duplicate element are next to eahch other ?
            bisect.insort(pairwise_sim[cols[0]], (float(cols[2]), cols[1]))
            bisect.insort(pairwise_sim[cols[1]], (float(cols[2]), cols[0]))
    return pairwise_sim


def compute_pairwise_similarity_matrix(input_dir,
                                       output_file,
                                       collection_dir=None,
                                       df=None,
                                       extension="xml",
                                       language="en",
                                       normalization="stemming",
                                       stoplist=None,
                                       encoding=None):
    """Compute the pairwise similarity between documents in `input_dir` and
    documents in `collection_dir`. Similarity scores are computed using a cosine
    similarity over TF x IDF term weights. If there is no collection to compute
    those scores, the similarities between documents in input_dir are returned
    instead.

    Args:
        input_dir (str): path to the input directory.
        output_file (str): path to the output file.
        collection_dir (str): path to the collection of documents, defaults to
            None.
        df (dict): df weights dictionary.
        extension (str): file extension for input documents, defaults to xml.
        language (str): language of the input documents, used for stop_words
            in sklearn CountVectorizer, defaults to 'en'.
        normalization (str): word normalization method, defaults to 'stemming'.
            Other possible values are 'lemmatization' or 'None' for using word
            surface forms instead of stems/lemmas.
        stoplist (list): the stop words for filtering tokens, default to [].
        encoding (str): encoding of files in `input_dir`, default to None.
    """

    # containers
    collection = {}
    documents = {}

    # initialize the number of documents
    N = df.get('--NB_DOC--', 1)

    # initialize stoplist as empty if None provided
    if stoplist is None:
        stoplist = []

    # build collection tf*idf vectors
    if collection_dir is not None:

        # loop throught the documents in the collection
        for input_file in glob.iglob(
                collection_dir + os.sep + '*.' + extension):
            logging.info('Reading file from {}'.format(input_file))

            # initialize document vector
            collection[input_file] = load_document_as_bos(
                input_file=input_file, language=language,
                normalization=normalization, stoplist=stoplist,
                encoding=encoding)

            # compute TF*IDF weights
            for stem in collection[input_file]:
                collection[input_file][stem] *= math.log(N / df.get(stem, 1), 2)

        # update N if a collection of documents is provided
        N += 1

    # loop throught the documents in the input directory
    for input_file in glob.iglob(input_dir + os.sep + '*.' + extension):

        logging.info('Reading file from {}'.format(input_file))

        # initialize document vector
        documents[input_file] = load_document_as_bos(
            input_file=input_file, language=language,
            normalization=normalization, stoplist=stoplist,
            encoding=encoding)

        # compute TF*IDF weights
        for stem in documents[input_file]:
            documents[input_file][stem] *= math.log(N / df.get(stem, 1), 2)

    # consider input documents as collection if None provided
    if not collection:
        collection = documents
        iterator = combinations(documents, 2)
    else:
        iterator = product(documents, collection)

    # create directories from path if not exists
    if os.path.dirname(output_file):
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # open the output file in gzip mode
    with gzip.open(output_file, 'wt', encoding='utf-8') as f:

        # compute pairwise similarity scores
        for doc_i, doc_j in iterator:
            # inner product
            inner = 0.0
            for stem in set(documents[doc_i]) & set(collection[doc_j]):
                inner += documents[doc_i][stem] * collection[doc_j][stem]

            # norms
            norm_i = sum([math.pow(documents[doc_i][t], 2) for t in
                          documents[doc_i]])
            norm_i = math.sqrt(norm_i)
            norm_j = sum([math.pow(collection[doc_j][t], 2) for t in
                          collection[doc_j]])
            norm_j = math.sqrt(norm_j)

            # compute cosine
            cosine = inner / (norm_i * norm_j)

            # write line to output file
            line = doc_i + '\t' + doc_j + '\t' + str(cosine) + '\n'
            f.write(line)
