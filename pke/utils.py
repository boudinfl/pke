# -*- coding: utf-8 -*-

"""Useful functions for the pke module."""

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import os
import sys
import csv
import pickle
import gzip
import json
import codecs
import logging

from collections import defaultdict

from pke.base import LoadFile
from pke.lang import stopwords, langcodes

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

from nltk.stem.snowball import SnowballStemmer


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
    with (gzip.open(input_file, 'rt', encoding='utf-8')
          if input_file.endswith('.gz')
          else codecs.open(input_file, 'rt', encoding='utf-8')) as f:
        # read the csv file
        df_reader = csv.reader(f, delimiter=delimiter)

        # populate the dictionary
        for row in df_reader:
            frequencies[row[0]] = int(row[1])

    # return the populated dictionary
    return frequencies


def compute_document_frequency(documents,
                               output_file,
                               language='en',
                               stoplist=None,
                               normalization='stemming',
                               delimiter='\t',
                               # TODO: What is the use case for changing this ?
                               n=3):
    """Compute the n-gram document frequencies from a set of input documents.
    An extra row is added to the output file for specifying the number of
    documents from which the document frequencies were computed
    (--NB_DOC-- tab XXX). The output file is compressed using gzip.

    Args:
        documents (list): list of pke-readable documents.
        output_file (str): the output file.
        language (str): language of the input documents (used for computing the
            n-stem or n-lemma forms), defaults to 'en' (english).
        stoplist (list): the stop words for filtering n-grams, default to
            pke.lang.stopwords[language].
        normalization (str): word normalization method, defaults to
            'stemming'. Other possible value is 'none' for using word surface
            forms instead of stems/lemmas.
        delimiter (str): the delimiter between n-grams and document
            frequencies, defaults to tabulation (\t).
        n (int): the size of the n-grams, defaults to 3.
    """

    # document frequency container
    frequencies = defaultdict(int)

    # initialize number of documents
    nb_documents = 0

    # loop through the documents
    for document in documents:

        # initialize load file object
        doc = LoadFile()

        # read the input file
        doc.load_document(input=document,
                          language=language,
                          stoplist=stoplist,
                          normalization=normalization)

        # candidate selection
        doc.ngram_selection(n=n)

        # filter candidates containing punctuation marks
        doc.candidate_filtering()

        # loop through candidates
        for lexical_form in doc.candidates:
            frequencies[lexical_form] += 1

        nb_documents += 1

        if nb_documents % 1000 == 0:
            logging.info("{} docs, memory used: {} mb".format(
                nb_documents,
                sys.getsizeof(frequencies) / 1024 / 1024))

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


def train_supervised_model(documents,
                           references,
                           model_file,
                           language='en',
                           stoplist=None,
                           normalization="stemming",
                           df=None,
                           model=None,
                           leave_one_out=False):
    """Build a supervised keyphrase extraction model from a set of documents
    and reference keywords.

    Args:
        documents (list): list of tuple (id, pke-readable documents). `id`s
            should match the one in reference.
        references (dict): reference keywords.
        model_file (str): the model output file.
        language (str): language of the input documents (used for computing the
            n-stem or n-lemma forms), defaults to 'en' (english).
        stoplist (list): the stop words for filtering n-grams, default to
            pke.lang.stopwords[language].
        normalization (str): word normalization method, defaults to 'stemming'.
            Other possible values are 'lemmatization' or 'None' for using word
            surface forms instead of stems/lemmas.
        df (dict): df weights dictionary.
        model (object): the supervised model to train, defaults to Kea.
        leave_one_out (bool): whether to use a leave-one-out procedure for
            training, creating one model per input, defaults to False.
    """
    training_instances = []
    training_classes = []
    masks = {}

    # get the input files from the input directory
    for doc_id, document in documents:

        # logging.info('reading file {}'.format(input_file))

        # get the document id from file name
        # doc_id = '.'.join(os.path.basename(input_file).split('.')[0:-1])

        # initialize the input file
        model.__init__()

        # load the document
        model.load_document(input=document,
                            language=language,
                            stoplist=stoplist,
                            normalization=normalization)

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
                        model_file='{}.{}.pickle'.format(model_file, doc_id))


def load_references(input_file,
                    sep_doc_id=':',
                    sep_ref_keyphrases=',',
                    normalize_reference=False,
                    language="en",
                    encoding=None,
                    excluded_file=None):
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
        excluded_file (str): file to exclude (for leave-one-out
            cross-validation), defaults to None.
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
            langcode = langcodes.get(language.replace('en', 'xx'), 'porter')
            stemmer = SnowballStemmer(langcode)

            for doc_id in references:
                for i, keyphrase in enumerate(references[doc_id]):
                    stems = [stemmer.stem(w) for w in keyphrase.split()]
                    references[doc_id][i] = ' '.join(stems)

    # remove excluded file if needed
    if excluded_file is not None:
        if excluded_file not in references:
            logging.warning("{} is not in references".format(excluded_file))
        else:
            logging.info("{} removed from references".format(excluded_file))
            del references[excluded_file]

    return references


def load_lda_model(input_file):
    """Load a gzip file containing lda model.

    Args:
        input_file (str): the gzip input file containing lda model.

    Returns:
        dictionary: a dictionary of the form {term_1: freq}, freq being an
            integer.
        model: an initialized sklearn.decomposition.LatentDirichletAllocation
            model.
    """
    model = LatentDirichletAllocation()
    with gzip.open(input_file, 'rb') as f:
        (dictionary,
         model.components_,
         model.exp_dirichlet_component_,
         model.doc_topic_prior_) = pickle.load(f)
    return dictionary, model


def compute_lda_model(documents,
                      output_file,
                      n_topics=500,
                      language="en",
                      stoplist=None,
                      normalization="stemming"):
    """Compute a LDA model from a collection of documents. Latent Dirichlet
    Allocation is computed using sklearn module.

    Args:
        documents (str): list fo pke-readable documents.
        output_file (str): the output file.
        n_topics (int): number of topics for the LDA model, defaults to 500.
        language (str): language of the input documents, used for stop_words
            in sklearn CountVectorizer, defaults to 'en'.
        stoplist (list): the stop words for filtering words, default to
            pke.lang.stopwords[language].
        normalization (str): word normalization method, defaults to
            'stemming'. Other possible value is 'none'
            for using word surface forms instead of stems/lemmas.
    """

    # texts container
    texts = []

    # loop throught the documents
    for document in documents:

        # initialize load file object
        doc = LoadFile()

        # read the input file
        doc.load_document(input=document,
                          language=language,
                          normalization=normalization)

        # container for current document
        text = []

        # loop through sentences
        for sentence in doc.sentences:
            # get the tokens (stems) from the sentence if they are not
            # punctuation marks
            text.extend([sentence.stems[i] for i in range(sentence.length)
                         if sentence.pos[i] != 'PUNCT'
                         and sentence.pos[i].isalpha()])

        # add the document to the texts container
        texts.append(' '.join(text))

    # vectorize dataset
    # get the stoplist from pke.lang because CountVectorizer only contains
    # english stopwords atm
    if stoplist is None:
        # CountVectorizer expects a list
        #  stopwords.get is a set
        stoplist = list(stopwords.get(language))
    tf_vectorizer = CountVectorizer(
        stop_words=stoplist)
    tf = tf_vectorizer.fit_transform(texts)

    # extract vocabulary
    vocabulary = tf_vectorizer.get_feature_names_out()

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
