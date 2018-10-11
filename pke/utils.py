# -*- coding: utf-8 -*-

""" Useful functions for the pke module. """

from __future__ import division
from __future__ import absolute_import

import re
import csv
import math
import glob
import pickle
import gzip
import codecs
import logging

from collections import defaultdict

from pke.base import LoadFile

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

from nltk.stem.snowball import SnowballStemmer as Stemmer
from nltk.corpus import stopwords


def load_document_frequency_file(input_file,
                                 delimiter='\t'):
    """ Load a csv file containing document frequencies. Automatically detects
        if input file is compressed (gzip) if extension is '.gz'.

        Args:
            input_file (str): the input file containing document frequencies in
                csv format.
            delimiter (str): the delimiter used for separating term-document
                frequencies tuples, defauts to '\t'.

        Returns:
            frequencies (dic): a dictionary of the form {term_1: freq,
                term_2: freq}, freq being an integer.
    """

    # initialize the DF dictionary
    frequencies = {}

    # open the input file
    with gzip.open(input_file, 'rt') if input_file.endswith('.gz') else \
         codecs.open(input_file, 'rt') as f:

        # read the csv file
        df_reader = csv.reader(f, delimiter=delimiter)

        # populate the dictionary
        for row in df_reader:
            frequencies[row[0]] = int(row[1])

    # return the populated dictionary
    return frequencies


def compute_document_frequency(input_dir,
                               output_file,
                               format="corenlp",
                               extension="xml",
                               use_lemmas=False,
                               stemmer="porter",
                               stoplist=None,
                               delimiter='\t',
                               n=3):
    """ Compute n-gram document frequencies from a set of input documents. An
        extra row is added to the output file for specifying the number of
        documents from which the frequencies were computed (--NB_DOC-- tab XX).

        Args:
            input_dir (str): the input directory.
            output_file (str): the output file.
            format (str): the input files format, defaults to corenlp.
            extension (str): file extension for input documents, defaults to
                xml.
            use_lemmas (bool): whether lemmas from stanford corenlp are used
                instead of stems (computed by nltk), defaults to False.
            stemmer (str): the stemmer in nltk to used (if used), defaults
                to porter.
            stoplist (list): the stop words for filtering n-grams, default to
                None.
            delimiter (str): the delimiter between n-grams and document
                frequencies, default to tabulation.
            n (int): the length for ngrams, defaults to 3.
    """

    # document frequency container
    frequencies = defaultdict(set)

    # initialize number of documents
    nb_documents = 0

    # loop throught the documents
    for input_file in glob.glob(input_dir+'/*.'+extension):

        logging.info('reading file '+input_file)

        # initialize load file object
        doc = LoadFile(input_file)

        # read the input file
        doc.read_document(format=format,
                          use_lemmas=use_lemmas,
                          stemmer=stemmer,
                          sep='/')

        # candidate selection
        doc.ngram_selection(n=n)

        # filter candidates containing punctuation marks
        doc.candidate_filtering(stoplist=stoplist)

        # loop through candidates
        for lexical_form in doc.candidates:
            frequencies[lexical_form].add(input_file)

        nb_documents += 1

    # Dump the df container
    with gzip.open(output_file, 'wb') as f:

        # add the number of documents as special token
        first_line = '--NB_DOC--' + delimiter + str(nb_documents)
        f.write(first_line.encode('utf-8') + b'\n')

        for ngram in frequencies:
            line = ngram + delimiter + str(len(frequencies[ngram]))
            f.write(line.encode('utf-8') + b'\n')


def train_supervised_model(input_dir,
                           reference_file,
                           model_file,
                           df=None,
                           format="corenlp",
                           use_lemmas=False,
                           stemmer="porter",
                           model=None,
                           language='english',
                           extension="xml",
                           sep_doc_id=':',
                           sep_ref_keyphrases=',',
                           reference_stemming=False):
    """ Build a supervised keyphrase extraction model from a set of documents
        and a reference file.

        Args:
            input_dir (str): the input directory.
            reference_file (str): the reference file.
            model_file (str): the model output file.
            df (dict): df weights dictionary.
            format (str): the input files format, defaults to corenlp.
            use_lemmas (bool): weither lemmas from stanford corenlp are used
                instead of stems (computed by nltk), defaults to False.
            stemmer (str): the stemmer in nltk to used (if used), defaults
                to porter.
            model (object): the supervised model to train, defaults to None.
            extension (str): file extension for input documents, defaults to
                xml.
            sep_doc_id (str): the separator used for doc_id in reference file,
                defaults to ':'.
            sep_ref_keyphrases (str): the separator used for keyphrases in
                reference file, defaults to ','.
    """

    logging.info('building model '+str(model)+' from '+input_dir)

    references = load_references(reference_file,
                                 sep_doc_id=sep_doc_id,
                                 sep_ref_keyphrases=sep_ref_keyphrases,
                                 reference_stemming=reference_stemming,
                                 stemmer=stemmer)
    training_instances = []
    training_classes = []
    files = glob.glob(input_dir+'/*.'+extension)

    # get the input files from the input directory
    for input_file in files:

        logging.info('reading file '+input_file)

        # initialize the input file
        model.__init__(input_file=input_file, language=language)

        doc_id = input_file.split('/')[-1].split('.')[0]

        model.read_document(format=format,
                            use_lemmas=use_lemmas,
                            stemmer=stemmer,
                            sep='/')

        # select candidates using default method
        model.candidate_selection()

        # extract features
        model.feature_extraction(df=df, training=True)

        # annotate the reference keyphrases in the instances
        for candidate in model.instances:
            if candidate in references[doc_id]:
                training_classes.append(1)
            else:
                training_classes.append(0)
            training_instances.append(model.instances[candidate])

    logging.info('writing model to '+model_file)
    model.train(training_instances=training_instances,
                training_classes=training_classes,
                model_file=model_file)


def load_references(input_file,
                    sep_doc_id=':',
                    sep_ref_keyphrases=',',
                    reference_stemming=False,
                    stemmer='porter'):
    """ Load a reference file and returns a dictionary. """

    logging.info('loading reference keyphrases from '+input_file)

    references = defaultdict(list)

    with codecs.open(input_file, 'r', 'utf-8') as f:
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
            if reference_stemming:
                for i, k in enumerate(references[doc_id]):
                    stems = [Stemmer(stemmer).stem(u) for u in k.split()]
                    references[doc_id][i] = ' '.join(stems)

    return references


def compute_lda_model(input_dir,
                      output_file,
                      n_topics=500,
                      format="corenlp",
                      extension="xml",
                      use_lemmas=False,
                      stemmer="porter",
                      language="english"):
    """ Compute a LDA model from a collection of documents. Latent Dirichlet
        Allocation is computed using sklearn module.

        Args:
            input_dir (str): the input directory.
            output_file (str): the output file.
            n_topics (int): number of topics for the LDA model, defaults to 500.
            format (str): the input files format, defaults to corenlp.
            extension (str): file extension for input documents, defaults to
                xml.
            use_lemmas (bool): whether lemmas from stanford corenlp are used
                instead of stems (computed by nltk), defaults to False.
            stemmer (str): the stemmer in nltk to used (if used), defaults
                to porter.
            language (str): the language of the documents, used for stop_words
                in sklearn CountVectorizer, defaults to 'english'.
    """

    # texts container
    texts = []

    # loop throught the documents
    for input_file in glob.glob(input_dir+'/*.'+extension):

        logging.info('reading file '+input_file)

        # initialize load file object
        doc = LoadFile(input_file)

        # read the input file
        doc.read_document(format=format,
                          use_lemmas=use_lemmas,
                          stemmer=stemmer,
                          sep='/')

        # container for current document
        text = []

        # loop through sentences
        for sentence in doc.sentences:

            # get the tokens (stems) from the sentence if they are not
            # punctuation marks 
            text.extend([ sentence.stems[i] for i in range(sentence.length) \
                          if not re.search('[^A-Z$]', sentence.pos[i]) ])
        
        # add the document to the texts container
        texts.append(' '.join(text))

    # vectorize dataset
    # get the stoplist from nltk because CountVectorizer only contains english
    # stopwords atm
    tf_vectorizer = CountVectorizer(stop_words=stopwords.words(language))
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
    logging.info('writing LDA model to '+output_file)
    with gzip.open(output_file, 'wb') as fp:
        pickle.dump(saved_model, fp)


def load_document_as_bos(input_file,
                         format="corenlp",
                         use_lemmas=False,
                         stemmer="porter",
                         stoplist=[]):
    """Load a document as a bag of stems.

    Args:
        input_file (str): path to input file.
        format (str): the input files format, defaults to corenlp.
        use_lemmas (bool): whether lemmas from stanford corenlp are used
            instead of stems (computed by nltk), defaults to False.
        stemmer (str): the stemmer in nltk to used (if used), defaults
            to porter.
        stoplist (list): the stop words for filtering tokens, default to [].
    """

    # initialize load file object
    doc = LoadFile(input_file)

    # read the input file
    doc.read_document(format=format,
                      use_lemmas=use_lemmas,
                      stemmer=stemmer,
                      sep='/')

    # initialize document vector
    vector = defaultdict(int)

    # loop through the sentences
    for i, sentence in enumerate(doc.sentences):

        # loop through the tokens
        for j, stem in enumerate(sentence.stems):

            # skip stem if it occurs in the stoplist
            if stem in stoplist:
                continue

            # count the occurrence of the stem
            vector[stem] += 1    

    return vector


def compute_pairwise_similarity_matrix(input_dir,
                                       output_file,
                                       collection_dir=None,
                                       df=None,
                                       format="corenlp",
                                       extension="xml",
                                       use_lemmas=False,
                                       stemmer="porter",
                                       stoplist=[]):
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
        format (str): the input files format, defaults to corenlp.
        extension (str): file extension for input documents, defaults to xml.
        use_lemmas (bool): whether lemmas from stanford corenlp are used
            instead of stems (computed by nltk), defaults to False.
        stemmer (str): the stemmer in nltk to used (if used), defaults
            to porter.
        stoplist (list): the stop words for filtering tokens, default to [].
    """

    # containers
    collection = {}
    documents = {}

    # initialize the number of documents
    N = df.get('--NB_DOC--', 1)

    # build collection tf*idf vectors
    if collection_dir is not None:

        # loop throught the documents in the collection
        for input_file in glob.glob(collection_dir+'/*.'+extension):

            logging.info('Reading file from {}'.format(input_file))

            # initialize document vector
            collection[input_file] = load_document_as_bos(input_file=input_file,
                                                          format=format,
                                                          use_lemmas=use_lemmas,
                                                          stemmer=stemmer,
                                                          stoplist=stoplist)

            # compute TF*IDF weights
            for stem in collection[input_file]:
                collection[input_file][stem] *= math.log(N / df.get(stem, 1), 2)

        # update N if a collection of documents is provided
        N += 1

    # loop throught the documents in the input directory
    for input_file in glob.glob(input_dir+'/*.'+extension):

        logging.info('Reading file from {}'.format(input_file))

        # initialize document vector
        documents[input_file] = load_document_as_bos(input_file=input_file,
                                                     format=format,
                                                     use_lemmas=use_lemmas,
                                                     stemmer=stemmer,
                                                     stoplist=stoplist)

        # compute TF*IDF weights
        for stem in documents[input_file]:
            documents[input_file][stem] *= math.log(N /(1 + df.get(stem, 1)), 2)

    # consider input documents as collection if None provided
    if not collection:
        collection = documents

    # open the output file in gzip mode
    with gzip.open(output_file, 'wb') as f:

        # compute pairwise similarity scores
        for doc_i in documents:
            for doc_j in collection:
                if doc_i == doc_j:
                    continue

                # inner product
                inner = 0.0
                for stem in set(documents[doc_i]) & set(collection[doc_j]):
                    inner += documents[doc_i][stem]*collection[doc_j][stem]

                # norms
                norm_i = sum([math.pow(documents[doc_i][t], 2) for t in documents[doc_i]])
                norm_i = math.sqrt(norm_i)
                norm_j = sum([math.pow(collection[doc_j][t], 2) for t in collection[doc_j]])
                norm_j = math.sqrt(norm_j)

                # compute cosine
                cosine = inner / (norm_i * norm_j)

                # encode line and write to output file
                line = doc_i + '\t' + doc_j + '\t' + str(cosine) + '\n'
                f.write(line.encode('utf-8'))

