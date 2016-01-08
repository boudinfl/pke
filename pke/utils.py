# -*- coding: utf-8 -*-

""" Useful functions for the pke module. """

import logging
import codecs
import cPickle
import math
from os import listdir
from collections import defaultdict
from nltk.stem.snowball import SnowballStemmer
from .readers import MinimalCoreNLPParser
from .base import LoadFile
from supervised import Kea
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib

def compute_inverse_document_frequency(input_dir, 
                                       output_file, 
                                       format="corenlp",
                                       stoplist = None,
                                       n=3):
    """ Compute the n-gram inverse document frequency from a set of documents. 

        Args:
            input_dir (str): the input directory.
            output_file (str): the output file.
            format (str): the input files format, defaults to corenlp.
            n (int): the lenght for ngrams, defaults to 3.
    """

    logging.info('computing IDF weights from '+input_dir)

    # document frequency container
    df = defaultdict(set)

    # number of documents
    N = 0

    for input_file in listdir(input_dir):

        if format == "corenlp" and input_file[-3:] != 'xml':
            continue

        logging.info('reading file '+input_file)

        doc = LoadFile(input_dir+'/'+input_file)

        if format == "corenlp":
            doc.read_corenlp_document(use_lemmas=False)

        for i, sentence in enumerate(doc.sentences):

            skip = min(n, sentence.length)

            for j in range(sentence.length):
                for k in range(j+1, min(j+1+skip, sentence.length+1)):

                    if set(sentence.words[j:k]).intersection(stoplist):
                        continue

                    ngram = ' '.join(sentence.stems[j:k]).lower()
                    df[ngram].add(input_file)

        N += 1

    # IDF calculation
    for ngram in df:
        df[ngram] = math.log(float(N) / len(df[ngram]), 2)

    logging.info('writing IDF weights to '+output_file)
    # Dump the df container
    with codecs.open(output_file, 'w') as f:
        cPickle.dump(df, f)


def train_supervised_model(input_dir,
                           reference_file,
                           model_file,
                           idf=None,
                           format="corenlp",
                           model="kea"):
    """ Build a supervised keyphrase extraction model from a set of documents
        and a reference file.

        Args:
            input_dir (str): the input directory.
            reference_file (str): the reference file.
            model_file (str): the model output file.
            idf (dict): idf weights dictionary.
            format (str): the input files format, defaults to corenlp.
            model (str): the supervised model to train, defaults to kea.
    """

    logging.info('building model '+model+' from '+input_dir)

    references = load_references(reference_file)
    training_instances = []
    training_classes = []

    for input_file in listdir(input_dir):

        if format == "corenlp" and input_file[-3:] != 'xml':
            continue

        logging.info('reading file '+input_file)

        doc_id = input_file.split('.')[0]

        if model == "kea":
            doc = Kea(input_dir+'/'+input_file)
            if format == "corenlp":
                doc.read_corenlp_document(use_lemmas=False)
            doc.candidate_selection()
            doc.feature_extraction(idf=idf)

            for candidate in doc.instances:
                if candidate in references[doc_id]:
                    training_classes.append(1)
                else:
                    training_classes.append(0)
                training_instances.append(doc.instances[candidate])

    clf = MultinomialNB()
    clf.fit(training_instances, training_classes)
    logging.info('writing model to '+model_file)
    joblib.dump(clf, model_file)

        
def load_references(input_file):
    """ Load a reference file and returns a dictionary. """

    references = defaultdict(list)

    with codecs.open(input_file, 'r', 'utf-8') as f:
        for line in f:
            cols = line.strip().split(':')
            doc_id = cols[0].strip()
            keyphrases = cols[1].strip().split(',')
            for v in keyphrases:
                if '+' in v:
                    for s in v.split('+'):
                        references[doc_id].append(s)
                else:
                    references[doc_id].append(v)

    return references






