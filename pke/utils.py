# -*- coding: utf-8 -*-

""" Useful functions for the pke module. """

import logging
import codecs
import cPickle
import glob
import math
import gzip
import csv
from os import listdir
from collections import defaultdict
from nltk.stem.snowball import SnowballStemmer
from .readers import MinimalCoreNLPParser
from .base import LoadFile
from supervised import Kea, WINGNUS
from sklearn.naive_bayes import MultinomialNB
# from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib
from sklearn import preprocessing


def load_document_frequency_file(input_file,
                                 delimiter='\t'):
    """ Load a csv file containing document frequencies. """
    df = {}
    with gzip.open(input_file, 'r') as f:
        df_reader = csv.reader(f, delimiter=delimiter)
        for row in df_reader:
            df[row[0]] = int(row[1])
    return df


def compute_document_frequency(input_dir, 
                               output_file, 
                               format="corenlp",
                               stoplist=None,
                               delimiter='\t',
                               n=3,
                               extension="xml"):
    """ Compute the n-gram document frequency from a set of documents. 

        Args:
            input_dir (str): the input directory.
            output_file (str): the output file.
            format (str): the input files format, defaults to corenlp.
            stoplist (list): the stop words for filtering n-grams, default to
                None.
            delimiter (str): the delimiter between n-grams and document
                frequencies, default to tabulation.
            n (int): the lenght for ngrams, defaults to 3.
            extension (str): file extension for input documents, defaults to 
                xml.
    """

    logging.info('computing document frequency from '+input_dir)

    # document frequency container
    df = defaultdict(set)

    # loop throught the documents
    for input_file in glob.glob(input_dir+'/*.xml'):

        logging.info('reading file '+input_file)

        doc = LoadFile(input_file)

        if format == "corenlp":
            doc.read_corenlp_document(use_lemmas=False)

        for i, sentence in enumerate(doc.sentences):

            skip = min(n, sentence.length)
            lowercase_words = [u.lower() for u in sentence.words]

            for j in range(sentence.length):
                for k in range(j+1, min(j+1+skip, sentence.length+1)):

                    if set(lowercase_words[j:k]).intersection(stoplist):
                        continue

                    ngram = ' '.join(sentence.stems[j:k]).lower()
                    df[ngram].add(input_file)

    logging.info('writing document frequencies to '+output_file)
    
    # Dump the df container
    with gzip.open(output_file, 'w') as f:
        for ngram in df:
            f.write((ngram).encode('utf-8')+delimiter+str(len(df[ngram]))+'\n')


def train_supervised_model(input_dir,
                           reference_file,
                           model_file,
                           df=None,
                           format="corenlp",
                           model=Kea(),
                           extension="xml"):
    """ Build a supervised keyphrase extraction model from a set of documents
        and a reference file.

        Args:
            input_dir (str): the input directory.
            reference_file (str): the reference file.
            model_file (str): the model output file.
            df (dict): df weights dictionary.
            format (str): the input files format, defaults to corenlp.
            model (pke.supervised object): the supervised model to train, 
                defaults to a Kea object.
            extension (str): file extension for input documents, defaults to 
                xml.
    """

    logging.info('building model '+str(model)+' from '+input_dir)

    references = load_references(reference_file)
    training_instances = []
    training_classes = []
    files = glob.glob(input_dir+'/*.xml')

    # get the input files from the input directory
    for input_file in files:

        logging.info('reading file '+input_file)

        # initialize the input file
        model.__init__(input_file)

        doc_id = input_file.split('/')[-1].split('.')[0]

        # input file format
        if format == "corenlp":
            model.read_corenlp_document(use_lemmas=False)

        # select candidates using default method
        model.candidate_selection()

        # extract features
        model.feature_extraction(df=df, 
                                 N=len(files)-1,
                                 training=True)
        
        # annotate the reference keyphrases in the instances
        for candidate in model.instances:
            if candidate in references[doc_id]:
                training_classes.append(1)
            else:
                training_classes.append(0)
            training_instances.append(model.instances[candidate])

    clf = MultinomialNB()
    # clf = LogisticRegression(class_weight='auto', 
    #                          solver='liblinear', 
    #                          dual=False,
    #                          penalty='l2')
    clf.fit(training_instances, training_classes)
    logging.info('writing model to '+model_file)
    joblib.dump(clf, model_file)

        
def load_references(input_file):
    """ Load a reference file and returns a dictionary. """

    logging.info('loading reference keyphrases from '+input_file)

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






