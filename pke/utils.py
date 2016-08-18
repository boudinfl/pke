# -*- coding: utf-8 -*-

""" Useful functions for the pke module. """

import csv
import glob
import gzip
import codecs
import logging
from collections import defaultdict

from .base import LoadFile
from .supervised import Kea, WINGNUS

from nltk.stem.snowball import SnowballStemmer as Stemmer


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
            df (dic): a dictionary of the form {term_1: freq, term_2: freq}.
    """

    df = {}
    if input_file.endswith('.gz'):
        with gzip.open(input_file, 'r') as f:
            df_reader = csv.reader(f, delimiter=delimiter)
            for row in df_reader:
                df[row[0]] = int(row[1])
    else:
        with codecs.open(input_file, 'r') as f:
            df_reader = csv.reader(f, delimiter=delimiter)
            for row in df_reader:
                print row
                df[row[0]] = int(row[1])
    return df


def compute_document_frequency(input_dir, 
                               output_file, 
                               format="corenlp",
                               use_lemmas=False,
                               stemmer="porter",
                               stoplist=None,
                               delimiter='\t',
                               n=3,
                               extension="xml"):
    """ Compute the n-gram document frequency from a set of documents. 

        Args:
            input_dir (str): the input directory.
            output_file (str): the output file.
            format (str): the input files format, defaults to corenlp.
            use_lemmas (bool): weither lemmas from stanford corenlp are used
                instead of stems (computed by nltk), defaults to False.
            stemmer (str): the stemmer in nltk to used (if used), defaults
                to porter.
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
    for input_file in glob.glob(input_dir+'/*.'+extension):

        logging.info('reading file '+input_file)

        doc = LoadFile(input_file)

        if format == "corenlp":
            doc.read_corenlp_document(use_lemmas=use_lemmas, stemmer=stemmer)
        elif format == "pre":
            doc.read_preprocessed_document(stemmer=stemmer)
        elif format == "raw":
            doc.read_raw_document(stemmer=stemmer)

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
                           use_lemmas=False,
                           stemmer="porter",
                           model=Kea(),
                           language='english',
                           extension="xml",
                           sep_doc_id=':',
                           sep_ref_keyphrases=',',
                           reference_stemming=False,
                           dblp_candidates=None):
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
            model (pke.supervised object): the supervised model to train, 
                defaults to a Kea object.
            extension (str): file extension for input documents, defaults to 
                xml.
            sep_doc_id (str): the separator used for doc_id in reference file, 
                defaults to ':'.
            sep_ref_keyphrases (str): the separator used for keyphrases in 
                reference file, defaults to ','.
            dblp_candidates (list): valid candidates according to the list of
                candidates extracted from the dblp titles.
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

    # number of files for IDF computation
    N = len(files)-1

    # get the input files from the input directory
    for input_file in files:

        logging.info('reading file '+input_file)

        # initialize the input file
        model.__init__(input_file=input_file, language=language)

        doc_id = input_file.split('/')[-1].split('.')[0]

        # input file format
        if format == "corenlp":
            model.read_corenlp_document(use_lemmas=use_lemmas, stemmer=stemmer)
        elif format == "pre":
            model.read_preprocessed_document(stemmer=stemmer)
        elif format == "raw":
            doc.read_raw_document(stemmer=stemmer)

        # select candidates using default method
        if dblp_candidates is not None:
            model.candidate_selection(dblp_candidates=dblp_candidates)
            N = 5082856
        else:
            model.candidate_selection()

        # extract features
        model.feature_extraction(df=df, N=N, training=True)
        
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






