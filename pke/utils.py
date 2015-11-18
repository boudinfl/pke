# -*- coding: utf-8 -*-

""" Useful functions for the pke module. """

import codecs
import cPickle
import math
from os import listdir
from collections import defaultdict
from nltk.stem.snowball import SnowballStemmer
from corenlp_parser import MinimalCoreNLPParser
from .base import LoadFile

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

    df = defaultdict(set)
    N = 0

    for input_file in listdir(input_dir):

        print input_file

        doc = LoadFile(input_dir+'/'+input_file)

        if format == "corenlp":
            doc.read_corenlp_document(use_lemmas=False)

        for i, sentence in enumerate(doc.sentences):

            skip = min(n, sentence.length)

            for j in range(sentence.length):
                for k in range(j+1, min(j+1+skip, sentence.length+1)):

                    if set(sentence.words[j:k]).intersection(stoplist):
                        continue

                    ngram = ' '.join(sentence.words[j:k]).lower()
                    df[ngram].add(input_file)

        N += 1

    # IDF calculation
    for ngram in df:
        df[ngram] = math.log(float(N) / len(df[ngram]), 2)

    # Dump the df container
    with codecs.open(output_file, 'w') as f:
        cPickle.dump(df, f)