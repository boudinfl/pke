#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import codecs
import argparse
import logging

sys.path.append(os.environ['PATH_CODE'])
import pke

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Keyphrase extraction script.')

    parser.add_argument('-i', 
                        '--input', 
                        help='input file', 
                        required=True)

    parser.add_argument('-o', 
                        '--output', 
                        help='output file',
                        required=True)

    parser.add_argument('-a',
                        '--approach', 
                        help='keyphrase extraction approach', 
                        required=True)

    parser.add_argument('-m',
                        '--model', 
                        help='Supervised model', 
                        default=None,
                        type=str)

    parser.add_argument('-f',
                        '--format',
                        help='input format',
                        required=True)

    parser.add_argument('-n',
                        '--nbest', 
                        help='number of extracted keyphrases',
                        default=10,
                        type=int)

    parser.add_argument('-d',
                        '--df',
                        help='path to the df weights file',
                        default=None,
                        type=str)

    parser.add_argument('-v',
                        '--verbose',
                        action='store_const',
                        const=True,
                        help='verbose mode using logging')

    args = parser.parse_args()

    # enabling verbose
    if args.verbose:
        logging.basicConfig(level=logging.INFO)

    # get class from module
    class_ = getattr(pke, args.approach, None)

    if not class_:
        logging.error('No valid extraction model given ['+args.approach+']')
        sys.exit(0)

    logging.info('keyphrase extraction using '+args.approach)

    if args.df:
        logging.info('loading df weights from '+args.df)
        df = pke.load_document_frequency_file(args.df, delimiter="\t")

    extr = class_(input_file=args.input)

    extr.read_document(format=args.format)

    extr.candidate_selection()

    if args.approach in ['TfIdf', 'TopicRank', 'SingleRank', 'KPMiner']:
        extr.candidate_weighting()
    elif args.approach in ['WINGNUS', 'Kea']:
        extr.feature_extraction(df=df)
        extr.classify_candidates(model=args.model)

    keyphrases = extr.get_n_best(n=args.nbest)

    with codecs.open(args.output, 'w', 'utf-8') as f:
        f.write(u'\n'.join([u+'\t'+str(v) for u, v in keyphrases]))












            