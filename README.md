# pke - python keyphrase extraction

pke currently implements the following keyphrase extraction models:

- SingleRank [(Xiaojun and Jianguo, 2008)][1]
- TopicRank [(Bougouin, Boudin and Daille, 2013)][2]
- KP-miner [(El-Beltagy and Rafea, 2010)][3]

## Requirements

    numpy
    scipy
    nltk
    networkx
    corenlp_parser

To install corenlp_parser:

    pip install git+https://github.com/boudinfl/corenlp_parser.git

## Installation

To install this module:

    pip install git+https://github.com/boudinfl/pke.git

## Example

A typical usage of this module is:

    import pke

    # Create a pke object using SingleRank model
	doc = pke.SingleRank(input_file=sys.argv[1])

	# Load the content of the document, here in CoreNLP XML format
	doc.read_corenlp_document()

	# Select the keyphrase candidates, for SingleRank the longest sequences of 
	# nouns and adjectives
	doc.candidate_selection()

	# Weight the candidates, for SingleRank using using random walk
	doc.candidate_weighting()

	# Get the n-highest scored candidates
	print (u';'.join([u for u,v in doc.get_n_best(n=10)])).encode('utf-8')


[1]: http://aclweb.org/anthology/C08-1122.pdf
[2]: http://aclweb.org/anthology/I13-1062.pdf
[3]: http://aclweb.org/anthology/S10-1041.pdf
