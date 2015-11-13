# pke - Python Keyphrase Extraction Module

pke currently holds the following keyphrase extraction models:

- SingleRank [(Xiaojun and Jianguo, 2008)][1]
- TopicRank [(Bougouin, Boudin and Daille, 2013)][2]


## Installation

To install this module:

    pip install git+https://github.com/boudinfl/pke.git


## Example

A typical usage of this module is:

    import pke

	doc = pke.SingleRank(input_file=sys.argv[1])
	doc.read_corenlp_document()
	doc.candidate_selection()
	doc.candidate_weighting()
	print (u';'.join([u for u,v in doc.get_n_best(n=10)])).encode('utf-8')


[1]: http://www.aclweb.org/anthology/C08-1122.pdf
[2]: http://aclweb.org/anthology/I/I13/I13-1062.pdf
