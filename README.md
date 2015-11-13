# pke - python keyphrase extraction

pke currently holds the following keyphrase extraction models:

- SingleRank [(Xiaojun and Jianguo, 2008)][1]
- TopicRank [(Bougouin, Boudin and Daille, 2013)][2]
- KP-miner [(El-Beltagy and Rafea, 2010)][3]


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


[1]: http://aclweb.org/anthology/C08-1122.pdf
[2]: http://aclweb.org/anthology/I13-1062.pdf
[3]: http://aclweb.org/anthology/S10-1041.pdf
