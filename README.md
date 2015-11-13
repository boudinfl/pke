# pke

Python Keyphrase Extraction module

To install this module:

    pip install git+https://github.com/boudinfl/pke.git

A typical usage of this module is:

    import pke

	doc = pke.SingleRank(input_file=sys.argv[1])
	doc.read_corenlp_document()
	doc.candidate_selection()
	doc.candidate_weighting()
	print (u';'.join([u for u,v in doc.get_n_best(n=10)])).encode('utf-8')

