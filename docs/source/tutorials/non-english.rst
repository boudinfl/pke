Non English languages
=====================

pke uses spacy to pre-process the text. As such,
`all the languages <https://spacy.io/usage/models>`_ that are supported in spacy
can be processed in pke.

An example of keyphrase extraction for `French` is given below:

.. code::

    import pke

    text="""Chaque année, la France déforeste potentiellement 5,1 millions
    d'hectares via ses importations. Ensemble, agissons !"""

    # initialize a TopicRank extractor
    extractor = pke.unsupervised.TopicRank()

    # load the content of the document and perform French stemming

    extractor.load_document(input=text,
                            language='fr',
                            normalization="stemming")

    # keyphrase candidate selection, here sequences of nouns and adjectives
    # defined by the Universal PoS tagset
    extractor.candidate_selection(pos={"NOUN", "PROPN" "ADJ"})

    # candidate weighting, here using a random walk algorithm
    extractor.candidate_weighting()

    # N-best selection, keyphrases contains the 10 highest scored candidates as
    # (keyphrase, score) tuples
    keyphrases = extractor.get_n_best(n=2)

    >>> [('hectares', 0.25085794471351), ('importations', 0.22508473817875038)]