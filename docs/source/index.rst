.. pke documentation master file, created by
   sphinx-quickstart on Fri Sep 30 16:02:29 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to pke's documentation!
===============================

`pke` is an open source python-based keyphrase extraction toolkit. It provides
an end-to-end keyphrase extraction pipeline in which each component can be
easily modified or extented to develop new approaches. `pke` also allows for
easy benchmarking of state-of-the-art keyphrase extraction approaches, and
ships with supervised models trained on the SemEval-2010 dataset.

A minimal example of unsupervised keyphrase extraction using TopicRank is shown
below:

.. code-block:: python
    :linenos:

    import pke

    # initialize TopicRank
    extractor = pke.TopicRank(input_file='/path/to/input')

    # load the content of the document, preprocessing is carried out using nltk
    extractor.read_document(format='raw')

    # keyphrase candidate selection, here sequences of nouns and adjectives
    extractor.candidate_selection()

    # candidate weighting, here using a random walk algorithm
    extractor.candidate_weighting()

    # N-best selection, here the 10 highest scored candidates
    keyphrases = extractor.get_n_best(n=10)

If you use this toolkit, please cite:

 - **pke: an open source python-based keyphrase extraction toolkit.** Florian
   Boudin. *International Conference on Computational Linguistics (COLING), 
   demonstration papers, 2016.*

.. toctree::
   :maxdepth: 2

.. automodule:: pke

Base classes
------------

.. automodule:: pke.base
   :members:

Unsupervised models
-------------------

.. automodule:: pke.unsupervised
   :members:

Supervised models
-----------------

.. automodule:: pke.supervised
   :members:

Reader classes
--------------

.. automodule:: pke.readers
   :members:

Useful functions
----------------

.. automodule:: pke.utils
   :members:

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

