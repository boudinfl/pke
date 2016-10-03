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

