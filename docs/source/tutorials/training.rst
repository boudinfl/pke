Training supervised models
==========================

pke ships with a collection of already trained models (for supervised
keyphrase extraction approaches) and document frequency counts that were
computed on the training set of the SemEval-2010 benchmark dataset. These
resources are located into the ``pke/models/`` directory.

**Note that already trained models/DF counts are used by default if no parameters
are given.**

The following snippet of code illustrates how to train a new supervised model:

.. code::

   import pke

   """Train a Kea model given a collection of document, a document frequency
   counts file and a reference file (gold keyphrases).
   """

   # load the DF counts from file
   df_counts = pke.load_document_frequency_file(input_file='/path/to/df_counts')

   # train a new Kea model
   pke.train_supervised_model(input_dir='/path/to/collection/of/documents/',
                              reference_file='/path/to/reference/file',
                              model_file='/path/to/model/file',
                              df=df_counts,
                              extension='xml',
                              language='en',
                              normalization="stemming",
                              model=pke.supervised.Kea())


The training data consists of a set of documents along with a reference file
containing annotated keyphrases in the following formats:

1. `SemEval-2010 format <http://docs.google.com/Doc?id=ddshp584_46gqkkjng4>`_,
   i.e. ``FILENAME\s:\sKEYPHRASE_LIST``

.. code::

    C-41 : hybrid system,quality of service+service quality, [...]

2. json format

.. code::

   {
     "C-41": [
       [
         "hybrid system"
       ],
       [
         "quality of service",
         "service quality"
       ],
       [...]
     ]
   }