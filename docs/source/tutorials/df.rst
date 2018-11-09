Document frequency counts
=========================

pke ships with document frequency (DF) counts computed on the SemEval-2010
benchmark dataset. These counts are used in various models (for example TfIdf
and Kea).

Computing DF counts
-------------------

The following code illustrates how to compute new document frequency
counts from another (or a larger) document collection:

.. code::

    from pke import compute_document_frequency
    from string import punctuation

    """Compute Document Frequency (DF) counts from a collection of documents.

    N-grams up to 3-grams are extracted and converted to their n-stems forms.
    Those containing a token that occurs in a stoplist are filtered out.
    Output file is in compressed (gzip) tab-separated-values format (tsv.gz).
    """

    # stoplist for filtering n-grams
    stoplist=list(punctuation)

    # compute df counts and store as n-stem -> weight values
    compute_document_frequency(input_dir='/path/to/collection/of/documents/',
                               output_file='/path/to/output.tsv.gz',
                               extension='xml',           # input file extension
                               language='en',                # language of files
                               normalization="stemming",    # use porter stemmer
                               stoplist=stoplist)

DF counts are stored as a compressed (gzip), tab-separated-values file.
The number of documents in the collection, used to compute Inverse Document
Frequency (IDF) weights, is stored as an extra line
``--NB_DOC-- tab number_of_documents``.
Below is an example of such a file (uncompressed):

.. code::

   --NB_DOC--  100
   greedi alloc  1
   sinc trial structur 1
   complex question  1
   [...]

Newly computed DF counts should be loaded and given as parameter to the
``candidate_weighting()`` method:

.. code::

   import pke

   """Keyphrase extraction using TfIdf and newly computed DF counts."""

   # initialize TfIdf model
   extractor = pke.unsupervised.TfIdf()

   # load the DF counts from file
   df_counts = pke.load_document_frequency_file(input_file='/path/to/df_counts')

   # load the content of the document
   extractor.load_document(input='/path/to/input.txt')

   # keyphrase candidate selection
   extractor.candidate_selection()

   # candidate weighting with the provided DF counts
   extractor.candidate_weighting(df=df_counts)

   # N-best selection, keyphrases contains the 10 highest scored candidates as
   # (keyphrase, score) tuples
   keyphrases = extractor.get_n_best(n=10)
