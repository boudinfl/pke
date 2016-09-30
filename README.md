# `pke` - python keyphrase extraction

`pke` is an open source python-based keyphrase extraction toolkit. It provides
an end-to-end keyphrase extraction pipeline in which each component can be
easily modified or extented to develop new approaches. `pke` also allows for 
easy benchmarking of state-of-the-art keyphrase extraction approaches, and 
ships with supervised models trained on the [SemEval-2010 dataset][8].

If you use this toolkit, please cite:

 - **`pke`: an open source python-based keyphrase extraction toolkit.** Florian
   Boudin. *International Conference on Computational Linguistics (COLING), 
   demonstration papers, 2016.*

## Requirements

    numpy
    scipy
    nltk
    networkx
    sklearn

## Installation

To install this module:

    pip install git+https://github.com/boudinfl/pke.git

## Implemented models

`pke` currently implements the following keyphrase extraction models:

- Unsupervised models
  - SingleRank [(Xiaojun and Jianguo, 2008)][1]
  - TopicRank [(Bougouin et al., 2013)][2]
  - KP-Miner [(El-Beltagy and Rafea, 2010)][3]
  - TF*IDF [(Spärck Jones, 1972)][4]

- Supervised models
  - Kea [(Witten et al., 1999)][5]
  - WINGNUS [(Nguyen and Luong, 2010)][6]

## Usage

### Input formats

Three input formats are currently supported:
  - raw text (`format='raw'`): text preprocessing (i.e. tokenization, sentence
    splitting and POS-tagging) is carried out using nltk.
  - preprocessed text (`format='preprocessed'`): whitespace-separated POS-tagged
    tokens, one sentence per line.
  - Stanford XML CoreNLP (`format='corenlp'`): output file produced using the
    annotators tokenize, ssplit, pos and lemma. Document logical structure
    information can by specified by incorporating attributes into the sentence
    elements of the CoreNLP XML format.

Detailed examples of input formats are provided in the `examples/` directory.

Default language in pke is English and default candidate selection methods are
based on the PTB tagset.

### Minimal example: unsupervised keyphrase extraction using TopicRank

All the unsupervised models can be used by typing the 5 lines below. For using
another model, simply replace TopicRank with SingleRank, KPMiner, TfIdf, etc.
A complete documented example is described in [`keyphrase-extraction.py`][7]
within the `examples/` directory.

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


### Minimal example: supervised keyphrase extraction using Kea

    import pke

    # initialize Kea
    extractor = pke.Kea(input_file='/path/to/input')

    # load the Document Frequency (DF) weights file
    df_counts = pke.load_document_frequency_file(input_file='/path/to/file')

    # load the content of the document, preprocessing is carried out using nltk
    extractor.read_document(format='raw')

    # candidate selection, here 1-3-grams that do not begin/end with a stopword
    extractor.candidate_selection()

    # feature extraction, here TF*IDF and relative position of first occurrence
    extractor.feature_extraction(df=df_counts)

    # candidate classification, here using a Naïve Bayes classifier
    extractor.classify_candidates(model='/path/to/model/file')

    # N-best selection, here the 10 highest scored candidates
    keyphrases = extractor.get_n_best(n=10)

### Provided supervised models and Document Frequency (DF) counts

The `models/` directory contains already trained models for Kea and WINGNUS as
well as DF counts computed on the [SemEval-2010 benchmark dataset][8].

### Using pke as a command line tool

A command line tool (`cmd_pke.py`) is also provided and allows users to perform 
keyphrase extraction without any knowledge of the Python programming language. A
minimal example of use is given below:

    python cmd_pke.py -i /path/to/input -f raw -o /path/to/output -a TopicRank

Here, unsupervised keyphrase extraction using TopicRank is performed on a raw
text input file, and the top ranked keyphrase candidates are outputted into a
file.

### Computing Document Frequency (DF) counts (required for some models)

Before using some keyphrase extraction algorithms (i.e. TfIdf, KP-Miner, Kea,
WINGNUS), one need to compute DF counts from a collection of documents. Such
counts can be computed as:

    from pke import compute_document_frequency
    from string import punctuation

    # path to the collection of documents
    input_dir = '/path/to/input/documents'

    # path to the DF counts dictionary, saved as a gzip tab separated values
    output_file = '/path/to/output/'

    # compute df counts and store stem -> weight values
    compute_document_frequency(input_dir=input_dir,
                               output_file=output_file,
                               format="corenlp",            # input files format
                               use_lemmas=False,    # do not use Stanford lemmas
                               stemmer="porter",            # use porter stemmer
                               stoplist=list(punctuation),            # stoplist
                               delimiter='\t',            # tab separated output
                               extension='xml',          # input files extension
                               n=5)              # compute n-grams up to 5-grams

DF counts are stored as a ngram tab count file. The number of documents in the
collection, used to compute Inverse Document Frequency (IDF) weigths, is stored
as an extra line --NB_DOC-- tab number_of_documents. Below is an example of such
a file:

    --NB_DOC--  100
    greedi alloc  1
    sinc trial structur 1
    complex question  1
    [...]

A documented example is described in [`compute-df-counts.py`][9] within the
`examples/` directory.

### Training supervised models

Here is a minimal example for training a new Kea model:

    import pke

    # load the Document Frequency (DF) weights file
    df = pke.load_document_frequency_file('/path/to/file')

    # train a new Kea model
    pke.train_supervised_model(input_dir='/path/to/input/documents/',
                               reference_file='/path/to/reference/file',
                               model_file='/path/to/model/file',
                               df=df,
                               model=pke.Kea())

A documented example is described in [`train-model.py`][10] within the
`examples/` directory.


[1]: http://aclweb.org/anthology/C08-1122.pdf
[2]: http://aclweb.org/anthology/I13-1062.pdf
[3]: http://aclweb.org/anthology/S10-1041.pdf
[4]: https://www.cl.cam.ac.uk/archive/ksj21/ksjdigipapers/jdoc72.pdf
[5]: http://arxiv.org/ftp/cs/papers/9902/9902007.pdf
[6]: http://aclweb.org/anthology/S10-1035.pdf
[7]: https://github.com/boudinfl/pke/blob/master/examples/keyphrase-extraction.py
[8]: http://aclweb.org/anthology/S10-1004.pdf
[9]: https://github.com/boudinfl/pke/blob/master/examples/compute-df-counts.py
[10]: https://github.com/boudinfl/pke/blob/master/examples/train-model.py

