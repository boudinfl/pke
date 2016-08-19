# pke - python keyphrase extraction

pke currently implements the following keyphrase extraction models:

- Unsupervised models
  - SingleRank [(Xiaojun and Jianguo, 2008)][1]
  - TopicRank [(Bougouin et al., 2013)][2]
  - KP-Miner [(El-Beltagy and Rafea, 2010)][3]
  - TF*IDF [(Spärck Jones, 1972)][4]

- Supervised models
  - Kea [(Witten et al., 1999)][5]
  - WINGNUS [(Nguyen and Luong, 2010)][6]

## Requirements

    numpy
    scipy
    nltk
    networkx
    sklearn

## Installation

To install this module:

    pip install git+https://github.com/boudinfl/pke.git

## Usage

### Input formats

Three input formats are currently supported:
  - raw text: text preprocessing (i.e. tokenization, sentence splitting and
    POS-tagging) is carried out using nltk.
  - preprocessed text: whitespace-separated POS-tagged tokens, one sentence per
    line.
  - Stanford XML CoreNLP format: output file produced using the annotators
    tokenize, ssplit, pos and lemma. Document logical structure information can
    by specified by incorporating attributes into the sentence elements of the
    CoreNLP XML format.

Detailed examples of input formats are provided in the examples/ directory.

Default language in pke is English and default candidate selection methods are
based on the PTB tagset.

### Minimal example: unsupervised keyphrase extraction using TopicRank

All the unsupervised models can be used by typing the 5 lines below. For using
another model, simply replace TopicRank with SingleRank, KPMiner, TfIdf, etc.

    import pke

    # initialize TopicRank
    extractor = pke.TopicRank(input_file='/path/to/input')

    # load the content of the document, preprocessing is carried out using nltk
    extractor.read_raw_document()

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
    df = pke.load_document_frequency_file(input_file='/path/to/file')

    # load the content of the document, preprocessing is carried out using nltk
    extractor.read_raw_document()

    # candidate selection, here 1-3-grams that do not begin/end with a stopword
    extractor.candidate_selection()

    # feature extraction, here TF*IDF and relative position of first occurrence
    extractor.feature_extraction(df=df)

    # candidate classification, here using a Naïve Bayes classifier
    extractor.classify_candidates(model='/path/to/model/file')

    # N-best selection, here the 10 highest scored candidates
    keyphrases = extractor.get_n_best(n=10)

### Provided supervised models and Document Frequency (DF) counts

The models/ directory contains already trained models for Kea and WINGNUS as
well as DF counts computed on the SemEval-2010 benchmark dataset.

### Computing Document Frequency (DF) weights (required for some models)

Before using some keyphrase extraction algorithms (i.e. TfIdf, KP-Miner, Kea,
WINGNUS), one need to compute DF weights from a collection of documents. Such
weights can be computed as:

    from pke import compute_document_frequency
    from string import punctuation

    # path to the collection of documents
    input_dir = '/path/to/input/documents'

    # path to the DF weights dictionary, saved as a gzip tab separated values
    output_file = '/path/to/output/'

    # compute df weights and store stem -> weight values
    compute_document_frequency(input_dir=input_dir,
                               output_file=output_file,
                               format="corenlp",            # input files format
                               use_lemmas=False,    # do not use Stanford lemmas
                               stemmer="porter",            # use porter stemmer
                               stoplist=list(punctuation),            # stoplist
                               delimiter='\t',            # tab separated output
                               extension='xml',          # input files extension
                               n=5)              # compute n-grams up to 5-grams

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


[1]: http://aclweb.org/anthology/C08-1122.pdf
[2]: http://aclweb.org/anthology/I13-1062.pdf
[3]: http://aclweb.org/anthology/S10-1041.pdf
[4]: https://www.cl.cam.ac.uk/archive/ksj21/ksjdigipapers/jdoc72.pdf
[5]: http://arxiv.org/ftp/cs/papers/9902/9902007.pdf
[6]: http://aclweb.org/anthology/S10-1035.pdf
