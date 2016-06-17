# pke - python keyphrase extraction

pke currently implements the following keyphrase extraction models:

- Unsupervised models
  - SingleRank [(Xiaojun and Jianguo, 2008)][1]
  - TopicRank [(Bougouin et al., 2013)][2]
  - KP-Miner [(El-Beltagy and Rafea, 2010)][3]
  - TF*IDF
- Supervised models
  - Kea [(Witten et al., 1999)][4]
  - WINGNUS [(Nguyen and Luong, 2010)][5]

## Requirements

    numpy
    scipy
    nltk
    networkx

## Installation

To install this module:

    pip install git+https://github.com/boudinfl/pke.git

## Usage

Note that input files must be either in Stanford XML CoreNLP format
(format=corenlp) or one tokenized/POS_tag sentence per line format (format=pre).

### Computing DF weights (required for some models)

Before using some keyphrase extraction algorithms (e.g. TfIdf, KP-Miner), one 
need to compute DF weights from a collection of documents. Such weights can
be computed as:

    from pke import compute_document_frequency
    from string import punctuation

    # path to the collection of documents
    input_dir = '/path/to/input/documents'

    # path to the DF weights dictionary, saved as a gzip tab separated values
    output_file = '/path/to/output/'

    # compute df weights and store stem -> weigth values
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

For supervised models, one need to build their classification model first 
(example for Kea):

    from pke import train_supervised_model
    from pke import load_document_frequency_file
    from pke import Kea

    # loads the DF weights file
    df = load_document_frequency_file('/path/to/file', delimiter='\t')

    # path to the collection of documents
    input_dir = '/path/to/input/documents'

    # path to the keyphrase reference file
    reference_file = '/path/to/reference/file'

    # path to the stored model file
    model_file = '/path/to/model/file'

    # build classification model
    train_supervised_model(input_dir=input_dir,
                           reference_file=reference_file,
                           model_file=model_file,
                           df=df,
                           model=Kea())


### Unsupervised models

All the unsupervised models can be used by typing the 5 lines below. For using
another model, simply replace SingleRank with KPMiner, TfIdf, TopicRank, etc.

    # this example uses SingleRank
    from pke import SingleRank 

    # create an unsupervised object
    extractor = SingleRank(input_file='/path/to/input')

    # load the content of the document, here in CoreNLP XML format
    # the use_lemmas parameter allows to choose using CoreNLP lemmas or stems 
    # computed using nltk
    extractor.read_corenlp_document(use_lemmas=False)

    # select the keyphrase candidates, for SingleRank the longest sequences of 
    # nouns and adjectives
    extractor.candidate_selection()

    # weight the candidates using a random walk
    extractor.candidate_weighting()

    # print the n-highest (10) scored candidates
    print (';'.join([u for u, v in extractor.get_n_best(n=10)])).encode('utf-8')


### Supervised models

Here is an example of supervised model (Kea):


    # this example uses SingleRank
    from pke import Kea 

    # loads the DF weights file
    df = load_document_frequency_file('/path/to/file', delimiter='\t')

    # create an supervised object
    extractor = Kea(input_file='/path/to/input')

    # load the content of the document, here in CoreNLP XML format
    # the use_lemmas parameter allows to choose using CoreNLP lemmas or stems 
    # computed using nltk
    extractor.read_corenlp_document(use_lemmas=False)

    # select the keyphrase candidates
    extractor.candidate_selection()

    # extract candidate features
    extractor.feature_extraction(df=df)

    # classifify candidates
    extractor.classify_candidates(model='/path/to/model/file')

    # print the n-highest (10) scored candidates
    print (';'.join([u for u, v in extractor.get_n_best(n=10)])).encode('utf-8')


[1]: http://aclweb.org/anthology/C08-1122.pdf
[2]: http://aclweb.org/anthology/I13-1062.pdf
[3]: http://aclweb.org/anthology/S10-1041.pdf
[4]: http://arxiv.org/ftp/cs/papers/9902/9902007.pdf
[5]: http://aclweb.org/anthology/S10-1035.pdf
