# `pke` - python keyphrase extraction

`pke` is an **open source** python-based **keyphrase extraction** toolkit. It
provides an end-to-end keyphrase extraction pipeline in which each component can
be easily modified or extented to develop new models. `pke` also allows for 
easy benchmarking of state-of-the-art keyphrase extraction models, and 
ships with supervised models trained on the
[SemEval-2010 dataset](http://aclweb.org/anthology/S10-1004).

[![Build Status](https://travis-ci.org/boudinfl/pke.svg?branch=master)](https://travis-ci.org/boudinfl/pke)

## Table of Contents

* [Installation](#installation)
* [Minimal example](#minimal-example)
* [Usage](#usage)
  - [Input formats](#input-formats)
  - [Implemented models](#implemented-models)
  - [Computing and loading Document Frequency counts](#computing-and-loading-document-frequency-counts)
  - [Training supervised models](#training-supervised-models)
  - [Non English languages](#non-english-languages)
* [Code documentation](#code-documentation)
* [Citing `pke`](#citing-pke)

## Installation

To pip install `pke` from github:

```bash
pip install git+https://github.com/boudinfl/pke.git
```

## Minimal example

`pke` provides a standardized API for extracting keyphrases from a document.
Start by typing the 5 lines below. For using another model, simply replace
`pke.unsupervised.TopicRank` with another model ([list of implemented models](#implemented-models)).

```python
import pke

# initialize keyphrase extraction model, here TopicRank
extractor = pke.unsupervised.TopicRank()

# load the content of the document, here document is expected to be in raw
# format (i.e. a simple text file) and preprocessing is carried out using spacy
extractor.load_document(input='/path/to/input.txt', language='en')

# keyphrase candidate selection, in the case of TopicRank: sequences of nouns
# and adjectives (i.e. `(Noun|Adj)*`)
extractor.candidate_selection()

# candidate weighting, in the case of TopicRank: using a random walk algorithm
extractor.candidate_weighting()

# N-best selection, keyphrases contains the 10 highest scored candidates as
# (keyphrase, score) tuples
keyphrases = extractor.get_n_best(n=10)
```

A detailed example is provided in the [`examples/`](examples/) directory.

## Usage

### Input formats

`pke` currently supports the following input formats (examples of formatted
input files are provided in the [`examples/`](examples/) directory):

1. *raw text*: text pre-processing (i.e. tokenization, sentence splitting and 
   POS-tagging) is carried out using [spacy](https://spacy.io/). 

   Example of content from a raw text file:

   ```
   Efficient discovery of grid services is essential for the success of
   grid computing. [...]
   ```

   To read a document in raw text format:

   ```python
   import pke

   extractor = pke.unsupervised.TopicRank()
   extractor.load_document(input='/path/to/input.txt', language='en')
   ```

2. *input text*: same as *raw text*, text pre-processing is carried out
    using spacy.

   To read an input text:

   ```python
   import pke

   extractor = pke.unsupervised.TopicRank()
   text = u'Efficient discovery of grid services is essential for the [...]'
   extractor.load_document(input=text, language='en')
   ```

3. *Stanford XML CoreNLP*: output file produced using the annotators `tokenize`,
   `ssplit`, `pos` and `lemma`. Document logical structure information can by
   specified by incorporating attributes into the sentence elements of the
   CoreNLP XML format.

   Example of CoreNLP XML:

   ```xml
   <?xml version="1.0" encoding="UTF-8"?>
   <root>
     <document>
       <sentences>
         <sentence id="1" section="abstract" type="bodyText" confidence="0.925">
           <tokens>
             <token id="1">
               <word>Efficient</word>
               <lemma>efficient</lemma>
               <CharacterOffsetBegin>362</CharacterOffsetBegin>
               <CharacterOffsetEnd>371</CharacterOffsetEnd>
               <POS>JJ</POS>
             </token>
             <token id="2">
               <word>discovery</word>
               <lemma>discovery</lemma>
               <CharacterOffsetBegin>372</CharacterOffsetBegin>
               <CharacterOffsetEnd>381</CharacterOffsetEnd>
               <POS>NN</POS>
              </token>
   [...]
   ```

   Here, the document logical structure information is added to the CoreNLP XML
   output by the use of the `section`, `type` and `confidence` attributes. We
   use the classification categories proposed by [Luong et al. (2012)](https://www.comp.nus.edu.sg/~kanmy/papers/ijdls-SectLabel.pdf).
   In `pke`, document logical structure information is exploited by the WINGNUS
   model and the following values are handled:

   ```xml
   section="title|abstract|introduction|related work|conclusions"
   type="sectionHeader|subsectionHeader|subsubsectionHeader|bodyText"
   ```

   To read a CoreNLP XML document:

   ```python
   import pke

   extractor = pke.unsupervised.TopicRank()
   extractor.load_document(input='/path/to/input.xml')
   ```

### Implemented models

`pke` currently implements the following keyphrase extraction models:

* Unsupervised models
  * Statistical models
    * [TfIdf](https://boudinfl.github.io/pke/build/html/unsupervised.html#tfidf)
    * [KPMiner](https://boudinfl.github.io/pke/build/html/unsupervised.html#kpminer) [(El-Beltagy and Rafea, 2010)](http://www.aclweb.org/anthology/S10-1041.pdf)
    * [YAKE](https://boudinfl.github.io/pke/build/html/unsupervised.html#yake) [(Campos et al., 2018)](https://repositorio.inesctec.pt/bitstream/123456789/7623/1/P-00N-NF5.pdf)
  * Graph-based models
    * [TextRank](https://boudinfl.github.io/pke/build/html/unsupervised.html#textrank) [(Mihalcea and Tarau, 2004)](http://www.aclweb.org/anthology/W04-3252.pdf)
    * [SingleRank](https://boudinfl.github.io/pke/build/html/unsupervised.html#singlerank) [(Wan and Xiao, 2008)](http://www.aclweb.org/anthology/C08-1122.pdf)
    * [TopicRank](https://boudinfl.github.io/pke/build/html/unsupervised.html#topicrank) [(Bougouin et al., 2013)](http://aclweb.org/anthology/I13-1062.pdf)
    * [TopicalPageRank](https://boudinfl.github.io/pke/build/html/unsupervised.html#topicalpagerank) [(Sterckx et al., 2015)](http://users.intec.ugent.be/cdvelder/papers/2015/sterckx2015wwwb.pdf)
    * [PositionRank](https://boudinfl.github.io/pke/build/html/unsupervised.html#positionrank) [(Florescu and Caragea, 2017)](http://www.aclweb.org/anthology/P17-1102.pdf)
    * [MultipartiteRank](https://boudinfl.github.io/pke/build/html/unsupervised.html#multipartiterank) [(Boudin, 2018)](https://arxiv.org/abs/1803.08721)
* Supervised models
  * Feature-based models
    * Kea [(Witten et al., 2005)](https://www.cs.waikato.ac.nz/ml/publications/2005/chap_Witten-et-al_Windows.pdf)
    * WINGNUS [(Nguyen and Luong, 2010)](http://www.aclweb.org/anthology/S10-1035.pdf)


### Computing and loading Document Frequency counts

`pke` ships with document frequency counts computed on the SemEval-2010
benchmark dataset. These counts are used in various models (for example TfIdf
and Kea). The following code illustrates how to compute new document frequency
counts from another (or a larger) document collection:

```python
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
                           extension='xml',               # input file extension
                           language='en',                 # language of files
                           normalization="stemming",      # use porter stemmer
                           stoplist=stoplist)   
```

A fully parametrized example for DF counts computation is available in the file
[`examples/compute-df-counts.py`](examples/compute-df-counts.py).

DF counts are stored as a n-gram tab count file. The number of documents in the
collection, used to compute Inverse Document Frequency (IDF) weigths, is stored
as an extra line --NB_DOC-- tab number_of_documents. Below is an example of such
a file:

```bash
--NB_DOC--  100
greedi alloc  1
sinc trial structur 1
complex question  1
[...]
```

Newly computed DF counts should be loaded and given as parameter to 
`candidate_weighting()` for unsupervised models and `feature_extraction()` for
supervised models:

```python
import pke

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
```

### Training supervised models

`pke` ships with a collection of already trained models (for supervised
keyphrase extraction approaches) and document frequency counts that were
computed on the training set of the SemEval-2010 benchmark dataset. These
are located into the `pke/models/` directory.
**These already trained models/DF counts are used by default if no parameters
are given.**

The following snippet of code illustrates how to train a new supervised model: 

```python
import pke

# load the DF counts from file
df_counts = pke.load_document_frequency_file(input_file='/path/to/df_counts')

# train a new Kea model
pke.train_supervised_model(input_dir='/path/to/collection/of/documents/',
                           reference_file='/path/to/reference/file',
                           model_file='/path/to/model/file',
                           df=df_counts,
                           model=pke.supervised.Kea())
```

The training data consists of a set of documents along with a reference file
containing annotated keyphrases in the SemEval-2010 [format](http://docs.google.com/Doc?id=ddshp584_46gqkkjng4).

A detailed and parametrized example for training and testing a supervised model
is provided in the [`examples/training_and_testing_a_kea_model/`](examples/training_and_testing_a_kea_model/) directory.

### Non English languages

`pke` uses `spacy` to pre-process document. As such, all the languages that are
supported in `spacy` can be processed.

Keyphrase extraction can then be performed by:

```python
import pke

# initialize TopicRank and set the language to French (used during candidate
# selection for filtering stopwords)
extractor = pke.unsupervised.TopicRank()

# load the content of the document and perform French stemming (instead of
# Porter stemmer)
extractor.load_document(input='/path/to/input', language='french')

# keyphrase candidate selection, here sequences of nouns and adjectives
# defined by the Universal PoS tagset
extractor.candidate_selection(pos={"NOUN", "PROPN" "ADJ"})

# candidate weighting, here using a random walk algorithm
extractor.candidate_weighting()

# N-best selection, keyphrases contains the 10 highest scored candidates as
# (keyphrase, score) tuples
keyphrases = extractor.get_n_best(n=10)
```

## Code documentation

For code documentation, please visit [https://boudinfl.github.io/pke/](https://boudinfl.github.io/pke/).

## Citing `pke`

If you use `pke`, please cite the following paper:

```
@InProceedings{boudin:2016:COLINGDEMO,
  author    = {Boudin, Florian},
  title     = {pke: an open source python-based keyphrase extraction toolkit},
  booktitle = {Proceedings of COLING 2016, the 26th International Conference on Computational Linguistics: System Demonstrations},
  month     = {December},
  year      = {2016},
  address   = {Osaka, Japan},
  pages     = {69--73},
  url       = {http://aclweb.org/anthology/C16-2015}
}
```
