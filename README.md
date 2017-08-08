# `pke` - python keyphrase extraction

`pke` is an open source python-based keyphrase extraction toolkit. It provides
an end-to-end keyphrase extraction pipeline in which each component can be
easily modified or extented to develop new approaches. `pke` also allows for 
easy benchmarking of state-of-the-art keyphrase extraction approaches, and 
ships with supervised models trained on the
[SemEval-2010 dataset](http://aclweb.org/anthology/S10-1004).

`pke` works only for Python 2.x at the moment.

If you use `pke`, please cite the following paper:

  * [Florian Boudin. **pke: an open source python-based keyphrase extraction toolkit**, *Proceedings of COLING 2016, the 26th International Conference on Computational Linguistics: System Demonstrations*](http://aclweb.org/anthology/C16-2015). [[bibtex](http://aclweb.org/anthology/C16-2015.bib)]

## Table of Contents

* [Installation](#installation)
* [Usage](#usage)
  - [Minimal example](#minimal-example)
  - [Input formats](#input-formats)
  - [Provided models](#provided-models)
  - [Document Frequency counts](#document-frequency-counts)
  - [Training supervised models](#training-supervised-models)
  - [Non English languages](#non-english-languages)
* [Benchmarking](#benchmarking)
* [Code documentation](#code-documentation)

## Installation

The following modules are required:

```bash
numpy
scipy
nltk
networkx
sklearn
```

To pip install `pke` from github:

```bash
pip install git+https://github.com/boudinfl/pke.git
```

## Usage

### Minimal example

`pke` provides a standardized API for keyphrases from a document. Start by
typing the 5 lines below. For using another model, simply replace TopicRank with
Kea, KPMiner, TfIdf, etc. 

```python
import pke

# initialize keyphrase extraction model, here TopicRank
extractor = pke.TopicRank(input_file='/path/to/input')

# load the content of the document, here document is expected to be in raw
# format (i.e. a simple text file) and preprocessing is carried out using nltk
extractor.read_document(format='raw')

# keyphrase candidate selection, in the case of TopicRank: sequences of nouns
# and adjectives
extractor.candidate_selection()

# candidate weighting, in the case of TopicRank: using a random walk algorithm
extractor.candidate_weighting()

# N-best selection, keyphrases contains the 10 highest scored candidates as
# (keyphrase, score) tuples
keyphrases = extractor.get_n_best(n=10)
```

A detailed example is provided in the `examples/` directory.

### Input formats

`pke` currently supports the following input formats (examples of formatted
input files are provided in the `examples/` directory):

1. *raw text*: text preprocessing (i.e. tokenization, sentence splitting and 
   POS-tagging) is carried out using nltk. 

   Example of raw text:

   ```
   Efficient discovery of grid services is essential for the success of
   grid computing. [...]
   ```

   To read raw text document:

   ```python
   extractor.read_document(format='raw')
   ```

2. *preprocessed text*: whitespace-separated POS-tagged tokens, one sentence per
   line.

   Example of preprocessed text:

   ```
   Efficient/NNP discovery/NN of/IN grid/NN services/NNS is/VBZ essential/JJ for/IN the/DT success/NN of/IN grid/JJ computing/NN ./.
   [...]
   ```

   To read preprocessed text document:

   ```python
   extractor.read_document(format='preprocessed')
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
   extractor.read_document(format='corenlp')
   ```

### Provided models

`pke` ships with a collection of already trained models and document frequency
counts that were computed on the training set of the SemEval-2010 benchmark
dataset. These models are located into the `pke/models/` directory.

For details about the provided models, see [pke/models/README.md](pke/models/README.md).

**These already trained models/DF counts are used by default if no parameters
are given.**

### Document Frequency counts

`pke` ships with document frequency counts computed on the SemEval-2010
benchmark dataset. These counts are used in various models (TfIdf, KP-Miner,
Kea and WINGNUS). The following code illustrates how to compute new counts
from another (or a larger) document collection:

```python
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
```

DF counts are stored as a ngram tab count file. The number of documents in the
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
extractor = pke.TfIDF(input_file='/path/to/input')

# load the DF counts from file
df_counts = pke.load_document_frequency_file(input_file='/path/to/file')

# load the content of the document
extractor.read_document(format='raw')

# keyphrase candidate selection
extractor.candidate_selection()

# candidate weighting with the provided DF counts
extractor.candidate_weighting(df=df_counts)

# N-best selection, keyphrases contains the 10 highest scored candidates as
# (keyphrase, score) tuples
keyphrases = extractor.get_n_best(n=10)
```

A detailed example for computing new DF counts is given in
`examples/compute-df-counts.py`.

### Training supervised models

Here is a minimal example for training a new supervised model:

```python
import pke

# load the DF counts from file
df_counts = pke.load_document_frequency_file('/path/to/file')

# train a new Kea model
pke.train_supervised_model(input_dir='/path/to/input/documents/',
                           reference_file='/path/to/reference/file',
                           model_file='/path/to/model/file',
                           df=df_counts,
                           model=pke.Kea()) # here a we train a Kea model
```

The training data consists of a set of documents along with a reference file
containing annotated keyphrases in the SemEval-2010 [format](http://docs.google.com/Doc?id=ddshp584_46gqkkjng4).

A detailed example for training a supervised model is given in
`examples/train-model.py`.

### Non English languages

While the default language in `pke` is English, extracting keyphrases from
documents in other languages is easily achieved by inputting already
preprocessed documents, and setting the `language` parameter to the desired
language. The only language dependent resources used in `pke` are the stoplist
and the stemming algorithm from `nltk` that is available in
[11 languages](http://www.nltk.org/_modules/nltk/corpus.html).

Given an already preprocessed document (here in French):

```
France/NPP :/PONCT disparition/NC de/P Thierry/NPP Roland/NPP [...]
Le/DET journaliste/NC et/CC commentateur/NC sportif/ADJ Thierry/NPP [...]
Commentateur/NC mythique/ADJ des/P+D matchs/NC internationaux/ADJ [...]
[...]
```

Keyphrase extraction can then be performed by:

```python
import pke

# initialize TopicRank and set the language to French (used during candidate
# selection for filtering stopwords)
extractor = pke.TopicRank(input_file='/path/to/input', language='French')

# load the content of the document and perform French stemming (instead of
# Porter stemmer)
extractor.read_document(format='preprocessed', stemmer='french')

# keyphrase candidate selection, here sequences of nouns and adjectives
# defined by the French POS tags NPP, NC and ADJ
extractor.candidate_selection(pos=["NPP", "NC", "ADJ"])

# candidate weighting, here using a random walk algorithm
extractor.candidate_weighting()

# N-best selection, keyphrases contains the 10 highest scored candidates as
# (keyphrase, score) tuples
keyphrases = extractor.get_n_best(n=10)
```

## Benchmarking

We evaluate the performance of our re-implementations using the SemEval-2010
benchmark dataset. This dataset is composed of 244 scientific articles (144 in
training and 100 for test) collected from the ACM Digital Library (conference
and workshop papers). Document logical structure information, required to
compute features in the WINGNUS approach, is annotated with
[ParsCit](https://github.com/knmnyn/ParsCit). The [Stanford CoreNLP pipeline](http://stanfordnlp.github.io/CoreNLP/)
(tokenization, sentence splitting and POS-tagging) is then applied to the
documents from which irrelevant pieces of text (e.g. tables, equations,
footnotes) were filtered out. The dataset we use (lvl-2) can be found at
[https://github.com/boudinfl/semeval-2010-pre](https://github.com/boudinfl/semeval-2010-pre).

We follow the evaluation procedure used in the SemEval-2010 competition and
evaluate the performance of each implemented approach in terms of precision (P),
recall (R) and f-measure (F) at the top 10 keyphrases. We use the set of
combined (stemmed) author- and reader-assigned keyphrases as reference
keyphrases.

| Approach   | P@10 | R@10 | F@10 |
|------------|------|------|------|
| TfIdf      | 20.0 | 14.1 | 16.4 |
| TopicRank  | 15.6 | 10.8 | 12.6 |
| SingleRank |  2.2 |  1.5 |  1.8 |
| KP-Miner   | 24.1 | 17.0 | 19.8 |
| Kea        | 23.5 | 16.6 | 19.3 |
| WINGNUS    | 24.7 | 17.3 | 20.2 |

## Code documentation

For code documentation, please visit [pke.readthedocs.io](http://pke.rtfd.io).
