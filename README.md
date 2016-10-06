# `pke` - python keyphrase extraction

`pke` is an open source python-based keyphrase extraction toolkit. It provides
an end-to-end keyphrase extraction pipeline in which each component can be
easily modified or extented to develop new approaches. `pke` also allows for 
easy benchmarking of state-of-the-art keyphrase extraction approaches, and 
ships with supervised models trained on the
[SemEval-2010 dataset](http://aclweb.org/anthology/S10-1004).

## Table of Contents

* [Installation](#Installation)
  - [Requirements](#Requirements)
  - [pip install](#pip-install-from-github)
* [Usage](#usage)

## Installation

### Requirements

```
numpy
scipy
nltk
networkx
sklearn
```

### pip install from github

To install this module:

```bash
pip install git+https://github.com/boudinfl/pke.git
```

## Usage

### Minimal example

Start extracting keyphrases from a document by typing the 5 lines below. For
using another model, simply replace TopicRank with Kea, KPMiner, TfIdf,
etc. 

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

### Input formats

`pke` currently supports the following input formats:

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

   To read a CoreNLP XML document:

   ```python
   extractor.read_document(format='corenlp')
   ```

### Provided models

`pke` ships with a collection of already trained models and document frequency
counts that were computed on the training set of the SemEval-2010 benchmark
dataset. These models are located into the `pke/models/` directory.

For details about the provided models, see [pke/models/README.md](pke/models/README.md).

## Code documentation

For code documentation, please visit [pke.readthedocs.io](http://pke.rtfd.io>).
