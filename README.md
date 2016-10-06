# `pke` - python keyphrase extraction

`pke` is an open source python-based keyphrase extraction toolkit. It provides
an end-to-end keyphrase extraction pipeline in which each component can be
easily modified or extented to develop new approaches. `pke` also allows for 
easy benchmarking of state-of-the-art keyphrase extraction approaches, and 
ships with supervised models trained on the
[SemEval-2010 dataset](http://aclweb.org/anthology/S10-1004).

## Requirements

```
numpy
scipy
nltk
networkx
sklearn
```

## Installation

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

```python
extractor.read_document(format='raw')
```

2. *preprocessed text*: whitespace-separated POS-tagged tokens, one sentence per
   line.

```python
extractor.read_document(format='preprocessed')
```

3. *Stanford XML CoreNLP*: output file produced using the annotators `tokenize`,
	`ssplit`, `pos` and `lemma`. Document logical structure information can by
	specified by incorporating attributes into the sentence elements of the
	CoreNLP XML format.

```python
extractor.read_document(format='corenlp')
```

### Provided models

`pke` ships with a collection of already trained models and document frequency
counts that were computed on the training set of the SemEval-2010 benchmark
dataset. These models are located into the `pke/models/` directory.

## Code documentation

For code documentation, please visit [pke.readthedocs.io](http://pke.rtfd.io>).
