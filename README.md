# `pke` - python keyphrase extraction

`pke` is an open source python-based keyphrase extraction toolkit. It provides
an end-to-end keyphrase extraction pipeline in which each component can be
easily modified or extented to develop new approaches. `pke` also allows for 
easy benchmarking of state-of-the-art keyphrase extraction approaches, and 
ships with supervised models trained on the
[SemEval-2010 dataset](http://aclweb.org/anthology/S10-1004).

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

## Code documentation

For code documentation, please visit [pke.readthedocs.io](http://pke.rtfd.io>).
