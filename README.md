# `pke` - python keyphrase extraction

`pke` is an **open source** python-based **keyphrase extraction** toolkit. It
provides an end-to-end keyphrase extraction pipeline in which each component can
be easily modified or extended to develop new models. `pke` also allows for 
easy benchmarking of state-of-the-art keyphrase extraction models, and 
ships with supervised models trained on the
[SemEval-2010 dataset](http://aclweb.org/anthology/S10-1004).

![python-package workflow](https://github.com/boudinfl/pke/actions/workflows/python-package.yml/badge.svg)

## Table of Contents

* [Installation](#installation)
* [Minimal example](#minimal-example)
* [Getting started](#getting-started)
* [Implemented models](#implemented-models)
* [Citing pke](#citing-pke)

## Installation

To pip install `pke` from github:

```bash
pip install git+https://github.com/boudinfl/pke.git
```

`pke` relies on `spacy` for text processing and requires [models](https://spacy.io/usage/models) to be installed: 

```bash
python -m spacy download en_core_web_sm # download the english model
```

## Minimal example

`pke` provides a standardized API for extracting keyphrases from a document.
Start by typing the 5 lines below. For using another model, simply replace
`pke.unsupervised.TopicRank` with another model ([list of implemented models](#implemented-models)).

```python
import pke

# initialize keyphrase extraction model, here TopicRank
extractor = pke.unsupervised.TopicRank()

# load the content of the document, here document is expected to be a simple 
# test string and preprocessing is carried out using spacy
extractor.load_document(input='text', language='en')

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

## Getting started

To get your hands dirty with `pke`, we invite you to try our tutorials out.

|                          Name                   |     Link     |
| ----------------------------------------------  |  ----------  |
| Getting started with `pke` and keyphrase extraction | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/keyphrasification/hands-on-with-pke/blob/main/part-1-graph-based-keyphrase-extraction.ipynb) |
| Model parameterization                          | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/keyphrasification/hands-on-with-pke/blob/main/part-2-parameterization.ipynb) |
| Benchmarking models                             | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/keyphrasification/hands-on-with-pke/blob/main/part-3-benchmarking-models.ipynb) |

## Implemented models

`pke` currently implements the following keyphrase extraction models:

* Unsupervised models
  * Statistical models
    * FirstPhrases
    * TfIdf
    * YAKE [(Campos et al., 2020)](https://doi.org/10.1016/j.ins.2019.09.013)
  * Graph-based models
    * TextRank [(Mihalcea and Tarau, 2004)](http://www.aclweb.org/anthology/W04-3252.pdf)
    * SingleRank  [(Wan and Xiao, 2008)](http://www.aclweb.org/anthology/C08-1122.pdf)
    * TopicRank [(Bougouin et al., 2013)](http://aclweb.org/anthology/I13-1062.pdf)
    * TopicalPageRank [(Sterckx et al., 2015)](http://users.intec.ugent.be/cdvelder/papers/2015/sterckx2015wwwb.pdf)
    * PositionRank [(Florescu and Caragea, 2017)](http://www.aclweb.org/anthology/P17-1102.pdf)
    * MultipartiteRank [(Boudin, 2018)](https://arxiv.org/abs/1803.08721)
* Supervised models
  * Feature-based models
    * Kea [(Witten et al., 2005)](https://www.cs.waikato.ac.nz/ml/publications/2005/chap_Witten-et-al_Windows.pdf)

## Citing pke

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
