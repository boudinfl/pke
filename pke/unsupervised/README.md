# Unsupervised models

## Statistical models

### [TfIdf](https://en.wikipedia.org/wiki/Tf-idf)
	
```python
from pke.unsupervised import TfIdf

# create a TfIdf extractor. The input file is considered to be in 
# Stanford XML CoreNLP.
extractor = TfIdf(input_file='C-1.xml')

# load the content of the document.
extractor.read_document(format='corenlp')

# select the keyphrase candidates, by default the 1-3-grams of words
# that do not contain punctuation marks.
extractor.candidate_selection()

# available parameters are the length of the n-grams and the stoplist
# for filtering candidates.
# >>> n = 5
# >>> stoplist = ['the', 'of', '.', '?', ...]
# >>> extractor.candidate_selection(n=n, stoplist=stoplist)

# weight the candidates using a `term frequency` x `inverse document
# frequency`, by defaults the document counts (df) are those computed
# on the training set of the SemEval-2010 dataset.
extractor.candidate_weighting()

# available parameters are the `df` counts that can be provided to the 
# weighting function.
# >>> counts = {'--NB_DOC--': 3, word1': 3, 'word2': 1, 'word3': 2}
# >>> extractor.candidate_weighting(df=counts)

# get the 10-highest scored candidates as keyphrases
keyphrases = extractor.get_n_best(n=10)

# available parameters are whether redundant candidates are filtered out
# (default to False) and if stemming is applied to candidates (default
# to True)
# >>> redundancy_removal=True
# >>> stemming=False
# >>> keyphrases = extractor.get_n_best(n=10,
# >>>                             redundancy_removal=redundancy_removal,
# >>>                             stemming=stemming)
```

### [KPMiner](http://www.aclweb.org/anthology/S10-1041.pdf)

```python
from pke.unsupervised import KPMiner

# create a KPMiner extractor and set the input language to English (used
# for the stoplist in the candidate selection method). The input file
# is considered to be in Stanford XML CoreNLP.
extractor = KPMiner(input_file='C-1.xml', language='english')

# load the content of the document.
extractor.read_document(format='corenlp')

# select the keyphrase candidates, by default the 1-5-grams of words
# that do not contain punctuation marks or stopwords. Candidates
# occurring less than 3 times or after the 400th word are filtered out.
extractor.candidate_selection()

# available parameters are the least allowable seen frequency and the 
# number of words after which candidates are filtered out.
# >>> lasf = 5
# >>> cutoff = 123
# >>> extractor.candidate_selection(lasf=lasf, cutoff=cutoff)

# weight the candidates using KPMiner weighting function.
extractor.candidate_weighting()

# available parameters are the `df` counts that can be provided to the 
# weighting function and the sigma and alpha values of the weighting
# function.
# >>> counts = {'--NB_DOC--': 3, word1': 3, 'word2': 1, 'word3': 2}
# >>> alpha = 2.3
# >>> sigma = 3.0
# >>> extractor.candidate_weighting(df=counts, alpha=alpha, sigma=sigma)

# get the 10-highest scored candidates as keyphrases
keyphrases = extractor.get_n_best(n=10)

# available parameters are whether redundant candidates are filtered out
# (default to False) and if stemming is applied to candidates (default
# to True)
# >>> redundancy_removal=True
# >>> stemming=False
# >>> keyphrases = extractor.get_n_best(n=10,
# >>>     redundancy_removal=redundancy_removal,
# >>>     stemming=stemming)
```

## Graph-based models

### [SingleRank](http://www.aclweb.org/anthology/C08-1122.pdf)

```python
from pke.unsupervised import SingleRank

# create a SingleRank extractor. The input file is considered to be in 
# Stanford XML CoreNLP.
extractor = SingleRank(input_file='C-1.xml')

# load the content of the document.
extractor.read_document(format='corenlp')

# select the keyphrase candidates, by default the longest sequences of 
# nouns and adjectives that do not contain punctuation marks or
# stopwords.
extractor.candidate_selection()

# available parameters are the Part-Of-Speech tags for selecting the
# sequences of words and the stoplist for filtering candidates.
# >>> pos = ["NN", "JJ"]
# >>> stoplist = ['the', 'of', '.', '?', ...]
# >>> extractor.candidate_selection(pos=pos, stoplist=stoplist)

# weight the candidates using a random walk.
extractor.candidate_weighting()

# available parameters are the window within the sentence for connecting
# two words in the graph. The set of valid pos for words to be
# considered as nodes in the graph.
# >>> window = 5
# >>> pos = set(["NN", "JJ"])
# >>> extractor.candidate_weighting(window=window, pos=pos)

# get the 10-highest scored candidates as keyphrases
keyphrases = extractor.get_n_best(n=10)

# available parameters are whether redundant candidates are filtered out
# (default to False) and if stemming is applied to candidates (default
# to True)
# >>> redundancy_removal=True
# >>> stemming=False
# >>> keyphrases = extractor.get_n_best(n=10,
# >>>     redundancy_removal=redundancy_removal,
# >>>     stemming=stemming)
```

### [TopicRank](http://aclweb.org/anthology/I13-1062.pdf)

```python
from pke.unsupervised import TopicRank

# create a TopicRank extractor. The input file is considered to be in 
# Stanford XML CoreNLP.
extractor = TopicRank(input_file='C-1.xml')

# load the content of the document.
extractor.read_document(format='corenlp')

# select the keyphrase candidates, by default the longest sequences of 
# nouns and adjectives that do not contain punctuation marks or
# stopwords.
extractor.candidate_selection()

# available parameters are the Part-Of-Speech tags for selecting the
# sequences of words and the stoplist for filtering candidates.
# >>> pos = ["NN", "JJ"]
# >>> stoplist = ['the', 'of', '.', '?', ...]
# >>> extractor.candidate_selection(pos=pos, stoplist=stoplist)

# weight the candidates using a random walk.
extractor.candidate_weighting()

# available parameters are the threshold for topic clustering, the
# linkage method and the heuristic for selecting candidates in topics.
# >>> threshold = 0.75
# >>> method = 'average'
# >>> heuristic = frequent
# >>> extractor.candidate_weighting(threshold=threshold,
    method=method,
    heuristic=heuristic)

# get the 10-highest scored candidates as keyphrases
keyphrases = extractor.get_n_best(n=10)

# available parameters are whether redundant candidates are filtered out
# (default to False) and if stemming is applied to candidates (default
# to True)
# >>> redundancy_removal=True
# >>> stemming=False
# >>> keyphrases = extractor.get_n_best(n=10,
# >>>     redundancy_removal=redundancy_removal,
# >>>     stemming=stemming)
```