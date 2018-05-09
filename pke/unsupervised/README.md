# Parameterized example for each unsupervised model

## TfIdf
	
```python
from pke.unsupervised import TfIdf

# 1. create a TfIdf extractor.
extractor = TfIdf(input_file='C-1.xml')

# 2. load the content of the document.
extractor.read_document(format='corenlp')

# 3. select n-grams as keyphrase candidates
#    n = 5 (length of n-grams)
#    stoplist = ['.', ...] (candidates containing these are removed)
extractor.candidate_selection(n=n, stoplist=stoplist)

# 4. weight the candidates using a `tf` x `idf`
#    df_counts = {'--NB_DOC--': 3, word1': 3, 'word2': 1, 'word3': 2}
extractor.candidate_weighting(df=df_counts)

# 5. get the 10-highest scored candidates as keyphrases
keyphrases = extractor.get_n_best(n=10)
```

### KPMiner

```python
from pke.unsupervised import KPMiner

# 1. create a KPMiner extractor. 
#    language='english' (language used for the stoplist)
extractor = KPMiner(input_file='path/to/input.xml',
    language='english')

# 2. load the content of the document.
extractor.read_document(format='corenlp')

# 3. select {1, 5}-grams that do not contain punctuation marks or
#    stopwords as keyphrase candidates.
# >>> lasf = 5 (least allowable seen frequency)
# >>> cutoff = 200 (nb of words after which candidates are filtered out)
extractor.candidate_selection(lasf=lasf,
      cutoff=cutoff)

# 4. weight the candidates using KPMiner weighting function.
# >>> df_counts = {'--NB_DOC--': 3, word1': 3, 'word2': 1, 'word3': 2}
# >>> alpha = 2.3
# >>> sigma = 3.0
extractor.candidate_weighting(df=df_counts,
      alpha=alpha,
      sigma=sigma)

# 5. get the 10-highest scored candidates as keyphrases
keyphrases = extractor.get_n_best(n=10)
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
# >>>                                   redundancy_removal=redundancy_removal,
# >>>                                   stemming=stemming)
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
# >>>                               method=method,
# >>>                               heuristic=heuristic)

# get the 10-highest scored candidates as keyphrases
keyphrases = extractor.get_n_best(n=10)

# available parameters are whether redundant candidates are filtered out
# (default to False) and if stemming is applied to candidates (default
# to True)
# >>> redundancy_removal=True
# >>> stemming=False
# >>> keyphrases = extractor.get_n_best(n=10,
# >>>                                   redundancy_removal=redundancy_removal,
# >>>                                   stemming=stemming)
```