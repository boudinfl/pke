# Unsupervised models

## Statistical models

### [TfIdf](https://en.wikipedia.org/wiki/Tf-idf)
	
```python
from pke.unsupervised import TfIdf

# create a TfIdf extractor and set the input language to English (used
# for the stoplist in the candidate selection method). The input file
# is considered to be in Stanford XML CoreNLP.
extractor = TfIdf(input_file='C-1.xml', language='english')

# load the content of the document.
extractor.read_document(format='corenlp')

# select the keyphrase candidates, by default the 1-3-grams of words
# that do not contain punctuation marks or stopwords.
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

### 