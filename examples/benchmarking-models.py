# -*- coding: utf-8 -*-

import re
import spacy
import numpy as np
from tqdm import tqdm
from spacy.tokenizer import _get_regex_pattern
from datasets import load_dataset
from pke.unsupervised import *
from pke import compute_document_frequency, load_document_frequency_file

# load the inspec dataset
dataset = load_dataset('boudinfl/inspec', "all")

nlp = spacy.load("en_core_web_sm")

# Tokenization fix for in-word hyphens (e.g. 'non-linear' would be kept
# as one token instead of default spacy behavior of 'non', '-', 'linear')
re_token_match = _get_regex_pattern(nlp.Defaults.token_match)
re_token_match = f"({re_token_match}|\w+-\w+)"
nlp.tokenizer.token_match = re.compile(re_token_match).match

# populates a docs list with spacy doc objects
train_docs = []
for sample in tqdm(dataset['train']):
    train_docs.append(nlp(sample["title"]+". "+sample["abstract"]))

test_docs = []
for sample in tqdm(dataset['test']):
    test_docs.append(nlp(sample["title"]+". "+sample["abstract"]))

# compute document frequencies
compute_document_frequency(
    documents=train_docs,
    output_file="df-inspec.tsv.gz",
    language='en',  # language of the input files
    normalization='stemming',  # use porter stemmer
    n=5  # compute n-grams up to 5-grams
)

# load df counts
df = load_document_frequency_file(input_file='df-inspec.tsv.gz')

outputs = {}
for model in [FirstPhrases, TopicRank, PositionRank, MultipartiteRank, TextRank]:
    outputs[model.__name__] = []

    extractor = model()
    for i, doc in enumerate(tqdm(test_docs)):
        extractor.load_document(input=doc, language='en')
        extractor.grammar_selection(grammar="NP: {<ADJ>*<NOUN|PROPN>+}")
        extractor.candidate_weighting()
        outputs[model.__name__].append([u for u, v in extractor.get_n_best(n=5, stemming=True)])

for model in [KPMiner, TfIdf]:
    outputs[model.__name__] = []

    extractor = model()
    for i, doc in enumerate(tqdm(test_docs)):
        extractor.load_document(input=doc, language='en')
        extractor.grammar_selection(grammar="NP: {<ADJ>*<NOUN|PROPN>+}")
        extractor.candidate_weighting(df=df)
        outputs[model.__name__].append([u for u, v in extractor.get_n_best(n=5, stemming=True)])


def evaluate(top_N_keyphrases, references):
    P = len(set(top_N_keyphrases) & set(references)) / len(top_N_keyphrases)
    R = len(set(top_N_keyphrases) & set(references)) / len(references)
    F = (2 * P * R) / (P + R) if (P + R) > 0 else 0
    return (P, R, F)


# loop through the models
for model in outputs:
    # compute the P, R, F scores for the model
    scores = []
    for i, output in enumerate(tqdm(outputs[model])):
        references = dataset['test'][i]["uncontr_stems"]
        scores.append(evaluate(output, references))

    # compute the average scores
    avg_scores = np.mean(scores, axis=0)

    # print out the performance of the model
    print("Model: {} P@5: {:.3f} R@5: {:.3f} F@5: {:.3f}".format(model, avg_scores[0], avg_scores[1], avg_scores[2]))
