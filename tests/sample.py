# -*- coding: utf-8 -*

import spacy

nlp = spacy.load("en_core_web_sm")

sample = """Inverse problems for a mathematical model of ion exchange in a compressible ion exchanger.
A mathematical model of ion exchange is considered, allowing for ion exchanger compression in the process
of ion exchange. Two inverse problems are investigated for this model, unique solvability is proved, and
numerical solution methods are proposed. The efficiency of the proposed methods is demonstrated by a
numerical experiment.""".replace("\n", " ")

sample_doc = nlp(sample)

sample_list = [[('Inverse', 'NOUN'), ('problems', 'NOUN'), ('for', 'ADP'), ('a', 'DET'), ('mathematical', 'ADJ'),
                ('model', 'NOUN'), ('of', 'ADP'), ('ion', 'NOUN'), ('exchange', 'NOUN'), ('in', 'ADP'), ('a', 'DET'),
                ('compressible', 'ADJ'), ('ion', 'NOUN'), ('exchanger', 'NOUN'), ('.', 'PUNCT')],
               [('A', 'DET'), ('mathematical', 'ADJ'), ('model', 'NOUN'), ('of', 'ADP'), ('ion', 'NOUN'),
                ('exchange', 'NOUN'), ('is', 'AUX'), ('considered', 'VERB'), (',', 'PUNCT'), ('allowing', 'VERB'),
                ('for', 'ADP'), ('ion', 'NOUN'), ('exchanger', 'NOUN'), ('compression', 'NOUN'), ('in', 'ADP'),
                ('the', 'DET'), ('process', 'NOUN'), ('of', 'ADP'), ('ion', 'NOUN'), ('exchange', 'NOUN'),
                ('.', 'PUNCT')],
               [('Two', 'NUM'), ('inverse', 'NOUN'), ('problems', 'NOUN'), ('are', 'AUX'), ('investigated', 'VERB'),
                ('for', 'ADP'), ('this', 'DET'), ('model', 'NOUN'), (',', 'PUNCT'), ('unique', 'ADJ'),
                ('solvability', 'NOUN'), ('is', 'AUX'), ('proved', 'VERB'), (',', 'PUNCT'), ('and', 'CCONJ'),
                ('numerical', 'ADJ'), ('solution', 'NOUN'), ('methods', 'NOUN'), ('are', 'AUX'), ('proposed', 'VERB'),
                ('.', 'PUNCT')],
               [('The', 'DET'), ('efficiency', 'NOUN'), ('of', 'ADP'), ('the', 'DET'), ('proposed', 'VERB'),
                ('methods', 'NOUN'), ('is', 'AUX'), ('demonstrated', 'VERB'), ('by', 'ADP'), ('a', 'DET'),
                ('numerical', 'ADJ'), ('experiment', 'NOUN'), ('.', 'PUNCT')]]
