import os
import logging

# import sent2vec  # See https://github.com/epfml/sent2vec
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from pke import LoadFile


class EmbedRank(LoadFile):
    """EmbedRank keyphrase extraction model.

    Parameterized example::

        import string
        import pke

        # 1. create an EmbedRank extractor.
        extractor = pke.unsupervised.EmbedRank()

        # 2. load the content of the document.
        extractor.load_document(input='path/to/input',
                                language='en',
                                normalization=None)

        # 3. select sequences of nouns and adjectives as candidates.
        extractor.candidate_selection()

        # 4. weight the candidates using EmbedRank method
        extractor.candidate_weighting()

        # 5. get the 10-highest scored candidates as keyphrases
        keyphrases = extractor.get_n_best(n=10)
    """

    def __init__(self, embedding_path=None):
        super(EmbedRank, self).__init__()
        if embedding_path is None:
            model_name = 'torontobooks_unigrams.bin'
            # model_name = 'wiki_bigrams.bin'
            self._embedding_path = os.path.join(self._models, model_name)
        else:
            self._embedding_path = embedding_path
        logging.info('Loading sent2vec model')
        #self._embedding_model = sent2vec.Sent2vecModel()
        #self._embedding_model.load_model(self._embedding_path)
        logging.info('Done loading sent2vec model')
        self._pos = None

    def candidate_selection(self, pos=None):
        """Candidate selection using longest sequences of PoS.

        Args:
            pos (set): set of valid POS tags, defaults to ('NOUN', 'PROPN',
                'ADJ').
        """

        if pos is None:
            pos = {'NOUN', 'PROPN', 'ADJ'}
        self._pos = pos

        # select sequence of adjectives and nouns
        self.longest_pos_sequence_selection(valid_pos=pos)

    def mmr_ranking(self, document, candidates, l):
        """Rank candidates according to a query

        Args:
            document (np.array): dense representation of document (query)
            candidates (np.array): dense representation of candidates
            l (float): ratio between distance to query or distance between
                chosen candidates
        Returns:
            ordered list of candidates indexes
        """

        def norm(sim, **kwargs):
            sim -= sim.min(**kwargs)
            sim /= sim.max(**kwargs)
            sim = 0.5 + (sim - sim.mean(**kwargs)) / sim.std(**kwargs)
            return sim

        sim_doc = cosine_similarity(document, candidates)
        sim_doc[np.isnan(sim_doc)] = 0.
        sim_doc = norm(sim_doc)

        sim_can = cosine_similarity(candidates)
        sim_can[np.isnan(sim_can)] = 0.
        sim_can = norm(sim_can, axis=1)


        sel = np.zeros(len(candidates), dtype=bool)

        # Compute first candidate, the second part of the calculation is 0
        # as there are no other chosen candidates to maximise distance to
        first = (sim_doc * l).argmax()
        selected = [first]
        sel[first] = True

        for _ in range(len(candidates) - 1):

            # Remove already chosen candidates
            sim_can[sel] = np.nan

            # Compute MMR score
            scores = l * sim_doc - (1 - l) * sim_can[:, sel].max(axis=1)
            max_score = np.nanargmax(scores)

            # Update output and mask with chosen candidate
            selected.append(max_score)
            sel[max_score] = True

        return selected

    def candidate_weighting(self, l=1):
        """Candidate weighting function using distance to document.

        Args:
            l (float): Lambda parameter for EmbedRank++ Maximal Marginal
            Relevance (MMR) computation. Use 1 to compute EmbedRank and 0 to not
            use the document, but only the most diverse set of candidates
            (defaults to 1).
        """

        np.random.seed(seed=78)

        def embed_sentence(sent):
            return np.random.randint(0, 300, (1, 300))

        def embed_sentences(sents):
            return np.concatenate(list(map(embed_sentence, sents)), axis=0)

        # Flatten sentences and remove words with unvalid POS
        doc = ' '.join(w for s in self.sentences
                       for i, w in enumerate(s.words)
                       if s.pos[i] in self._pos)
        #doc_embed = self._embedding_model.embed_sentence(doc)
        doc_embed = embed_sentence(doc)

        cand_name = list(self.candidates.keys())
        cand = (self.candidates[k] for k in cand_name)
        cand = [' '.join(k.surface_forms[0]).lower() for k in cand]

        # cand_embed = self._embedding_model.embed_sentences(cand)
        cand_embed = embed_sentences(cand)

        rank = self.mmr_ranking(doc_embed, cand_embed, l)

        for rank, candidate_id in enumerate(rank):
            self.weights[cand_name[candidate_id]] = rank
