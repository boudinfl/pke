import os
import logging

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

    _embedding_path = None
    _embedding_model = None

    def __init__(self, embedding_path=None):
        try:
            import sent2vec  # See https://github.com/epfml/sent2vec
        except ImportError:
            logging.warning('Module sent2vec was not found.')
            logging.warning('Please install using `python -m pip install cython;'
                            'python -m pip install git+https://github.com/epfml/sent2vec` '
                            'to use EmbedRank')
            return

        super(EmbedRank, self).__init__()

        if embedding_path is None:
            model_name = 'wiki_bigrams.bin'
            self._embedding_path = os.path.join(self._models, model_name)
        else:
            self._embedding_path = embedding_path

        if not os.path.exists(self._embedding_path):
            logging.error('Could not find {}'.format(self._embedding_path))
            logging.error('Please download "sent2vec_wiki_bigrams" model from '
                            'https://github.com/epfml/sent2vec#downloading-sent2vec-pre-trained-models.')
            logging.error('And place it in {}.'.format(self._models))
            logging.error('Or provide an embedding path.')

        if EmbedRank._embedding_path is None or EmbedRank._embedding_path != self._embedding_path:
            logging.info('Loading sent2vec model')
            EmbedRank._embedding_model = sent2vec.Sent2vecModel()
            EmbedRank._embedding_model.load_model(self._embedding_path)
            self._embedding_model = EmbedRank._embedding_model
            EmbedRank._embedding_path = self._embedding_path
            logging.info('Done loading sent2vec model')

        # Initialize _pos here, if another selection function is used.
        self._pos = {'NOUN', 'PROPN', 'ADJ'}

    def candidate_selection(self, pos=None):
        """Candidate selection using longest sequences of PoS.

        Args:
            pos (set): set of valid POS tags, defaults to ('NOUN', 'PROPN',
                'ADJ').
        """

        if pos is not None:
            self._pos = pos

        # select sequence of adjectives and nouns
        self.longest_pos_sequence_selection(valid_pos=self._pos)

    def mmr_ranking(self, document, candidates, l):
        """Rank candidates according to a query

        Args:
            document (np.array): dense representation of document (query)
            candidates (np.array): dense representation of candidates
            l (float): ratio between distance to query or distance between
                chosen candidates
        Returns:
            list of candidates rank
        """

        def norm(sim, **kwargs):
            sim -= sim.min(**kwargs)
            sim /= sim.max(**kwargs)
            sim = 0.5 + (sim - sim.mean(**kwargs)) / sim.std(**kwargs)
            return sim

        sim_doc = cosine_similarity(document, candidates)
        sim_doc[np.isnan(sim_doc)] = 0.
        sim_doc = norm(sim_doc)
        sim_doc[np.isnan(sim_doc)] = 0.

        sim_can = cosine_similarity(candidates)
        sim_can[np.isnan(sim_can)] = 0.
        sim_can = norm(sim_can, axis=1)
        sim_can[np.isnan(sim_can)] = 0.

        sel = np.zeros(len(candidates), dtype=bool)
        ranks = [None] * len(candidates)
        # Compute first candidate, the second part of the calculation is 0
        # as there are no other chosen candidates to maximise distance to
        chosen_candidate = (sim_doc * l).argmax()
        sel[chosen_candidate] = True
        ranks[chosen_candidate] = 0

        for r in range(1, len(candidates)):

            # Remove already chosen candidates
            sim_can[sel] = np.nan

            # Compute MMR score
            scores = l * sim_doc - (1 - l) * sim_can[:, sel].max(axis=1)
            chosen_candidate = np.nanargmax(scores)

            # Update output and mask with chosen candidate
            sel[chosen_candidate] = True
            ranks[chosen_candidate] = r

        return ranks

    def candidate_weighting(self, l=1, lower=False):
        """Candidate weighting function using distance to document.

        Args:
            l (float): Lambda parameter for EmbedRank++ Maximal Marginal
            Relevance (MMR) computation. Use 1 to compute EmbedRank and 0 to not
            use the document, but only the most diverse set of candidates
            (defaults to 1).
        """
        # Flatten sentences and remove words with unvalid POS
        doc = ' '.join(w.lower() if lower else w for s in self.sentences
                       for i, w in enumerate(s.words)
                       if s.pos[i] in self._pos)

        doc_embed = self._embedding_model.embed_sentence(doc)
        cand_name = list(self.candidates.keys())
        cand = (self.candidates[k] for k in cand_name)
        cand = [' '.join(k.surface_forms[0]) for k in cand]
        cand = [k.lower() if lower else k for k in cand]
        cand_embed = self._embedding_model.embed_sentences(cand)
        rank = self.mmr_ranking(doc_embed, cand_embed, l)
        for candidate_id, r in enumerate(rank):
            if len(rank) > 1:
                # Inverting ranks so the first ranked candidate has the biggest score
                score = (len(rank) - 1 - r) / (len(rank) - 1)
            else:
                score = r
            self.weights[cand_name[candidate_id]] = score
