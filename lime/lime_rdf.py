"""
Functions for explaining classifiers based on knowledge graph embeddings.
"""

import numpy as np
import scipy as sp
from tqdm import tqdm
import itertools
import sklearn
from sklearn.utils import check_random_state
from pyrdf2vec import RDF2VecTransformer
import logging
import pickle


logging.basicConfig(level=logging.WARN)


class IndexedWalks(object):
    """Index walks per entity."""

    def __init__(self, entities, walks, strict_mode=False):
        """Initializer."""

        self._walkIndex = {}

        if strict_mode:
            # Strict mode: Index walks that have been generated for different entities
            for walk_group in tqdm(walks):
                for walk in walk_group:
                    for entity in entities:
                        if entity in walk:
                            entry = self._walkIndex.setdefault(entity, [])
                            entry.append(walk)
                            self._walkIndex[entity] = entry

        else:
            for entity, entityWalks in zip(entities, walks):
                self._walkIndex[entity] = entityWalks

    def walks(self, entity):
        """Returns all walks for a given entity."""
        return self._walkIndex[entity]

    @staticmethod
    def walk_as_triples(walk):
        """Returns all triples within a given walk."""
        return [(walk[i-2], walk[i-1], walk[i]) for i in range(2, len(walk)-1, 2)]

    @staticmethod
    def walks_as_triples(walks):
        """Returns the set of distinct triples within a given list of walks."""
        walkLists = [IndexedWalks.walk_as_triples(walk) for walk in walks]
        return set(itertools.chain.from_iterable(walkLists))


class LimeRdfExplainer(object):
    """Explains classifiers based on knowledge graph embeddings."""

    def __init__(self, transformer: RDF2VecTransformer, entities, random_state=None):
        """Initializer."""
        print("Hello, world!")

        # Can safely do this since transformer object is serializable by design
        self.oldTransformer = pickle.loads(pickle.dumps(transformer))

        self.transformer = transformer

        self.embeddings = transformer._embeddings
        self.entities = entities
        self.indexedWalks = IndexedWalks(entities, transformer._walks, strict_mode=False)
        self.random_state = check_random_state(random_state)

    def explain_instance(self, entity, classifier_fn, num_samples, max_removed_triples=None):
        """Generates explanations for a prediction."""

        # Generate and evaluate random neighborhood
        return self.__data_labels_distances(entity, classifier_fn, num_samples, max_removed_triples)

        # TODO retrieve explanations from lime base

    def __data_labels_distances(self, entity, classifier_fn, num_samples, max_removed_triples=None, distance_metric="cosine"):
        """Generates a neighborhood around a prediction.

        Generates neighborhood data by removing random triples from
        the instance, and predicting with the classifier. Triples are
        removed by eliminating all walks that contain them.
        """

        def distance_fn(x):
            return sklearn.metrics.pairwise.pairwise_distances(x, x[0], metric=distance_metric).ravel() * 100

        originalWalks = self.indexedWalks.walks(entity)
        originalTriples = list(IndexedWalks.walks_as_triples(originalWalks))
        tripleCount = len(originalTriples)

        num_samples = int(num_samples)
        max_removed_triples = int(max_removed_triples)

        # How many triples to remove in a specific perturbed sample
        if max_removed_triples is None:
            max_removed_triples = tripleCount

        # TODO Allow random number of samples to be removed -> see Value Error below
        sample = np.ones(num_samples, dtype=np.int64) * max_removed_triples
        # sample = self.random_state.randint(1, max_removed_triples + 1, num_samples)

        # Mark random triples as removed, creating a new corpus for artificial entities
        data = np.ones((num_samples, tripleCount))
        newEntities = [f"{entity}_{i}" for i in range(num_samples)]
        newCorpus = []

        for i, nRemove in enumerate(tqdm(sample)):

            inactive = self.random_state.choice(
                range(tripleCount), nRemove, replace=False) if i != 0 else []
            data[i, inactive] = 0

            # Build new corpus by removing walks that contain inactive triples
            removedTriples = [t for i, t in enumerate(originalTriples) if i in inactive]
            remainingWalks = []

            for walk in originalWalks:
                walkTriples = IndexedWalks.walk_as_triples(walk)

                if any([removed in walkTriples for removed in removedTriples]):
                    continue

                # Rename entity of interest
                modifiedWalk = [newEntities[i] if e == entity else e for e in walk]

                remainingWalks.append(modifiedWalk)

            newCorpus.append(remainingWalks)

        averageWalks = sum([len(x) for x in newCorpus])/len(newCorpus)
        logging.info(f"Average remaining walks per artificial entity (from 484): {averageWalks}")

        # Get embeddings for new entities

        # TODO
        """
        corpus = [walk for entity_walks in walks for walk in entity_walks]
        self._model.build_vocab(corpus, update=is_update)
        self._model.train(
            corpus,
            total_examples=self._model.corpus_count,
            epochs=self._model.epochs,
        )
        """
        model = self.transformer.embedder._model
        wv = model.wv

        #locked = np.zeros(len(wv))

        addition = [walk for entity_walks in newCorpus for walk in entity_walks]
        model.build_vocab(addition, update=True)

        """
        open = np.ones(len(wv)-len(locked))
        model.wv.vectors_lockf = np.concatenate([locked, open])
        """

        model.train(addition, total_examples=model.corpus_count, epochs=model.epochs)

        # Changing epochs did not help
        # self.transformer.fit(newCorpus, is_update=True)

        entities = newEntities  # self.entities + newEntities

        # TODO when corpus contains entitites with all walks removed, throws Value Error (entities must have been provided to fit first)
        embeddings = self.transformer.embedder.transform(entities)

        # self.transformer._update(self.transformer._embeddings, embeddings)

        # Get predictions for new embeddings
        labels = classifier_fn(embeddings)

        # Determine distances
        distances = distance_fn(sp.sparse)

        return data, labels, distances


if __name__ == "__main__":
    import os
    import pandas as pd

    moviePath = "/workspaces/rdflime/rdflime-util/data/metacritic-movies"
    with open(os.path.join(moviePath, "rdf2vec_transformer_cbow_50"), "rb") as f:
        t = pickle.load(f)
    movieFull = pd.read_csv(os.path.join(moviePath, "movies_fixed.tsv"), sep="\t")
    movies = [movie.DBpedia_URI for index, movie in movieFull.iterrows()]

    explainer = LimeRdfExplainer(t, movies)
    explainer.explain_instance(movies[123], None)
