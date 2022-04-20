"""
Functions for explaining classifiers based on knowledge graph embeddings.
"""

from copy import deepcopy
import numpy as np
from tqdm import tqdm
import itertools
from sklearn.utils import check_random_state
from pyrdf2vec import RDF2VecTransformer
import logging

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
        """Returns the set of all triples within a given list of walks."""
        walkLists = [IndexedWalks.walk_as_triples(walk) for walk in walks]
        return set(itertools.chain.from_iterable(walkLists))


class LimeRdfExplainer(object):
    """Explains classifiers based on knowledge graph embeddings."""

    def __init__(self, transformer: RDF2VecTransformer, entities, random_state=None):
        """Initializer."""
        print("Hello, world!")
        self.oldTransformer = deepcopy(transformer)
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

    def __data_labels_distances(self, entity, classifier_fn, num_samples, max_removed_triples=None):
        """Generates a neighborhood around a prediction.

        Generates neighborhood data by removing random triples from
        the instance, and predicting with the classifier. Triples are
        removed by eliminating all walks that contain them.
        """
        originalWalks = self.indexedWalks.walks(entity)
        originalTriples = list(IndexedWalks.walks_as_triples(originalWalks))
        tripleCount = len(originalTriples)

        # How many triples to remove in a specific perturbed sample
        if not max_removed_triples:
            max_removed_triples = tripleCount
        sample = self.random_state.randint(1, max_removed_triples + 1, num_samples)

        # !!!!! TODO !!!!!
        sample = np.ones(num_samples, dtype=np.int64)

        # Mark random triples as removed, creating a new corpus for artificial entities
        data = np.ones((num_samples, tripleCount))
        newEntities = [f"{entity}*{i}*" for i in range(num_samples)]
        newCorpus = []

        for i, nRemove in enumerate(tqdm(sample)):

            inactive = self.random_state.choice(
                range(tripleCount), nRemove, replace=False) if i != 0 else []
            data[i, inactive] = 0

            # Build new corpus by removing walks that contain inactive triples
            removedTriples = [t for i, t in enumerate(originalTriples) if i in inactive]
            remainingWalks = []

            print("-----\n", i, nRemove)
            print(removedTriples, "removed")

            for walk in originalWalks:
                walkTriples = IndexedWalks.walk_as_triples(walk)

                if any([removed in walkTriples for removed in removedTriples]):
                    print(walk, "removed")
                    continue

                # Rename entity of interest
                modifiedWalk = [f"{e}*{i}*" if e == entity else e for e in walk]

                remainingWalks.append(modifiedWalk)

            newCorpus.append(remainingWalks)

        averageWalks = sum([len(x) for x in newCorpus])/len(newCorpus)
        logging.info(f"Average remaining walks per artificial entity (from 484): {averageWalks}")

        # Get embeddings for new entities

        # fit expects walks as
        """
        [
            [ #entity1
                (s, v, o, v, o2, ...),
                (s, v, o, v, o2, ...)
            ],

            [ #entity2

            ]
        ]
        """
        self.transformer.fit(newCorpus, is_update=True)

        entities = newEntities  # self.entities + newEntities

        # TODO when corpus contains entitites with all walks removed, throws Value Error (entities must have been provided to fit first)
        embeddings = self.transformer.embedder.transform(entities)

        self.transformer._update(self.transformer._embeddings, embeddings)
        return (data, embeddings, newCorpus)
        # Get predictions for new embeddings

        # Determine distances

        # Return


if __name__ == "__main__":
    import pickle
    import os
    import pandas as pd

    moviePath = "/workspaces/rdflime/rdflime-util/data/metacritic-movies"
    with open(os.path.join(moviePath, "rdf2vec_transformer_cbow_50"), "rb") as f:
        t = pickle.load(f)
    movieFull = pd.read_csv(os.path.join(moviePath, "movies_fixed.tsv"), sep="\t")
    movies = [movie.DBpedia_URI for index, movie in movieFull.iterrows()]

    explainer = LimeRdfExplainer(t, movies)
    explainer.explain_instance(movies[123], None)
