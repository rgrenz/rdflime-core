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
        self.old_transformer = pickle.loads(pickle.dumps(transformer))

        self.transformer = transformer

        self.embeddings = transformer._embeddings
        self.entities = entities
        self.indexed_walks = IndexedWalks(entities, transformer._walks, strict_mode=False)
        self.random_state = check_random_state(random_state)

    def explain_instance(self, entity, classifier_fn, num_samples,
                         max_removed_triples=None,
                         removal_count_fixed=True,
                         use_w2v_freeze=True,
                         center_correction=True,
                         distance_metric="cosine"):
        """Generates explanations for a prediction."""

        # Generate and evaluate random neighborhood
        return self.__data_labels_distances(entity,
                                            classifier_fn,
                                            num_samples,
                                            max_removed_triples,
                                            removal_count_fixed,
                                            use_w2v_freeze,
                                            center_correction,
                                            distance_metric)

        # TODO retrieve explanations from lime base

    def __data_labels_distances(self, entity, classifier_fn, num_samples,
                                max_removed_triples=None,
                                removal_count_fixed=True,
                                use_w2v_freeze=True,
                                center_correction=True,
                                distance_metric="cosine"):
        """Generates a neighborhood around a prediction.

        Generates neighborhood data by removing random triples from
        the instance, and predicting with the classifier. Triples are
        removed by eliminating all walks that contain them.
        """

        def distance_fn(x):
            return sklearn.metrics.pairwise.pairwise_distances(
                x, x[0], metric=distance_metric).ravel() * 100

        original_walks = self.indexed_walks.walks(entity)
        original_triples = list(IndexedWalks.walks_as_triples(original_walks))
        triple_count = len(original_triples)

        num_samples = int(num_samples)
        max_removed_triples = int(max_removed_triples)

        # How many triples to remove in a specific perturbed sample
        if max_removed_triples is None:
            max_removed_triples = triple_count

        if removal_count_fixed:
            # Always remove fixed number of triples given in parameter removal_count_fixed
            sample = np.ones(num_samples, dtype=np.int64) * max_removed_triples
        else:
            # Draw the number of removed triples for each perturbed entity at random
            sample = self.random_state.randint(1, max_removed_triples + 1, num_samples)

        # Mark random triples as removed, creating a new corpus for artificial entities
        data = np.ones((num_samples, triple_count))
        new_entities = [f"{entity}_{i}" for i in range(num_samples)]
        new_corpus = []

        for i, removal_count in enumerate(tqdm(sample)):

            # Choose inactive triple for this perturbed sample
            # Special case: first sample is a reference without changes
            inactive = self.random_state.choice(
                range(triple_count), removal_count, replace=False) if i != 0 else []
            data[i, inactive] = 0

            # Build new corpus by removing walks that contain inactive triples
            removed_triples = [t for i, t in enumerate(original_triples) if i in inactive]
            remaining_walks = []

            for walk in original_walks:
                walkTriples = IndexedWalks.walk_as_triples(walk)

                if any([removed in walkTriples for removed in removed_triples]):
                    continue

                # Rename entity of interest
                modifiedWalk = [new_entities[i] if e == entity else e for e in walk]

                remaining_walks.append(modifiedWalk)

            new_corpus.append(remaining_walks)

        average_walks = sum([len(x) for x in new_corpus])/len(new_corpus)
        logging.warn(f"Average remaining walks per artificial entity (from 484): {average_walks}")

        # Get embeddings for new entities

        """
        For now we ignore some of the RDF2Vec wrapper functions and talk to W2V directly
        (to use advanced features such as the freeze functionality)
        self.transformer.fit(newCorpus, is_update=True)
        self.transformer._update(self.transformer._embeddings, embeddings)
        new_embeddings = self.transformer.embedder.transform(new_entities)
        """

        w2v = self.transformer.embedder._model
        wv = w2v.wv
        original_embedding = wv.get_vector(entity)

        freeze_vector_locked = np.zeros(len(wv))

        addition = [walk for entity_walks in new_corpus for walk in entity_walks]
        w2v.build_vocab(addition, update=True)

        freeze_vector_open = np.ones(len(wv)-len(freeze_vector_locked))
        if use_w2v_freeze:
            w2v.wv.vectors_lockf = np.concatenate([freeze_vector_locked, freeze_vector_open])
            print(len(freeze_vector_locked), len(freeze_vector_open))

        # TODO Throws Value Error when corpus contains entitites with all walks removed
        # ("entities must have been provided to fit first") -> ensure we never remove all walks
        #
        # Also: displays warning "effective 'Alpha' higher than previous cycles"
        w2v.train(addition, total_examples=w2v.corpus_count, epochs=w2v.epochs)

        new_embeddings = [wv.get_vector(entity) for entity in new_entities]

        if center_correction:
            diff_to_center = new_embeddings[0] - original_embedding
            new_embeddings = [e - diff_to_center for e in new_embeddings]

        # Get prediction probabilities for new embeddings
        labels = classifier_fn(new_embeddings)

        # Determine distances
        distances = distance_fn(sp.sparse.csr_matrix(new_embeddings))

        # Debug
        self.new_corpus = new_corpus

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
