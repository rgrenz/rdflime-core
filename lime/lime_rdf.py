"""
Functions for explaining classifiers based on knowledge graph embeddings.
"""

from functools import partial
from lime import lime_base
import numpy as np
import scipy as sp
from tqdm import tqdm
import itertools
import sklearn
from sklearn.utils import check_random_state
from pyrdf2vec import RDF2VecTransformer
import logging
import pickle
import random

from . import explanation

logging.basicConfig(level=logging.WARN)


class GraphDomainMapper(explanation.DomainMapper):
    """Maps feature ids to triples."""

    def __init__(self, triples, short_uris):
        self.triples = triples
        self.short_uris = short_uris

    def map_exp_ids(self, exp, **kwargs):
        mappings = []

        for x in exp:
            triple = self.triples[x[0]]
            triple = tuple([i.split("/")[-1] if self.short_uris else i for i in list(triple)])
            mappings.append((triple, x[1]))

        return mappings


class IndexedWalks(object):
    """Index walks per entity."""

    def __init__(self, entities, walks, strict_mode=False):
        """Initializer."""

        # Key: Entity, Value: Walks
        self._walkIndex = {}

        # Key: Triple, Value: {walks: Walks, isPrefix: boolean}
        self._adjacent_triples = {}

        """
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
        """
        for entity, entityWalks in zip(entities, walks):
            self._walkIndex[entity] = entityWalks

            for walk in entityWalks:
                entity_index = walk.index(entity)

                if entity_index != 0:
                    # There is a prefix triple
                    # (JonDoe, writer, _)
                    t = list(walk[entity_index-2:entity_index+1])
                    t[2] = "_"
                    t = tuple(t)

                    if t not in self._adjacent_triples:
                        self._adjacent_triples[t] = {"walk_stubs": [], "isPrefix": True}
                    stub = walk[:entity_index]
                    if stub not in self._adjacent_triples[t]["walk_stubs"]:
                        self._adjacent_triples[t]["walk_stubs"].append(stub)

                if entity_index != len(walk) - 1:
                    # There is a postifx triple
                    t = list(walk[entity_index:entity_index+3])
                    t[0] = "_"
                    t = tuple(t)

                    if t not in self._adjacent_triples:
                        self._adjacent_triples[t] = {"walk_stubs": [], "isPrefix": False}

                    stub = walk[entity_index+1:]
                    if stub not in self._adjacent_triples[t]["walk_stubs"]:
                        self._adjacent_triples[t]["walk_stubs"].append(stub)

        self._walks = walks

    def walks(self, entity, triple=None):
        """Returns all walks for a given entity (that contain a given triple)."""
        walks = self._walkIndex[entity]
        if triple:
            return [w for w in walks if triple in IndexedWalks.walk_as_triples(w)]
        return walks

    def node_degree(self, entity, out_degree=True):
        # How many other distinct nodes are reached from this node?
        entity_index = 0 if out_degree else 2
        triples = IndexedWalks.walks_as_triples(self.walks(entity))
        relevant_set = {t for t in triples if t[entity_index] == entity}
        return len(relevant_set)

    @ staticmethod
    def walk_as_triples(walk):
        """Returns all triples within a given walk."""
        return [(walk[i-2], walk[i-1], walk[i]) for i in range(2, len(walk), 2)]

    @ staticmethod
    def walks_as_triples(walks):
        """Returns the set of distinct triples within a given list of walks."""
        walkLists = [IndexedWalks.walk_as_triples(walk) for walk in walks]
        return set(itertools.chain.from_iterable(walkLists))

    @ staticmethod
    def walk_contains_triple(walk, triple):
        return triple in IndexedWalks.walk_as_triples(walk)

    @ staticmethod
    def filter_walks(input_walks, removed_triples):
        """Returns only the walks from input_walks that contain none of the removed_triples."""
        return [w for w in input_walks if any([IndexedWalks.walk_contains_triple(w, t) for t in removed_triples])]


class LimeRdfExplainer(object):
    """Explains classifiers based on knowledge graph embeddings."""

    def __init__(self,
                 transformer: RDF2VecTransformer,
                 entities,
                 class_names=None,
                 kernel=None,
                 kernel_width=25,
                 verbose=False,
                 feature_selection="auto",
                 random_state=None):
        """Initializer."""

        if kernel is None:
            def kernel(d, kernel_width):
                return np.sqrt(np.exp(-(d ** 2) / kernel_width ** 2))
        kernel_fn = partial(kernel, kernel_width=kernel_width)

        self.transformer = transformer
        self.entities = entities
        self.indexed_walks = IndexedWalks(entities, transformer._walks, strict_mode=False)
        self.random_state = check_random_state(random_state)

        self.class_names = class_names
        self.feature_selection = feature_selection

        self.base = lime_base.LimeBase(kernel_fn, verbose, self.random_state)

    def explain_instance(self,
                         entity,
                         classifier_fn,
                         labels=(1,),
                         num_features=10,
                         num_samples=5000,
                         allow_triple_addition=True,
                         allow_triple_substraction=True,
                         max_changed_triples=None,
                         change_count_fixed=True,
                         use_w2v_freeze=True,
                         center_correction=True,
                         single_run=True,
                         train_with_all=False,
                         distance_metric="cosine",
                         model_regressor=None,
                         short_uris=False):
        """Generates explanations for a prediction."""

        # Compute relevant set of triples for addition / substraction
        assert allow_triple_addition or allow_triple_substraction, "Either triple addition or substraction must be allowed!"
        existing_triples = IndexedWalks.walks_as_triples(self.indexed_walks.walks(entity))
        relevant_triples = existing_triples if allow_triple_substraction else set()

        if allow_triple_addition:
            for triple in self.indexed_walks._adjacent_triples:
                reconstructed_triple = tuple([entity if x == "_" else x for x in triple])
                if reconstructed_triple not in existing_triples:
                    # Only allow addition of triples that do not yet exist on the entity
                    relevant_triples.add(triple)

        # Generate and evaluate random neighborhood
        data, yss, distances = self.__data_labels_distances(entity,
                                                            relevant_triples,
                                                            classifier_fn,
                                                            num_samples,
                                                            max_changed_triples,
                                                            change_count_fixed,
                                                            use_w2v_freeze,
                                                            center_correction,
                                                            single_run,
                                                            train_with_all,
                                                            distance_metric)
        print("Got data, labels, and distances")

        # Initialize explanation
        domain_mapper = GraphDomainMapper(list(relevant_triples), short_uris)
        ret_exp = explanation.Explanation(
            domain_mapper,
            mode="classification",
            class_names=self.class_names,
            random_state=self.random_state)
        ret_exp.predict_proba = yss[0]

        """if top_labels..."""

        for label in labels:
            (ret_exp.intercept[label],
             ret_exp.local_exp[label],
             ret_exp.score[label],
             ret_exp.local_pred[label]) = self.base.explain_instance_with_data(
                 data, yss, distances, label, num_features,
                 feature_selection=self.feature_selection,
                 model_regressor=model_regressor
            )

        return data, yss, distances, ret_exp

    def __data_labels_distances(self, entity, relevant_triples, classifier_fn, num_samples,
                                max_changed_triples=None,
                                change_count_fixed=True,
                                use_w2v_freeze=True,
                                center_correction=True,
                                single_run=True,
                                train_with_all=False,
                                distance_metric="cosine"):
        """Generates a neighborhood around a prediction.
        Generates neighborhood data by changing random triples on
        the instance, and predicting with the classifier. Triples are
        removed by eliminating all walks that contain them.
        """

        def distance_fn(x):
            return sklearn.metrics.pairwise.pairwise_distances(
                x, x[0], metric=distance_metric).ravel() * 100

        original_walks = self.indexed_walks.walks(entity)
        relevant_triples = list(relevant_triples)
        triple_count = len(relevant_triples)

        # Triples that actually exist on the entity (that can be subtracted)
        subtraction_triple_count = len([x for x in relevant_triples if "_" not in x])

        # Triples that do not exist on the entity (that can be added)
        addition_triple_count = triple_count - subtraction_triple_count

        num_samples = int(num_samples)
        max_changed_triples = int(max_changed_triples)

        # How many triples to change in a specific perturbed sample
        if max_changed_triples is None:
            max_changed_triples = triple_count

        if change_count_fixed:
            # Always change fixed number of triples given in parameter change_count_fixed
            sample = np.ones(num_samples, dtype=np.int64) * max_changed_triples
        else:
            # Draw the number of removed triples for each perturbed entity at random
            sample = self.random_state.randint(1, max_changed_triples + 1, num_samples)

        # Mark random triples as changed, creating a new corpus for artificial entities
        data = np.ones((num_samples, triple_count))
        data[:, subtraction_triple_count:] = np.zeros((num_samples, addition_triple_count))

        new_entities = [f"{entity}_" for i in range(num_samples)]
        new_corpus = []

        print("Preparing walks:")
        for i, change_count in enumerate(tqdm(sample)):

            # Choose inactive triple for this perturbed sample
            # Special case: first sample is a reference without changes
            changed = self.random_state.choice(
                range(triple_count), change_count, replace=False) if i != 0 else []

            data[i, changed] = 1 - data[i, changed]

            # Build new corpus by removing walks that contain inactive triples
            # changed_triples = [t for i, t in enumerate(relevant_triples) if i in changed]
            changed_triples = [relevant_triples[idx] for idx in changed]
            modified_walks = pickle.loads(pickle.dumps(original_walks))

            # 1st phase - remove walks with eliminated triples
            for t in [t for t in changed_triples if "_" not in t]:
                modified_walks = [
                    w for w in modified_walks if t not in IndexedWalks.walk_as_triples(w)]

            # 2nd phase - mutate walks with added triples
            for t in [t for t in changed_triples if "_" in t]:

                # Choose a random walk that we would like to apply
                triple_info = self.indexed_walks._adjacent_triples[t]
                walk_stub = random.choice(triple_info["walk_stubs"])
                degree = self.indexed_walks.node_degree(
                    entity, out_degree=not triple_info["isPrefix"])

                # For every walk with entity of interest, draw random number
                p_mutate = 1 / (degree + 1)
                ctr = 0
                for idx, w in enumerate(modified_walks):
                    if random.random() <= p_mutate:
                        ctr += 1
                        entity_index = w.index(entity)
                        if triple_info["isPrefix"]:
                            w = (*walk_stub, *w[entity_index:])
                        else:
                            w = (*w[:entity_index+1], *walk_stub)
                    modified_walks[idx] = w

            # Rename new entity
            modified_walks = [
                tuple([w if w != entity else new_entities[i] for w in walk]) for walk in modified_walks
            ]
            new_corpus.append(modified_walks)

        average_walks = sum([len(x) for x in new_corpus])/len(new_corpus)
        print(f"Average remaining walks per artificial entity (from 484): {average_walks}")
        # print(new_corpus[0])

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

        freeze_vector_locked = np.zeros(len(wv), dtype=np.float32)

        new_embeddings = []
        runs = []

        if single_run:  # TODO !
            runs.append([tuple(walk) for entity_walks in new_corpus for walk in entity_walks])
        else:
            for entity_walk in new_corpus:
                runs.append([tuple(walk) for walk in entity_walk])

        """
        w2v.build_vocab([tuple(walk)
                        for entity_walks in new_corpus for walk in entity_walks], update=True)
        """
        w2v.build_vocab([tuple(walk)
                         for entity_walks in new_corpus for walk in entity_walks], update=True)
        w2v_p = pickle.dumps(self.transformer.embedder._model)

        print("Training W2V with new corpus:")
        for i, run in enumerate(tqdm(runs)):
            w2v = pickle.loads(w2v_p)
            wv = w2v.wv

            if(train_with_all):
                run = [tuple(walk)
                       for entity_walks in self.transformer._walks for walk in entity_walks] + run

            run_entities = new_entities if single_run else [new_entities[i]]
            # w2v.build_vocab(run, update=True)

            for run_entity in run_entities:
                wv[run_entity] = np.copy(original_embedding)

            freeze_vector_open = np.ones(len(wv)-len(freeze_vector_locked), dtype=np.float32)
            if use_w2v_freeze:
                # TODO maybe freeze depends on the length of input to train (which is not always the length of the whole corpus)
                w2v.wv.vectors_lockf = np.concatenate([freeze_vector_locked, freeze_vector_open])
                # print(len(freeze_vector_locked), len(freeze_vector_open))

            # TODO Throws Value Error when corpus contains entitites with all walks removed
            # ("entities must have been provided to fit first") -> ensure we never remove all walks
            #
            # Also: displays warning "effective 'Alpha' higher than previous cycles"

            w2v.train(run, total_examples=len(run),  # w2v.corpus_count,
                      epochs=w2v.epochs, start_alpha=w2v.min_alpha)

            # Add embedding of fake entity to our collection
            # Need to clone vector to avoid memory leak in multi run scenario
            new_embeddings += [np.copy(wv.get_vector(entity)) for entity in run_entities]

        if center_correction:
            diff_to_center = new_embeddings[0] - original_embedding
            new_embeddings = [e - diff_to_center for e in new_embeddings]

        # Get prediction probabilities for new embeddings
        labels = classifier_fn(new_embeddings)

        # Determine distances
        distances = distance_fn(sp.sparse.csr_matrix(new_embeddings))

        # Debug
        # self.new_corpus = new_corpus
        # self.new_embeddings = new_embeddings

        return data, labels, distances


if __name__ == "__main__":
    import os
    import pandas as pd

    moviePath = "/workspaces/rdflime/rdflime-util/data/metacritic-movies"
    movieFull = pd.read_csv(os.path.join(moviePath, "movies_fixed.tsv"), sep="\t")
    movies = [movie.DBpedia_URI for index, movie in movieFull.iterrows()]

    with open(os.path.join(moviePath, "transformers", "rdf2vec_transformer_cbow_200"), "rb") as file:
        rdf2vec_transformer = pickle.load(file)

    with open(os.path.join(moviePath,  "classifiers", "svc_100_cbow_200"), "rb") as file:
        clf = pickle.load(file)

    explainer = LimeRdfExplainer(
        transformer=rdf2vec_transformer,
        entities=movies,
        class_names=clf.classes_,
        kernel=None,
        kernel_width=25,
        verbose=False,
        feature_selection="auto",
        random_state=42
    )

    explained_entity_id = 100  # 0-400 -> test data
    explained_entity_uri = movies[explained_entity_id]
    prediction = clf.predict_proba([rdf2vec_transformer._embeddings[explained_entity_id]])

    print("Explaining", explained_entity_uri)
    print("Original prediction:", prediction, " / ".join(clf.classes_))
    print("True class:", movieFull.iloc[explained_entity_id].label)

    """
    Grid search
    ids = [1662, 1735, 1796, 1856, 1935]
    max_changed_triples = [1, 10, 25, 100]
    change_count_fixed = [True, False]
    use_w2v_freeze = [True, False]
    center_correction = [True, False]
    center_init = [True, False]
    single_run = [True, False]
    """

    data, labels, distances, explanation = explainer.explain_instance(
        entity=explained_entity_uri,
        classifier_fn=clf.predict_proba,
        num_features=25,
        num_samples=2,
        allow_triple_addition=True,
        allow_triple_substraction=False,
        max_changed_triples=5,
        change_count_fixed=True,
        use_w2v_freeze=True,
        center_correction=False,
        single_run=False,
        train_with_all=False,
        distance_metric="cosine",
        model_regressor=None,
        short_uris=False
    )
