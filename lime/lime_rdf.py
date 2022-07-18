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
                         allow_triple_subtraction=True,
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
        self.use_w2v_freeze = use_w2v_freeze

        # Compute relevant set of triples for addition / subtraction
        assert allow_triple_addition or allow_triple_subtraction, "Either triple addition or subtraction must be allowed!"
        existing_triples = list(IndexedWalks.walks_as_triples(self.indexed_walks.walks(entity)))
        relevant_triples = existing_triples if allow_triple_subtraction else []

        if allow_triple_addition:
            for triple in self.indexed_walks._adjacent_triples:
                reconstructed_triple = tuple([entity if x == "_" else x for x in triple])
                if reconstructed_triple not in existing_triples:
                    # Only allow addition of triples that do not yet exist on the entity
                    relevant_triples.append(triple)

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
        if self.class_names is None:
            self.class_names = [str(x) for x in range(yss[0].shape[0])]
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

    def get_perturbed_walks(self, entity, added_triples, removed_triples, entity_suffix="_"):
        """Builds new random walks for an entity"""
        original_walks = self.indexed_walks.walks(entity)
        modified_walks = pickle.loads(pickle.dumps(original_walks))

        # 1st phase - remove walks with eliminated triples
        for t in removed_triples:
            modified_walks = [
                w for w in modified_walks if t not in IndexedWalks.walk_as_triples(w)]

            """
            n_removed = len(original_walks) - len(modified_walks)
            replacement_indices = self.random_state.choice(range(len(modified_walks)), n_removed)
            modified_walks.extend([pickle.loads(pickle.dumps(modified_walks[idx]))
                                  for idx in replacement_indices])
            """

        # 2nd phase - mutate walks with added triples
        for t in added_triples:
            # Choose a random walk that we would like to apply
            triple_info = self.indexed_walks._adjacent_triples[t]
            walk_stub_idx = self.random_state.randint(len(triple_info["walk_stubs"]))
            walk_stub = triple_info["walk_stubs"][walk_stub_idx]

            # Calculate probability of walk modification
            degree = self.indexed_walks.node_degree(
                entity, out_degree=not triple_info["isPrefix"])
            p_mutate = 1 / (degree + 1)

            # Modify walks
            for idx, w in enumerate(modified_walks):
                if self.random_state.random() <= p_mutate:
                    entity_index = w.index(entity)
                    if triple_info["isPrefix"]:
                        w = (*walk_stub, *w[entity_index:])
                    else:
                        w = (*w[:entity_index+1], *walk_stub)
                modified_walks[idx] = w

        # Rename new entity
        return [
            tuple([w if w != entity else entity + entity_suffix for w in walk]) for walk in modified_walks
        ]

    def get_perturbed_embedding(self, entity, perturbed_entity_walks, entity_suffix="_"):
        """
            For now we ignore some of the RDF2Vec wrapper functions and talk to W2V directly
            (to use advanced features such as the freeze functionality)
            self.transformer.fit(newCorpus, is_update=True)
            self.transformer._update(self.transformer._embeddings, embeddings)
            new_embeddings = self.transformer.embedder.transform(new_entities)
        """

        # Ensure that the vocabulary has already been extended.
        w2v = self.transformer.embedder._model
        wv = w2v.wv
        original_embedding = wv[entity]

        if entity + entity_suffix not in wv:
            w2v.build_vocab(perturbed_entity_walks, update=True)
            wv[entity + entity_suffix] = np.copy(original_embedding)

            if self.use_w2v_freeze:
                freeze_vector = np.zeros(len(wv), dtype=np.float32)
                freeze_vector[-1] = 1
                w2v.wv.vectors_lockf = freeze_vector

            self.w2v_dump = pickle.dumps(w2v)
            self.wv_dump = pickle.dumps(wv)

        # Clone w2v instance
        w2v = pickle.loads(self.w2v_dump)
        # w2v.wv = pickle.loads(self.wv_dump)
        wv = w2v.wv

        # Continue training
        # w2v.min_alpha_yet_reached = 1
        w2v.train(perturbed_entity_walks, total_examples=len(perturbed_entity_walks),
                  epochs=w2v.epochs, start_alpha=w2v.min_alpha)

        # Need to clone vector to avoid memory leak in multi run scenario
        return np.copy(wv.get_vector(entity + entity_suffix))

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

        relevant_triples = list(relevant_triples)
        num_samples = int(num_samples)
        max_changed_triples = int(max_changed_triples)

        triple_count = len(relevant_triples)
        subtraction_triple_count = len([x for x in relevant_triples if "_" not in x])
        addition_triple_count = triple_count - subtraction_triple_count

        # Prepare regression input data
        data = np.ones((num_samples, triple_count))
        data[:, subtraction_triple_count:] = np.zeros((num_samples, addition_triple_count))

        # How many triples to change in a specific perturbed sample
        if max_changed_triples is None:
            max_changed_triples = triple_count

        if change_count_fixed:
            sample = np.ones(num_samples, dtype=np.int64) * max_changed_triples
        else:
            sample = self.random_state.randint(1, max_changed_triples + 1, num_samples)

        # Mark random triples as changed, creating a new corpus for artificial entities
        new_corpus = []

        progress_bar = tqdm(sample)
        progress_bar.set_description("Preparing perturbed walks")
        for i, change_count in enumerate(progress_bar):
            changed = self.random_state.choice(
                range(triple_count), change_count, replace=False) if i != 0 else []

            data[i, changed] = 1 - data[i, changed]

            removed_triples = [relevant_triples[idx]
                               for idx in changed if idx < subtraction_triple_count]
            added_triples = [relevant_triples[idx]
                             for idx in changed if idx >= subtraction_triple_count]

            modified_walks = self.get_perturbed_walks(entity, added_triples, removed_triples)
            assert len(
                modified_walks) > 0, "Entity has lost all walks, consider lowering max_changed_triples."
            new_corpus.append(modified_walks)

        average_walks = sum([len(x) for x in new_corpus])/len(new_corpus)
        print(f"Average remaining walks per artificial entity (from 484): {average_walks}")

        new_embeddings = []
        progress_bar = tqdm(new_corpus)
        progress_bar.set_description("Training perturbed W2V embeddings")
        for i, entity_walks in enumerate(progress_bar):
            new_embeddings.append(self.get_perturbed_embedding(entity, entity_walks))

            """
            if center_correction:
                diff_to_center = new_embeddings[0] - original_embedding
                new_embeddings = [e - diff_to_center for e in new_embeddings]
            """

        # Get prediction probabilities for new embeddings
        labels = classifier_fn(new_embeddings)

        # Determine distances
        distances = distance_fn(sp.sparse.csr_matrix(new_embeddings))

        # Debug
        self.new_corpus = new_corpus
        self.new_embeddings = new_embeddings

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

    data, probabilities, distances, explanation = explainer.explain_instance(
        entity=explained_entity_uri,
        classifier_fn=clf.predict_proba,
        num_features=50,
        num_samples=5000,
        allow_triple_addition=False,
        allow_triple_subtraction=True,
        max_changed_triples=20,
        change_count_fixed=True,
        use_w2v_freeze=True,
        center_correction=False,
        single_run=False,
        train_with_all=False,
        distance_metric="cosine",
        model_regressor=None,
        short_uris=False
    )
