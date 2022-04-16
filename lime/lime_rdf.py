"""
Functions for explaining classifiers based on knowledge graph embeddings.
"""

from tqdm import tqdm


class IndexedWalks(object):
    """Index walks per entity."""

    def __init__(self, entities, walks):
        """Initializer."""

        self._index = {}

        #print("Building entity<->walks index.")
        for walk_group in tqdm(walks):
            for walk in walk_group:
                for entity in entities:
                    if entity in walk:
                        entry = self._index.setdefault(entity, [])
                        entry.append(walk)
                        self._index[entity] = entry

    def walks(self, entity):
        """Returns all walks for a given entity."""
        return self._index[entity]


class LimeRdfExplainer(object):
    """Explains classifiers based on knowledge graph embeddings."""

    def __init__(self, entities, embeddings, walks):
        """Initializer."""
        pass

    def explain_instance(self, embedding_instance, classifier_fn):
        """Generated explanations for a prediction."""
        pass

    def __data_labels_distances(self):
        """Generates a neighborhood around a prediction."""
        pass
