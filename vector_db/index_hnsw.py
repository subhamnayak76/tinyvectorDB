import hnswlib
import numpy as np

class HNSWIndex:
    def __init__(self, store, space="cosine", M=16, ef_construction=200, ef=50):
        self.store = store
        self.ef = ef
        self.index = hnswlib.Index(space=space, dim=store.get_vectors().shape[1])
        vecs = store.get_vectors()
        self.index.init_index(max_elements=max(len(store)*2, 100), ef_construction=ef_construction, M=M)
        self.index.add_items(vecs, list(range(len(store))))
        self.index.set_ef(ef)

    def search(self, query_vector, k=5):
        labels, distances = self.index.knn_query(
            query_vector.reshape(1, -1).astype(np.float32),
            k=min(k, len(self.store))
        )
        return [
            {
                "id":       self.store.ids[i],
                "score":    round(1 - d, 4),
                "text":     self.store.texts[i],
                "metadata": self.store.metadata[i],
            }
            for i, d in zip(labels[0], distances[0])
        ]