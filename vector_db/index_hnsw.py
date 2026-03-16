from hnsw import Node, HNSW, get_random_layer, cosine_sim, get_nearest

class HNSWIndex:
    def __init__(self, store):
        self.store = store
        self.hnsw = HNSW()
        for vector in store.vectors:
            self.hnsw.insert(vector.tolist())

    def search(self, query_vector, k=5):
        results = self.hnsw.search(query_vector.tolist(), k)
        return [
            {
                "id":       self.store.ids[i],
                "score": round(cosine_sim(query_vector.tolist(), self.hnsw.nodes[i].vector), 4),
                "text":     self.store.texts[i],
                "metadata": self.store.metadata[i]
            }
            for i in results
        ]