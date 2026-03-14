import numpy as np

def cosine_sim(query,vectors):
    query = query / np.linalg.norm(query)
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    return np.dot(vectors, query)

class FlatIndex:
    def __init__(self,store):
        self.store = store
    def search(self,query_vector,k=5):
        vectors = self.store.get_vectors()
        scores = cosine_sim(query_vector,vectors)
        idx = np.argsort(scores)[-k:][::-1]

        results = []

        for i in idx:
            results.append({
                "id" : self.store.ids[i],
                "score": float(scores[i]),
                "text": self.store.texts[i],
                "metadata" : self.store.metadata[i]
            })

        return results