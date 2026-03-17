import numpy as np
from sklearn.cluster import KMeans


def cosine_sim(query,vectors):
    query = query / np.linalg.norm(query)
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    return np.dot(vectors, query)

class IVFIndex:
  def __init__(self,store,n_cluster,n_probe):
    self.store = store
    self.n_cluster = n_cluster
    self.n_probe = n_probe
    self.centroids = None
    self.clusters = []

  def train(self, vectors):
    kmeans = KMeans(n_clusters=self.n_cluster)
    kmeans.fit(vectors)
    self.centroids  = kmeans.cluster_centers_
    labels = kmeans.labels_
    self.clusters = [[] for _ in range(self.n_cluster)]
    for i in range(len(labels)):
      self.clusters[labels[i]].append(i)

  def search(self,query_vector,k=5):
    sim_scores = cosine_sim(query_vector, self.centroids)
    top_centroids = np.argsort(sim_scores)[-self.n_probe:][::-1]
    candidates = []
    for i in top_centroids:
      candidates.extend(self.clusters[i])

    results = []
    for i in candidates:
      score = cosine_sim(query_vector,self.store.vectors[i].reshape(1, -1))
      results.append((score,i))

    results.sort(reverse=True)
    final = []
    for score, i in results[:k]:
        final.append({
            "id": self.store.ids[i],
            "score": float(score[0]),
            "text": self.store.texts[i],
            "metadata": self.store.metadata[i]
        })
    return final