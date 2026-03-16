import math 
import random


class Node:
    def __init__(self,id,vector):
        self.id = id
        self.vector = vector
        self.neighbors = {}

def get_random_layer(M):
    ml = 1 / math.log(M)
    return math.floor(-math.log(random.random()) * ml)

def cosine_sim(a,b):
    dot    = sum(x*y for x,y in zip(a,b))
    norm_a = math.sqrt(sum(x**2 for x in a))
    norm_b = math.sqrt(sum(x**2 for x in b))
    return dot / (norm_a * norm_b)


def get_nearest(hnsw,query,candidates,k):
    results = []
    for id in candidates:
        score = cosine_sim(query,hnsw.nodes[id].vector)
        results.append((score,id))
    results.sort(reverse=True)
    return [id for score,id in results[:k]]


class HNSW:
    def __init__(self, M=16, ef_construction=200):
        self.M = M
        self.ef_construction = ef_construction
        self.nodes = []
    def connect(self, node_a, node_b, layer):
        node_a.neighbors.setdefault(layer, []).append(node_b.id)
        node_b.neighbors.setdefault(layer, []).append(node_a.id)

    def insert(self, vector):
      node = Node(len(self.nodes), vector)
      node.max_layer = get_random_layer(self.M)
      self.nodes.append(node)

      if len(self.nodes) == 1:
          return                   

      candidates =  [n.id for n in self.nodes if n.id != node.id]             
      
      for layer in range(node.max_layer, -1, -1):
          nearest = get_nearest(self, node.vector, candidates,  self.M)
          for neighbor_id in nearest:
              self.connect(node, self.nodes[neighbor_id], layer)
              
   

    def search(self, query, k=5):
        entry = self.nodes[-1]
        current = entry
        for layer in range(entry.max_layer, -1, -1):
            improved = True
            while improved:
                improved = False
                neighbors = current.neighbors.get(layer, [])
                nearest = get_nearest(self, query, neighbors, k=1)
                if nearest:
                    candidate = self.nodes[nearest[0]]
                    if cosine_sim(query, candidate.vector) > cosine_sim(query, current.vector):
                        current = candidate
                        improved = True
        neighbors = current.neighbors.get(0, [])
        return get_nearest(self, query, neighbors + [current.id], k)