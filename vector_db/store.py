import numpy as np
from embedding import Embedder
class VectorStore:
    def __init__(self):
        self.ids = []
        self.vectors = []
        self.texts = []
        self.metadata = []
        self.embedder = Embedder()
        
    def add(self,id,text=None,metadata=None):
        vector = self.embedder.encode(text)
        self.ids.append(id)
        self.vectors.append(np.array(vector))
        self.texts.append(text)
        self.metadata.append(metadata)

    def get_vectors(self):
        return np.vstack(self.vectors)
    
    def __len__(self):
        return len(self.ids)