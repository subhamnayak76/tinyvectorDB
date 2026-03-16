from store import VectorStore
from index_flat import FlatIndex
from index_hnsw import HNSWIndex

store = VectorStore()
store.add(id="1", text="Vector databases are used in RAG",                metadata={"source": "blog"})
store.add(id="2", text="Transformers changed NLP",                        metadata={"source": "paper"})
store.add(id="3", text="Retrieval augmented generation improves LLM answers", metadata={"source": "research"})

flat_index    = FlatIndex(store)
hnsw_index    = HNSWIndex(store)

query        = "What is RAG?"
query_vector = store.embedder.encode(query)

print("=== Flat ===")
for r in flat_index.search(query_vector, k=3):
    print(r)

print("\n=== HNSW  ===")
for r in hnsw_index.search(query_vector, k=3):
    print(r)