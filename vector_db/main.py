from store import VectorStore
from index_flat import FlatIndex

store = VectorStore()
store.add(
    id="1",
    text="Vector databases are used in RAG",
    metadata={"source": "blog"}
)

store.add(
    id="2",
    text="Transformers changed NLP",
    metadata={"source": "paper"}
)

store.add(
    id="3",
    text="Retrieval augmented generation improves LLM answers",
    metadata={"source": "research"}
)

index = FlatIndex(store)
query = "What is RAG?"
query_vector = store.embedder.encode(query)
results = index.search(query_vector, k=3)
for r in results:
    print(r)