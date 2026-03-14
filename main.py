from store import VectorStore

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

print("Stored documents:", len(store))

print("\nVectors:")
print(store.get_vectors())