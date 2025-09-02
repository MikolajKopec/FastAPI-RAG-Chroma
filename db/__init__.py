from pathlib import Path
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
PERSIST_DIR = Path("vectordb")
PERSIST_DIR.mkdir(parents=True, exist_ok=True)


class VectorStore:
    def __init__(self):
        self.vs = Chroma(
            collection_name="rag_docs",
            embedding_function=embeddings,
            persist_directory=str(PERSIST_DIR),
        )

    def get_all(self, limit: int = 1000, offset: int = 0, include=None):
        if not include:
            include = ["metadatas", "documents"]
        return self.vs._collection.get(
            limit=limit,
            offset=offset,
            include=include,
        )

    def add_texts(self, texts, metadatas, ids) -> int:
        added = 0
        try:
            self.vs.add_texts(texts, metadatas=metadatas, ids=ids)
            added = len(texts)
        except Exception as e:
            print(e)
        self.vs.persist()
        return added

    def get_retriever(self, k: int = 4):
        return self.vs.as_retriever(search_kwargs={"k": k})


vector_store = VectorStore()


def get_vector_store() -> VectorStore:
    """Dependency to inject the global VectorStore instance."""
    return vector_store
