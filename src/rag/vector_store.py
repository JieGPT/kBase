from typing import List, Dict, Any
import chromadb
from .document_processor import Document


class VectorStore:
    def __init__(self, persist_path: str, collection_name: str = "documents"):
        self.client = chromadb.PersistentClient(path=persist_path)
        self.collection = self.client.get_or_create_collection(
            name=collection_name, metadata={"hnsw:space": "cosine"}
        )

    def add_documents(self, documents: List[Document], embeddings: List[List[float]]):
        ids = [doc.id for doc in documents]
        contents = [doc.content for doc in documents]
        metadatas = [doc.metadata or {} for doc in documents]

        self.collection.add(ids=ids, documents=contents, embeddings=embeddings, metadatas=metadatas)

    def search(self, query_embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        retrieved = []
        for i in range(len(results["ids"][0])):
            retrieved.append(
                {
                    "id": results["ids"][0][i],
                    "content": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "distance": results["distances"][0][i],
                }
            )

        return retrieved

    def count(self) -> int:
        return self.collection.count()

    def get_document_ids(self) -> List[str]:
        results = self.collection.get(include=[])
        return results["ids"]

    def delete_documents(self, ids: List[str]):
        if ids:
            self.collection.delete(ids=ids)

    def clear(self):
        all_ids = self.get_document_ids()
        if all_ids:
            self.collection.delete(ids=all_ids)
