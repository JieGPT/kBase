"""Vector store implementation using LangChain Chroma integration."""

from typing import List, Dict, Any
from langchain_chroma import Chroma
from langchain_core.documents import Document as LangchainDocument
from .document_processor import Document


class VectorStore:
    """Vector store using LangChain's Chroma integration."""

    def __init__(self, persist_path: str, collection_name: str = "documents"):
        """Initialize the vector store.

        Args:
            persist_path: Path to persist the database
            collection_name: Name of the collection
        """
        self.persist_path = persist_path
        self.collection_name = collection_name
        # Initialize Chroma vector store
        # Note: We don't pass embedding_function here as we handle embeddings externally
        self.client = Chroma(
            collection_name=collection_name,
            persist_directory=persist_path,
        )

    def add_documents(self, documents: List[Document], embeddings: List[List[float]]):
        """Add documents with pre-computed embeddings.

        Args:
            documents: List of Document objects
            embeddings: List of embedding vectors
        """
        # Convert to LangChain documents
        lc_docs = []
        for i, doc in enumerate(documents):
            lc_doc = LangchainDocument(
                page_content=doc.content,
                metadata={
                    "id": doc.id,
                    "source": doc.source,
                    "page": doc.page,
                    **doc.metadata,
                },
            )
            lc_docs.append(lc_doc)

        # Add documents with embeddings
        self.client.add_documents(lc_docs, embeddings=embeddings)

    def search(self, query_embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar documents.

        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return

        Returns:
            List of document dicts with id, content, metadata, distance
        """
        # Search by vector
        results = self.client.similarity_search_by_vector_with_relevance_scores(
            embedding=query_embedding,
            k=top_k,
        )

        # Convert to expected format
        retrieved = []
        for doc, score in results:
            # score is similarity (higher is better), convert to distance (lower is better)
            distance = 1.0 - score
            retrieved.append(
                {
                    "id": doc.metadata.get("id", ""),
                    "content": doc.page_content,
                    "metadata": {
                        "source": doc.metadata.get("source", ""),
                        "page": doc.metadata.get("page"),
                        **{
                            k: v
                            for k, v in doc.metadata.items()
                            if k not in ["id", "source", "page"]
                        },
                    },
                    "distance": distance,
                }
            )

        return retrieved

    def count(self) -> int:
        """Get the number of documents in the collection."""
        return self.client._collection.count()

    def get_document_ids(self) -> List[str]:
        """Get all document IDs in the collection."""
        results = self.client._collection.get(include=[])
        return results["ids"]

    def delete_documents(self, ids: List[str]):
        """Delete documents by ID."""
        if ids:
            self.client._collection.delete(ids=ids)

    def clear(self):
        """Clear all documents from the collection."""
        all_ids = self.get_document_ids()
        if all_ids:
            self.client._collection.delete(ids=all_ids)
