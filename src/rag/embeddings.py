"""Embedding generation using LangChain."""

from typing import List, Optional
from langchain_openai import OpenAIEmbeddings


class EmbeddingGenerator:
    """Generate embeddings using LangChain's OpenAI embeddings."""

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        """Initialize the embedding generator.

        Args:
            model: Name of the embedding model
            base_url: Optional API base URL
            api_key: Optional API key
        """
        # Build kwargs for OpenAIEmbeddings
        kwargs = {"model": model}

        if api_key:
            kwargs["api_key"] = api_key
        if base_url:
            kwargs["base_url"] = base_url

        self.client = OpenAIEmbeddings(**kwargs)
        self.model = model

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple texts.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        return self.client.embed_documents(texts)

    def embed_query(self, query: str) -> List[float]:
        """Embed a single query.

        Args:
            query: Query text

        Returns:
            Embedding vector
        """
        return self.client.embed_query(query)
