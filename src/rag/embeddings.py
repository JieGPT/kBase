from typing import List, Optional
from openai import OpenAI
import os


class EmbeddingGenerator:
    def __init__(
        self,
        model: str = "text-embedding-3-small",
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        # Use provided values or fall back to legacy env var for backward compatibility
        api_key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY")
        base_url = base_url or os.getenv("BASE_URL")

        if not api_key:
            raise ValueError(
                "API key is required. Set API_KEY or OPENAI_API_KEY environment variable."
            )

        if base_url:
            self.client = OpenAI(api_key=api_key, base_url=base_url)
        else:
            self.client = OpenAI(api_key=api_key)
        self.model = model

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        response = self.client.embeddings.create(model=self.model, input=texts)
        return [item.embedding for item in response.data]

    def embed_query(self, query: str) -> List[float]:
        response = self.client.embeddings.create(model=self.model, input=[query])
        return response.data[0].embedding
