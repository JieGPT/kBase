import os
from pathlib import Path
from typing import Any
import yaml
from dotenv import load_dotenv
from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import field_validator
from pydantic import model_validator


class LLMConfig(BaseModel):
    provider: str = "openai"
    model: str = "gpt-4o-mini"
    temperature: float = 0.7
    max_tokens: int = 2048


class EmbeddingConfig(BaseModel):
    provider: str = "openai"
    model: str = "text-embedding-3-small"


class VectorDBConfig(BaseModel):
    type: str = "chromadb"
    persist_path: str = "./data/chromadb"
    collection_name: str = "documents"


class DocumentStorageConfig(BaseModel):
    path: str = "./docs"
    supported_formats: list[str] = [".pdf", ".txt"]


class ChunkingConfig(BaseModel):
    chunk_size: int = 1000
    chunk_overlap: int = 200


class MemoryConfig(BaseModel):
    max_turns: int = 5


class RetrievalConfig(BaseModel):
    top_k: int = 5


class AppConfig(BaseModel):
    name: str = "kBase"
    version: str = "0.1.0"
    debug: bool = False
    llm: LLMConfig
    embedding: EmbeddingConfig
    vector_db: VectorDBConfig
    document_storage: DocumentStorageConfig
    chunking: ChunkingConfig
    memory: MemoryConfig
    retrieval: RetrievalConfig

    @model_validator(mode="before")
    @classmethod
    def validate_config(cls, values):
        # Ensure required fields exist
        if "llm" not in values:
            raise ValueError("Missing required llm configuration")
        return values


class ConfigManager:
    def __init__(self, config_dir: str = "./config"):
        self.config_dir = Path(config_dir)
        env_path = self.config_dir / ".env"
        if env_path.exists():
            load_dotenv(env_path)
        else:
            print(f"Warning: .env file not found at {env_path}")
        self._config = self._load_config()

    def _load_config(self) -> AppConfig:
        config_path = self.config_dir / "config.yaml"
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        with open(config_path) as f:
            data = yaml.safe_load(f)
        return AppConfig(**data)

    @property
    def config(self) -> AppConfig:
        return self._config
