import os
from pathlib import Path
from typing import Any, Optional
import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pydantic import model_validator


def get_env_or_default(env_var: str, default: Any) -> Any:
    """Get value from environment variable or return default."""
    value = os.getenv(env_var)
    if value is None:
        return default
    # Try to convert to appropriate type based on default
    if isinstance(default, bool):
        return value.lower() in ("true", "1", "yes", "on")
    elif isinstance(default, int):
        return int(value)
    elif isinstance(default, float):
        return float(value)
    return value


class LLMConfig(BaseModel):
    provider: str = "openai"
    model: str = Field(default="gpt-4o-mini", alias="MODEL")
    temperature: float = Field(default=0.7, alias="TEMPERATURE")
    max_tokens: int = Field(default=2048, alias="MAX_TOKENS")
    base_url: Optional[str] = Field(default=None, alias="BASE_URL")
    api_key: Optional[str] = Field(default=None, alias="API_KEY")

    @model_validator(mode="after")
    def apply_env_overrides(self):
        """Apply environment variable overrides."""
        self.model = get_env_or_default("MODEL", self.model)
        self.temperature = get_env_or_default("TEMPERATURE", self.temperature)
        self.max_tokens = get_env_or_default("MAX_TOKENS", self.max_tokens)
        self.base_url = get_env_or_default("BASE_URL", self.base_url)
        self.api_key = get_env_or_default("API_KEY", self.api_key)
        return self


class EmbeddingConfig(BaseModel):
    provider: str = "openai"
    model: str = Field(default="text-embedding-3-small", alias="EMBEDDING_MODEL")
    base_url: Optional[str] = Field(default=None, alias="BASE_URL")
    api_key: Optional[str] = Field(default=None, alias="API_KEY")

    @model_validator(mode="after")
    def apply_env_overrides(self):
        """Apply environment variable overrides."""
        self.model = get_env_or_default("EMBEDDING_MODEL", self.model)
        self.base_url = get_env_or_default("BASE_URL", self.base_url)
        self.api_key = get_env_or_default("API_KEY", self.api_key)
        return self


class VectorDBConfig(BaseModel):
    type: str = "chromadb"
    persist_path: str = Field(default="./data/chromadb", alias="VECTOR_DB_PERSIST_PATH")
    collection_name: str = Field(default="documents", alias="VECTOR_DB_COLLECTION")

    @model_validator(mode="after")
    def apply_env_overrides(self):
        """Apply environment variable overrides."""
        self.persist_path = get_env_or_default("VECTOR_DB_PERSIST_PATH", self.persist_path)
        self.collection_name = get_env_or_default("VECTOR_DB_COLLECTION", self.collection_name)
        return self


class DocumentStorageConfig(BaseModel):
    path: str = Field(default="./docs", alias="DOC_PATH")
    supported_formats: list[str] = [".pdf", ".txt"]


class ChunkingConfig(BaseModel):
    chunk_size: int = Field(default=1000, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=200, alias="CHUNK_OVERLAP")

    @model_validator(mode="after")
    def apply_env_overrides(self):
        """Apply environment variable overrides."""
        self.chunk_size = get_env_or_default("CHUNK_SIZE", self.chunk_size)
        self.chunk_overlap = get_env_or_default("CHUNK_OVERLAP", self.chunk_overlap)
        return self


class MemoryConfig(BaseModel):
    max_turns: int = Field(default=5, alias="MAX_TURNS")

    @model_validator(mode="after")
    def apply_env_overrides(self):
        """Apply environment variable overrides."""
        self.max_turns = get_env_or_default("MAX_TURNS", self.max_turns)
        return self


class RetrievalConfig(BaseModel):
    top_k: int = Field(default=5, alias="TOP_K")

    @model_validator(mode="after")
    def apply_env_overrides(self):
        """Apply environment variable overrides."""
        self.top_k = get_env_or_default("TOP_K", self.top_k)
        return self


class AppConfig(BaseModel):
    name: str = "kBase"
    version: str = "0.1.0"
    debug: bool = Field(default=False, alias="DEBUG")
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

    @model_validator(mode="after")
    def apply_env_overrides(self):
        """Apply environment variable overrides."""
        self.debug = get_env_or_default("DEBUG", self.debug)
        return self


class ConfigManager:
    def __init__(self, config_dir: str = "./config"):
        self.config_dir = Path(config_dir)

        # Load .env from project root first (if exists)
        root_env_path = Path(".env")
        if root_env_path.exists():
            load_dotenv(root_env_path)

        # Also load from config directory (if exists, for backward compatibility)
        env_path = self.config_dir / ".env"
        if env_path.exists():
            load_dotenv(env_path, override=True)

        if not root_env_path.exists() and not env_path.exists():
            print("Warning: .env file not found at ./.env or ./config/.env")

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
