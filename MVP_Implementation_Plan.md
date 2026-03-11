# MVP Implementation Plan: LLM Knowledge Exploration Application

## Overview

This MVP plan delivers a functional prototype in **4 weeks** with core features:
- CLI query interface
- OpenAI LLM integration
- RAG with semantic search (ChromaDB)
- Short-term conversation memory
- Basic document ingestion (PDF/TXT)

**Deferred to Full Implementation:**
- Meilisearch hybrid retrieval
- Web search agent
- Long-term memory (conversation persistence)
- Langfuse observability
- Multiple LLM providers
- Refinement/critique agent

---

## 1. MVP Project Structure

```
kBase/
├── config/
│   ├── config.yaml          # Simplified config
│   └── .env                 # Environment variables
├── src/
│   ├── __init__.py
│   ├── main.py              # CLI entry point
│   ├── cli/
│   │   ├── __init__.py
│   │   └── commands.py      # CLI commands
│   ├── core/
│   │   ├── __init__.py
│   │   └── config.py        # Configuration loader
│   ├── llm/
│   │   ├── __init__.py
│   │   ├── base.py          # Abstract LLM interface
│   │   └── openai_client.py # OpenAI implementation
│   ├── rag/
│   │   ├── __init__.py
│   │   ├── document_processor.py
│   │   ├── embeddings.py
│   │   └── vector_store.py
│   └── memory/
│       ├── __init__.py
│       └── conversation_buffer.py
├── docs/                    # Document storage
├── data/
│   └── chromadb/           # Vector DB persistence
├── pyproject.toml
└── README.md
```

---

## 2. MVP Configuration

### 2.1 Simplified Config (`config/config.yaml`)

```yaml
app:
  name: "kBase"
  version: "0.1.0-mvp"
  debug: false

llm:
  provider: "openai"
  model: "gpt-4o-mini"
  temperature: 0.7
  max_tokens: 2048

embedding:
  provider: "openai"
  model: "text-embedding-3-small"

vector_db:
  type: "chromadb"
  persist_path: "./data/chromadb"
  collection_name: "documents"

document_storage:
  path: "./docs"
  supported_formats:
    - ".pdf"
    - ".txt"

chunking:
  chunk_size: 1000
  chunk_overlap: 200

memory:
  max_turns: 5

retrieval:
  top_k: 5
  rerank_enabled: false  # Optional: enable for better retrieval quality
  rerank_model: "cross-encoder/ms-marco-MiniLM-L-6-v2"
  rerank_candidate_pool: 20
```

### 2.2 Environment Variables (`config/.env`)

```env
OPENAI_API_KEY=sk-your-key-here
```

---

## 3. MVP Implementation Phases

### Phase 1: Project Foundation (Days 1-3)

#### Day 1: Project Setup
- [ ] Initialize project with UV
- [ ] Create directory structure
- [ ] Setup pyproject.toml

```bash
uv init kBase
cd kBase
uv venv
source .venv/bin/activate
uv add click pyyaml python-dotenv pydantic pydantic-settings openai chromadb pypdf langchain-text-splitters tiktoken
uv add --dev pytest pytest-asyncio ruff
```

#### Day 2: Configuration Module
- [ ] Implement `src/core/config.py`
- [ ] Create config files

```python
# src/core/config.py
import os
from pathlib import Path
from typing import Any
import yaml
from dotenv import load_dotenv
from pydantic import BaseModel

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

class AppConfig(BaseModel):
    name: str = "kBase"
    version: str = "0.1.0"
    debug: bool = False
    llm: LLMConfig
    embedding: EmbeddingConfig
    vector_db: VectorDBConfig
    document_storage: dict
    chunking: dict
    memory: dict
    retrieval: dict

class ConfigManager:
    def __init__(self, config_dir: str = "./config"):
        self.config_dir = Path(config_dir)
        load_dotenv(self.config_dir / ".env")
        self._config = self._load_config()
    
    def _load_config(self) -> AppConfig:
        config_path = self.config_dir / "config.yaml"
        with open(config_path) as f:
            data = yaml.safe_load(f)
        return AppConfig(**data)
    
    @property
    def config(self) -> AppConfig:
        return self._config
```

#### Day 3: Basic CLI
- [ ] Implement `src/main.py`
- [ ] Implement `src/cli/commands.py`

```python
# src/main.py
import click
from src.core.config import ConfigManager
from src.cli.commands import CLI

def main():
    config = ConfigManager()
    cli = CLI(config)
    cli.run()

if __name__ == "__main__":
    main()
```

```python
# src/cli/commands.py
import click
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory

class CLI:
    def __init__(self, config):
        self.config = config
        self.session = PromptSession(history=FileHistory('.kbase_history'))
        self.running = True
    
    def run(self):
        click.echo(f"Welcome to {self.config.config.name} v{self.config.config.version}")
        click.echo("Type 'help' for commands, 'exit' to quit.\n")
        
        while self.running:
            try:
                user_input = self.session.prompt(">>> ").strip()
                if not user_input:
                    continue
                self.handle_input(user_input)
            except KeyboardInterrupt:
                click.echo("\nGoodbye!")
                self.running = False
            except EOFError:
                self.running = False
    
    def handle_input(self, user_input: str):
        parts = user_input.split(maxsplit=1)
        command = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""
        
        if command == "exit" or command == "quit":
            click.echo("Goodbye!")
            self.running = False
        elif command == "help":
            self.show_help()
        elif command == "ingest":
            self.ingest_documents(args)
        elif command == "list-docs":
            self.list_documents()
        else:
            click.echo(f"Unknown command: {command}")
    
    def show_help(self):
        help_text = """
Available Commands:
  help              Show this help message
  exit / quit       Exit the application
  ingest [path]     Ingest documents from path (default: ./docs)
  list-docs         List indexed documents
  <query>           Ask a question (not implemented in MVP phase 1)
        """
        click.echo(help_text)
    
    def ingest_documents(self, path: str = ""):
        click.echo(f"Ingesting documents... (not yet implemented)")
    
    def list_documents(self):
        click.echo("Listing documents... (not yet implemented)")
```

---

### Phase 2: LLM Integration (Days 4-5)

#### Day 4: LLM Base & OpenAI Client
- [ ] Implement `src/llm/base.py`
- [ ] Implement `src/llm/openai_client.py`

```python
# src/llm/base.py
from abc import ABC, abstractmethod
from typing import AsyncIterator, Optional

class LLMResponse:
    def __init__(self, content: str, tokens_used: int, model: str):
        self.content = content
        self.tokens_used = tokens_used
        self.model = model

class BaseLLM(ABC):
    @abstractmethod
    async def generate(self, prompt: str, **kwargs) -> LLMResponse:
        pass
    
    @abstractmethod
    async def stream(self, prompt: str, **kwargs) -> AsyncIterator[str]:
        pass
    
    @abstractmethod
    def count_tokens(self, text: str) -> int:
        pass
```

```python
# src/llm/openai_client.py
import os
from typing import AsyncIterator
from openai import AsyncOpenAI
import tiktoken
from .base import BaseLLM, LLMResponse

class OpenAIClient(BaseLLM):
    def __init__(self, model: str = "gpt-4o-mini", temperature: float = 0.7, max_tokens: int = 2048):
        self.client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._encoding = tiktoken.encoding_for_model(model)
    
    async def generate(self, prompt: str, **kwargs) -> LLMResponse:
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=kwargs.get("temperature", self.temperature),
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
        )
        return LLMResponse(
            content=response.choices[0].message.content,
            tokens_used=response.usage.total_tokens,
            model=self.model
        )
    
    async def stream(self, prompt: str, **kwargs) -> AsyncIterator[str]:
        stream = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=kwargs.get("temperature", self.temperature),
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
            stream=True,
        )
        async for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    
    def count_tokens(self, text: str) -> int:
        return len(self._encoding.encode(text))
    
    async def generate_with_messages(self, messages: list, **kwargs) -> LLMResponse:
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=kwargs.get("temperature", self.temperature),
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
        )
        return LLMResponse(
            content=response.choices[0].message.content,
            tokens_used=response.usage.total_tokens,
            model=self.model
        )
```

#### Day 5: Test LLM Integration
- [ ] Write basic tests
- [ ] Verify OpenAI connection

```python
# tests/test_llm.py
import pytest
from src.llm.openai_client import OpenAIClient

@pytest.mark.asyncio
async def test_generate():
    llm = OpenAIClient(model="gpt-4o-mini")
    response = await llm.generate("Say 'hello'")
    assert response.content
    assert response.tokens_used > 0

@pytest.mark.asyncio
async def test_stream():
    llm = OpenAIClient(model="gpt-4o-mini")
    chunks = []
    async for chunk in llm.stream("Say 'hello'"):
        chunks.append(chunk)
    assert "".join(chunks)
```

---

### Phase 3: RAG Module (Days 6-10)

#### Day 6: Document Processor
- [ ] Implement `src/rag/document_processor.py`

```python
# src/rag/document_processor.py
import os
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter

@dataclass
class Document:
    id: str
    content: str
    source: str
    page: Optional[int] = None
    metadata: dict = None

class DocumentProcessor:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    
    def parse_pdf(self, file_path: Path) -> List[str]:
        reader = PdfReader(file_path)
        pages = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text.strip():
                pages.append((i + 1, text))
        return pages
    
    def parse_txt(self, file_path: Path) -> List[str]:
        with open(file_path, 'r', encoding='utf-8') as f:
            return [(None, f.read())]
    
    def process_document(self, file_path: Path) -> List[Document]:
        suffix = file_path.suffix.lower()
        filename = file_path.name
        
        if suffix == '.pdf':
            pages = self.parse_pdf(file_path)
        elif suffix == '.txt':
            pages = self.parse_txt(file_path)
        else:
            return []
        
        documents = []
        for page_num, text in pages:
            chunks = self.splitter.split_text(text)
            for i, chunk in enumerate(chunks):
                doc = Document(
                    id=f"{filename}_{page_num or 0}_{i}",
                    content=chunk,
                    source=filename,
                    page=page_num,
                    metadata={"source": filename, "page": page_num, "chunk_index": i}
                )
                documents.append(doc)
        
        return documents
    
    def process_directory(self, dir_path: str, extensions: List[str] = ['.pdf', '.txt']) -> List[Document]:
        dir_path = Path(dir_path)
        all_documents = []
        
        for ext in extensions:
            for file_path in dir_path.glob(f"*{ext}"):
                documents = self.process_document(file_path)
                all_documents.extend(documents)
        
        return all_documents
```

#### Day 7: Embeddings
- [ ] Implement `src/rag/embeddings.py`

```python
# src/rag/embeddings.py
from typing import List
from openai import OpenAI
import os

class EmbeddingGenerator:
    def __init__(self, model: str = "text-embedding-3-small"):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = model
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        response = self.client.embeddings.create(
            model=self.model,
            input=texts
        )
        return [item.embedding for item in response.data]
    
    def embed_query(self, query: str) -> List[float]:
        response = self.client.embeddings.create(
            model=self.model,
            input=[query]
        )
        return response.data[0].embedding
```

#### Day 8-9: Vector Store
- [ ] Implement `src/rag/vector_store.py`

```python
# src/rag/vector_store.py
from typing import List, Optional, Dict, Any
import chromadb
from chromadb.config import Settings
from .document_processor import Document

class VectorStore:
    def __init__(self, persist_path: str, collection_name: str = "documents"):
        self.client = chromadb.PersistentClient(path=persist_path)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
    
    def add_documents(self, documents: List[Document], embeddings: List[List[float]]):
        ids = [doc.id for doc in documents]
        contents = [doc.content for doc in documents]
        metadatas = [doc.metadata or {} for doc in documents]
        
        self.collection.add(
            ids=ids,
            documents=contents,
            embeddings=embeddings,
            metadatas=metadatas
        )
    
    def search(self, query_embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )
        
        retrieved = []
        for i in range(len(results['ids'][0])):
            retrieved.append({
                "id": results['ids'][0][i],
                "content": results['documents'][0][i],
                "metadata": results['metadatas'][0][i],
                "distance": results['distances'][0][i]
            })
        
        return retrieved
    
    def count(self) -> int:
        return self.collection.count()
    
    def get_document_ids(self) -> List[str]:
        results = self.collection.get(include=[])
        return results['ids']
    
    def delete_documents(self, ids: List[str]):
        if ids:
            self.collection.delete(ids=ids)
    
    def clear(self):
        all_ids = self.get_document_ids()
        if all_ids:
            self.collection.delete(ids=all_ids)
```

#### Day 10: RAG Integration Test
- [ ] Test end-to-end document ingestion and retrieval

```python
# tests/test_rag.py
import pytest
from src.rag.document_processor import DocumentProcessor
from src.rag.embeddings import EmbeddingGenerator
from src.rag.vector_store import VectorStore
import tempfile
import os

@pytest.fixture
def temp_vector_store():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = VectorStore(persist_path=tmpdir)
        yield store

def test_document_processor():
    processor = DocumentProcessor(chunk_size=500, chunk_overlap=50)
    # Create test file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("This is a test document. " * 50)
        temp_path = f.name
    
    documents = processor.process_document(temp_path)
    os.unlink(temp_path)
    
    assert len(documents) > 0
    assert all(doc.content for doc in documents)

@pytest.mark.asyncio
async def test_rag_pipeline(temp_vector_store):
    processor = DocumentProcessor()
    embedder = EmbeddingGenerator()
    
    # Process and embed
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("Python is a programming language. Machine learning uses Python.")
        temp_path = f.name
    
    documents = processor.process_document(temp_path)
    os.unlink(temp_path)
    
    embeddings = embedder.embed_texts([doc.content for doc in documents])
    temp_vector_store.add_documents(documents, embeddings)
    
    # Search
    query_embedding = embedder.embed_query("What is Python?")
    results = temp_vector_store.search(query_embedding, top_k=3)
    
    assert len(results) > 0
    assert "Python" in results[0]['content']
```

---

### Phase 3.5 (Optional): Re-ranking Module (Day 10-11 Extension)

**Objective:** Improve retrieval quality by adding a re-ranking step after initial semantic search.

**Why Re-ranking?**
- Cross-encoders achieve significantly better relevance scoring than bi-encoders (embeddings)
- Filters out false positives from initial semantic search
- Especially useful when query intent differs from semantic similarity

**Implementation:**

```python
# src/rag/reranker.py (Optional enhancement)
from typing import List
from sentence_transformers import CrossEncoder

class CrossEncoderReranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model = CrossEncoder(model_name)
    
    def rerank(self, query: str, documents: List[Document], top_k: int = 5) -> List[Document]:
        if not documents:
            return documents
        
        # Create query-document pairs
        pairs = [(query, doc.content) for doc in documents]
        
        # Get relevance scores from cross-encoder
        scores = self.model.predict(pairs)
        
        # Sort by score and return top-k
        scored_docs = list(zip(documents, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        return [doc for doc, _ in scored_docs[:top_k]]
```

**Integration in CLI:**

```python
# In src/cli/commands.py __init__
if config.config.retrieval.rerank_enabled:
    from src.rag.reranker import CrossEncoderReranker
    self.reranker = CrossEncoderReranker(
        model_name=config.config.retrieval.rerank_model
    )
else:
    self.reranker = None

# In process_query method
async def process_query(self, query: str):
    self.memory.add_user_message(query)
    
    # Retrieve candidates (larger pool if re-ranking)
    candidate_k = self.config.config.retrieval.rerank_candidate_pool \
        if self.reranker else self.config.config.retrieval.top_k
    
    query_embedding = self.embedder.embed_query(query)
    results = self.vector_store.search(query_embedding, top_k=candidate_k)
    
    # Re-rank if enabled
    if self.reranker:
        results = self.reranker.rerank(
            query, 
            results, 
            top_k=self.config.config.retrieval.top_k
        )
    
    # Continue with context building...
```

**Dependencies:**
```bash
uv add sentence-transformers
```

**Configuration:**
```yaml
retrieval:
  top_k: 5
  rerank_enabled: true  # Set to false to disable
  rerank_model: "cross-encoder/ms-marco-MiniLM-L-6-v2"
  rerank_candidate_pool: 20  # Retrieve 20, re-rank to 5
```

**Trade-offs:**
- ✅ Significantly improved retrieval accuracy
- ✅ Better handling of precise queries
- ❌ Additional latency (~100-500ms for 20 documents)
- ❌ Additional memory usage for cross-encoder model
- ❌ Extra dependency (sentence-transformers)

---

### Phase 4: Memory & Query (Days 11-14)

#### Day 11: Conversation Buffer
- [ ] Implement `src/memory/conversation_buffer.py`

```python
# src/memory/conversation_buffer.py
from typing import List, Dict
from dataclasses import dataclass, field
from datetime import datetime

@dataclass
class Message:
    role: str
    content: str
    timestamp: datetime = field(default_factory=datetime.now)

class ConversationBuffer:
    def __init__(self, max_turns: int = 5):
        self.max_turns = max_turns
        self.messages: List[Message] = []
    
    def add_user_message(self, content: str):
        self.messages.append(Message(role="user", content=content))
        self._trim_if_needed()
    
    def add_assistant_message(self, content: str):
        self.messages.append(Message(role="assistant", content=content))
        self._trim_if_needed()
    
    def _trim_if_needed(self):
        if len(self.messages) > self.max_turns * 2:
            self.messages = self.messages[-(self.max_turns * 2):]
    
    def get_messages(self) -> List[Dict[str, str]]:
        return [{"role": msg.role, "content": msg.content} for msg in self.messages]
    
    def clear(self):
        self.messages = []
    
    def get_context_string(self) -> str:
        if not self.messages:
            return ""
        lines = []
        for msg in self.messages:
            prefix = "User" if msg.role == "user" else "Assistant"
            lines.append(f"{prefix}: {msg.content}")
        return "\n".join(lines)
```

#### Day 12-13: Query Processing
- [ ] Implement query handling in CLI
- [ ] Integrate LLM + RAG

```python
# Add to src/cli/commands.py
from src.llm.openai_client import OpenAIClient
from src.rag.embeddings import EmbeddingGenerator
from src.rag.vector_store import VectorStore
from src.memory.conversation_buffer import ConversationBuffer
from src.rag.document_processor import DocumentProcessor

class CLI:
    def __init__(self, config):
        self.config = config
        self.session = PromptSession(history=FileHistory('.kbase_history'))
        self.running = True
        
        # Initialize components
        self.llm = OpenAIClient(
            model=config.config.llm.model,
            temperature=config.config.llm.temperature,
            max_tokens=config.config.llm.max_tokens
        )
        self.embedder = EmbeddingGenerator(model=config.config.embedding.model)
        self.vector_store = VectorStore(
            persist_path=config.config.vector_db.persist_path,
            collection_name=config.config.vector_db.collection_name
        )
        self.memory = ConversationBuffer(max_turns=config.config.memory.max_turns)
        self.processor = DocumentProcessor(
            chunk_size=config.config.chunking.chunk_size,
            chunk_overlap=config.config.chunking.chunk_overlap
        )
    
    def handle_input(self, user_input: str):
        parts = user_input.split(maxsplit=1)
        command = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""
        
        if command in ("exit", "quit"):
            click.echo("Goodbye!")
            self.running = False
        elif command == "help":
            self.show_help()
        elif command == "ingest":
            self.ingest_documents(args)
        elif command == "list-docs":
            self.list_documents()
        elif command == "clear":
            self.memory.clear()
            click.echo("Conversation cleared.")
        else:
            self.process_query(user_input)
    
    async def process_query(self, query: str):
        self.memory.add_user_message(query)
        
        # Retrieve relevant documents
        query_embedding = self.embedder.embed_query(query)
        results = self.vector_store.search(
            query_embedding, 
            top_k=self.config.config.retrieval.top_k
        )
        
        # Optional: Re-ranking for improved retrieval quality
        # if self.config.config.retrieval.rerank_enabled:
        #     results = await self.rerank_results(query, results)
        
        # Build context
        if results:
            context = "\n\n".join([
                f"[{r['metadata'].get('source', 'Unknown')}, Page {r['metadata'].get('page', 'N/A')}]\n{r['content']}"
                for r in results
            ])
        else:
            context = "No relevant documents found."
        
        # Build prompt
        conversation_context = self.memory.get_context_string()
        prompt = f"""Based on the following documents, answer the user's question.
If the answer is not in the documents, say so.

Documents:
{context}

Conversation History:
{conversation_context}

Question: {query}

Answer:"""
        
        # Stream response
        click.echo("\nAssistant: ", nl=False)
        full_response = ""
        async for chunk in self.llm.stream(prompt):
            click.echo(chunk, nl=False)
            full_response += chunk
        click.echo("\n")
        
        self.memory.add_assistant_message(full_response)
    
    def ingest_documents(self, path: str = ""):
        doc_path = path or self.config.config.document_storage.path
        
        click.echo(f"Processing documents from: {doc_path}")
        documents = self.processor.process_directory(
            doc_path,
            extensions=self.config.config.document_storage.supported_formats
        )
        
        if not documents:
            click.echo("No documents found.")
            return
        
        click.echo(f"Found {len(documents)} chunks. Generating embeddings...")
        
        # Generate embeddings in batches
        batch_size = 100
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i+batch_size]
            embeddings = self.embedder.embed_texts([doc.content for doc in batch])
            self.vector_store.add_documents(batch, embeddings)
            click.echo(f"  Processed {min(i+batch_size, len(documents))}/{len(documents)} chunks")
        
        click.echo(f"Done! Indexed {len(documents)} chunks.")
    
    def list_documents(self):
        count = self.vector_store.count()
        click.echo(f"Total indexed chunks: {count}")
        
        # Get unique sources
        results = self.vector_store.collection.get(include=["metadatas"])
        sources = set()
        for meta in (results.get('metadatas') or []):
            if meta and 'source' in meta:
                sources.add(meta['source'])
        
        if sources:
            click.echo("\nDocuments:")
            for source in sorted(sources):
                click.echo(f"  - {source}")
```

#### Day 14: Final Integration & Testing
- [ ] Run end-to-end test
- [ ] Fix bugs
- [ ] Create README

```markdown
# kBase MVP

A simple CLI application for querying your documents using RAG.

## Quick Start

1. Install dependencies:
```bash
uv sync
```

2. Set your OpenAI API key:
```bash
export OPENAI_API_KEY=sk-your-key
```

3. Run the application:
```bash
uv run python src/main.py
```

## Usage

```
>>> ingest ./docs
Processing documents from: ./docs
Found 50 chunks. Generating embeddings...
Done! Indexed 50 chunks.

>>> What is machine learning?
Assistant: Based on the documents, machine learning is...

>>> list-docs
Total indexed chunks: 50

Documents:
  - report.pdf
  - notes.txt

>>> clear
Conversation cleared.

>>> exit
Goodbye!
```
```

---

## 4. MVP Dependencies

```toml
# pyproject.toml
[project]
name = "kbase"
version = "0.1.0"
description = "LLM Knowledge Exploration MVP"
requires-python = ">=3.10"

dependencies = [
    "click>=8.1.0",
    "pyyaml>=6.0",
    "python-dotenv>=1.0.0",
    "pydantic>=2.0.0",
    "pydantic-settings>=2.0.0",
    "openai>=1.0.0",
    "tiktoken>=0.5.0",
    "chromadb>=0.4.0",
    "pypdf>=3.0.0",
    "langchain-text-splitters>=0.0.1",
    "prompt-toolkit>=3.0.0",
]

[tool.uv]
dev-dependencies = [
    "pytest>=7.4.0",
    "pytest-asyncio>=0.21.0",
    "ruff>=0.1.0",
]
```

---

## 5. MVP Timeline Summary

| Day | Task | Deliverable |
|-----|------|-------------|
| 1 | Project setup | UV project, directory structure |
| 2 | Configuration | Config loader, YAML files |
| 3 | Basic CLI | Interactive prompt, commands |
| 4 | LLM base | Abstract interface, OpenAI client |
| 5 | LLM tests | Working LLM integration |
| 6 | Document processor | PDF/TXT parsing, chunking |
| 7 | Embeddings | OpenAI embedding generation |
| 8-9 | Vector store | ChromaDB integration |
| 10 | RAG tests | Working document retrieval |
| 11 | Memory buffer | Conversation context |
| 12-13 | Query processing | End-to-end Q&A |
| 14 | Final testing | Bug fixes, documentation |

---

## 6. MVP Success Criteria

- [ ] CLI starts and accepts commands
- [ ] Documents can be ingested from a folder
- [ ] Queries return relevant answers from documents
- [ ] Conversation context is maintained
- [ ] Application handles errors gracefully
- [ ] Basic README with usage instructions

---

## 7. Post-MVP Roadmap

After MVP is working:

1. **Week 5-6:** Add Meilisearch for hybrid retrieval
2. **Week 6-7:** Add re-ranking for improved retrieval quality
   - Implement `src/rag/reranker.py` with cross-encoder support
   - Add configuration for re-ranking (enable/disable, model selection)
   - Integrate re-ranking into query processing pipeline
   - Test and evaluate retrieval quality improvements
3. **Week 7-8:** Add conversation persistence (SQLite)
4. **Week 9:** Add Langfuse observability
5. **Week 10:** Add web search agent
6. **Week 11-12:** Add multiple LLM providers
7. **Week 13-14:** Add refinement agent, advanced features

---

## 8. Quick Commands

```bash
# Setup
uv venv && source .venv/bin/activate
uv sync

# Run
uv run python src/main.py

# Test
uv run pytest

# Lint
uv run ruff check src/
```
