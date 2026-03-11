# Implementation Plan: Modular LLM Knowledge Exploration Framework

## 1. Project Structure

```
kBase/
├── config/
│   ├── config.yaml              # Main configuration file
│   ├── llm.yaml                 # LLM providers and models config
│   ├── agents.yaml              # Agentic workflow definitions
│   ├── .env.example             # Environment variables template
│   └── .env                     # Actual environment variables (gitignored)
├── src/
│   ├── __init__.py
│   ├── main.py                  # CLI entry point
│   ├── cli/
│   │   ├── __init__.py
│   │   ├── commands.py          # CLI command definitions
│   │   └── formatters.py        # Output formatting utilities
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py            # Configuration loader
│   │   └── exceptions.py        # Custom exceptions
│   ├── llm/
│   │   ├── __init__.py
│   │   ├── base.py              # Abstract LLM interface
│   │   ├── openai_compatible.py # OpenAI-compatible implementation
│   │   └── factory.py           # LLM factory for dynamic selection
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── base.py              # Base agent class
│   │   ├── router.py            # Router agent
│   │   ├── web_search.py        # Web search agent
│   │   ├── rag.py               # RAG agent
│   │   ├── summarizer.py        # Summarization agent
│   │   ├── refiner.py           # Refinement/critique agent
│   │   ├── final_answer.py      # Final answer generation agent
│   │   └── workflow.py          # LangGraph workflow orchestrator
│   ├── rag/
│   │   ├── __init__.py
│   │   ├── document_processor.py # PDF/TXT parsing and chunking
│   │   ├── embeddings.py         # Embedding generation
│   │   ├── vector_store.py       # Vector DB interface
│   │   └── retriever.py          # Retrieval logic
│   ├── memory/
│   │   ├── __init__.py
│   │   ├── short_term.py         # Conversation buffer
│   │   ├── long_term.py          # Persistent storage
│   │   └── models.py             # DB models (SQLAlchemy)
│   ├── observability/
│   │   ├── __init__.py
│   │   ├── langfuse_handler.py   # Langfuse integration
│   │   └── tracing.py            # Trace/span utilities
│   └── utils/
│       ├── __init__.py
│       ├── helpers.py            # Utility functions
│       └── validators.py         # Input validation
├── docs/
│   └── (document storage folder)
├── data/
│   ├── vector_db/               # Vector database persistence
│   └── conversations.db         # SQLite conversation storage
├── tests/
│   ├── __init__.py
│   ├── test_llm/
│   ├── test_agents/
│   ├── test_rag/
│   └── test_memory/
├── pyproject.toml
├── uv.lock
└── README.md
```

## 2. Configuration Files Design

### 2.1 Main Configuration (`config/config.yaml`)

```yaml
app:
  name: "kBase"
  version: "1.0.0"
  debug: false

document_storage:
  path: "./docs"
  watch_enabled: true
  supported_formats:
    - ".pdf"
    - ".txt"

vector_db:
  type: "chromadb"  # Options: chromadb, faiss, qdrant
  persist_path: "./data/vector_db"
  collection_name: "documents"

embedding:
  provider: "openai"  # Options: openai, huggingface, sentence-transformers
  model: "text-embedding-3-small"
  chunk_size: 1000
  chunk_overlap: 200

memory:
  short_term:
    max_turns: 10
    summarization_enabled: true
    summarization_threshold: 8
  long_term:
    database_path: "./data/conversations.db"
    auto_title_enabled: true

observability:
  langfuse:
    enabled: true
    public_key: "${LANGFUSE_PUBLIC_KEY}"
    secret_key: "${LANGFUSE_SECRET_KEY}"
    host: "https://cloud.langfuse.com"
    session_id_tracking: true

cli:
  prompt_style: ">>>"
  pagination_enabled: true
  page_size: 50
```

### 2.2 LLM Configuration (`config/llm.yaml`)

```yaml
default_provider: "openai"
default_model: "gpt-4"

providers:
  openai:
    api_base: "https://api.openai.com/v1"
    api_key: "${OPENAI_API_KEY}"
    models:
      - name: "gpt-4"
        max_tokens: 8192
        temperature: 0.7
      - name: "gpt-4-turbo"
        max_tokens: 128000
        temperature: 0.7
      - name: "gpt-3.5-turbo"
        max_tokens: 16384
        temperature: 0.7

  ollama:
    api_base: "http://localhost:11434/v1"
    api_key: "ollama"  # Placeholder
    models:
      - name: "llama2"
        max_tokens: 4096
        temperature: 0.7
      - name: "mistral"
        max_tokens: 8192
        temperature: 0.7

  custom:
    api_base: "${CUSTOM_LLM_API_BASE}"
    api_key: "${CUSTOM_LLM_API_KEY}"
    models:
      - name: "custom-model"
        max_tokens: 8192
        temperature: 0.7

retry:
  max_attempts: 3
  initial_delay: 1.0
  max_delay: 30.0
  exponential_base: 2

rate_limit:
  requests_per_minute: 60
  tokens_per_minute: 90000
```

### 2.3 Agent Workflow Configuration (`config/agents.yaml`)

```yaml
workflow:
  name: "knowledge_exploration"
  max_iterations: 5

agents:
  router:
    enabled: true
    model: "${LLM_ROUTER_MODEL:gpt-3.5-turbo}"
    temperature: 0.3
    prompt_template: |
      Analyze the user query and determine the best action:
      - DIRECT: Answer directly from LLM knowledge
      - WEB_SEARCH: Search the web for current information
      - RAG: Search local documents
      - COMBINED: Use multiple sources
      
      Query: {query}
      
      Respond with only the action type.

  web_search:
    enabled: true
    provider: "brave"  # Options: brave, google, serpapi
    max_results: 5
    timeout: 30
    api_key: "${WEB_SEARCH_API_KEY}"

  rag:
    enabled: true
    model: "${LLM_RAG_MODEL:gpt-4}"
    temperature: 0.5
    top_k_results: 5
    min_relevance_score: 0.7
    prompt_template: |
      Based on the following documents, answer the query.
      
      Documents:
      {documents}
      
      Query: {query}
      
      Provide a comprehensive answer with citations.

  summarizer:
    enabled: true
    model: "${LLM_SUMMARIZER_MODEL:gpt-3.5-turbo}"
    temperature: 0.3
    default_style: "bullet_points"  # bullet_points, paragraph, executive
    max_summary_length: 500

  refiner:
    enabled: true
    model: "${LLM_REFINER_MODEL:gpt-4}"
    temperature: 0.4
    max_iterations: 2

  final_answer:
    enabled: true
    model: "${LLM_FINAL_MODEL:gpt-4}"
    temperature: 0.5
    prompt_template: |
      Synthesize the following information into a comprehensive answer.
      
      Original Query: {query}
      
      Sources:
      {sources}
      
      Provide a clear, well-structured response.

edges:
  - from: "router"
    to: "web_search"
    condition: "action == 'WEB_SEARCH' or action == 'COMBINED'"
  - from: "router"
    to: "rag"
    condition: "action == 'RAG' or action == 'COMBINED'"
  - from: "router"
    to: "final_answer"
    condition: "action == 'DIRECT'"
  - from: "web_search"
    to: "summarizer"
  - from: "rag"
    to: "summarizer"
  - from: "summarizer"
    to: "refiner"
  - from: "refiner"
    to: "final_answer"
```

### 2.4 Environment Variables (`config/.env.example`)

```env
# LLM Provider Keys
OPENAI_API_KEY=sk-your-openai-key-here
CUSTOM_LLM_API_KEY=your-custom-key
CUSTOM_LLM_API_BASE=https://your-api-endpoint.com/v1

# Web Search API
WEB_SEARCH_API_KEY=your-search-api-key

# Langfuse Observability
LANGFUSE_PUBLIC_KEY=pk-lf-xxx
LANGFUSE_SECRET_KEY=sk-lf-xxx
LANGFUSE_HOST=https://cloud.langfuse.com

# Optional Model Overrides
LLM_ROUTER_MODEL=gpt-3.5-turbo
LLM_RAG_MODEL=gpt-4
LLM_SUMMARIZER_MODEL=gpt-3.5-turbo
LLM_REFINER_MODEL=gpt-4
LLM_FINAL_MODEL=gpt-4

# Application Settings
APP_DEBUG=false
LOG_LEVEL=INFO
```

## 3. Implementation Phases

### Phase 1: Foundation (Week 1-2)

#### Step 1.1: Project Setup with UV

**What is UV?**
UV is an extremely fast Python package installer and resolver, written in Rust. It serves as a drop-in replacement for pip, pip-tools, and virtualenv, offering 10-100x speed improvements.

**Installation:**
```bash
# Install UV (choose one method)
curl -LsSf https://astral.sh/uv/install.sh | sh
# OR
pip install uv
# OR
brew install uv  # macOS
```

**Setup Steps:**
- [ ] Install UV package manager
- [ ] Initialize Python project with `uv init`
- [ ] Create directory structure
- [ ] Setup virtual environment with `uv venv`
- [ ] Add core dependencies with `uv add`
- [ ] Setup logging infrastructure
- [ ] Configure UV settings in `pyproject.toml`

**Commands:**
```bash
# Initialize project
uv init kBase
cd kBase

# Create virtual environment
uv venv

# Activate virtual environment
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# Add dependencies
uv add click pyyaml python-dotenv pydantic

# Add dev dependencies
uv add --dev pytest pytest-asyncio pytest-cov ruff myk

# Install all dependencies
uv sync

# Run commands with UV
uv run python src/main.py
uv run pytest
```

**Core Dependencies (Phase 1):**
```toml
# pyproject.toml (generated by uv init, then edited)
[project]
name = "kbase"
version = "0.1.0"
description = "LLM-Powered Knowledge Exploration Application"
readme = "README.md"
requires-python = ">=3.10"
license = { text = "MIT" }
authors = [{ name = "Your Name", email = "your@email.com" }]

dependencies = [
    "click>=8.1.0",
    "pyyaml>=6.0",
    "python-dotenv>=1.0.0",
    "pydantic>=2.0.0",
    "pydantic-settings>=2.0.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.4.0",
    "pytest-asyncio>=0.21.0",
    "pytest-cov>=4.1.0",
    "ruff>=0.1.0",
    "mypy>=1.5.0",
]

[project.scripts]
kbase = "src.main:main"

[tool.uv]
dev-dependencies = [
    "pytest>=7.4.0",
    "pytest-asyncio>=0.21.0",
    "pytest-cov>=4.1.0",
    "ruff>=0.1.0",
    "mypy>=1.5.0",
]

[tool.ruff]
line-length = 100
target-version = "py310"

[tool.mypy]
python_version = "3.10"
strict = true
```

**UV Benefits:**
- **Speed:** 10-100x faster than pip
- **Lock file:** Automatic `uv.lock` for reproducible builds
- **Dependency groups:** Better than extras for dev/prod separation
- **Script running:** `uv run` handles venv automatically
- **Tool management:** `uv tool install` for CLI tools

#### Step 1.2: Configuration Module
- [ ] Implement `src/core/config.py` - Configuration loader
  - Load YAML configs with environment variable substitution
  - Validate configuration schema with Pydantic
  - Support config reload without restart
- [ ] Implement `src/core/exceptions.py` - Custom exception hierarchy
- [ ] Create configuration validation tests

**Key Classes:**
```python
class ConfigLoader:
    def load_yaml(path: str) -> dict
    def substitute_env_vars(config: dict) -> dict
    def validate(config: dict, schema: Type[BaseModel]) -> BaseModel

class AppConfig(BaseModel):
    # Pydantic models for each config section
```

#### Step 1.3: CLI Foundation
- [ ] Implement `src/cli/commands.py` - CLI framework
  - Interactive REPL mode
  - Command parsing and routing
  - Help system
- [ ] Implement `src/cli/formatters.py` - Output formatting
  - Markdown rendering
  - Table formatting for lists
  - Color support (optional)

**CLI Commands to Implement:**
- `help` - Show available commands
- `config` - Show/edit configuration
- `exit`/`quit` - Exit application

---

### Phase 2: LLM Integration Layer (Week 3)

#### Step 2.1: Abstract LLM Interface
- [ ] Implement `src/llm/base.py` - Abstract base class
  ```python
  class BaseLLM(ABC):
      @abstractmethod
      async def generate(self, prompt: str, **kwargs) -> LLMResponse
      @abstractmethod
      async def stream(self, prompt: str, **kwargs) -> AsyncIterator[str]
      @abstractmethod
      def count_tokens(self, text: str) -> int
  ```

#### Step 2.2: OpenAI-Compatible Implementation
- [ ] Implement `src/llm/openai_compatible.py`
  - OpenAI API client wrapper
  - Streaming support
  - Token counting
  - Error handling and retry logic
  - Rate limiting

**Dependencies (add with `uv add`):**
```bash
uv add openai tenacity tiktoken
```

#### Step 2.3: LLM Factory
- [ ] Implement `src/llm/factory.py`
  - Dynamic provider selection
  - Model instance caching
  - Configuration-based instantiation

**Factory Pattern:**
```python
class LLMFactory:
    _registry: Dict[str, Type[BaseLLM]] = {}
    
    @classmethod
    def register(cls, provider: str, llm_class: Type[BaseLLM])
    
    @classmethod
    def create(cls, provider: str, model: str, config: dict) -> BaseLLM
    
    @classmethod
    def get_available_providers(cls) -> List[str]
```

---

### Phase 3: RAG Module (Week 4-5)

#### Step 3.1: Document Processing
- [ ] Implement `src/rag/document_processor.py`
  - PDF parsing (PyPDF2)
  - TXT file reading
  - Document chunking strategies
  - Metadata extraction

**Dependencies (add with `uv add`):**
```bash
uv add pypdf langchain-text-splitters
```

#### Step 3.2: Embedding Generation
- [ ] Implement `src/rag/embeddings.py`
  - OpenAI embeddings support
  - HuggingFace/Sentence-Transformers support
  - Batch processing
  - Caching layer

**Dependencies (add with `uv add`):**
```bash
uv add sentence-transformers
```

#### Step 3.3: Vector Store
- [ ] Implement `src/rag/vector_store.py`
  - Abstract vector store interface
  - ChromaDB implementation
  - FAISS implementation (optional)
  - CRUD operations for documents
  - Persistence management

**Dependencies (add with `uv add`):**
```bash
uv add chromadb  # Core vector store
uv add faiss-cpu  # Optional, for local FAISS support
```

#### Step 3.4: Retriever
- [ ] Implement `src/rag/retriever.py`
  - Similarity search
  - Hybrid search (keyword + semantic)
  - Re-ranking (optional)
  - Context window optimization

---

### Phase 4: Memory Management (Week 6)

#### Step 4.1: Short-Term Memory
- [ ] Implement `src/memory/short_term.py`
  - Conversation buffer with configurable window
  - Token-aware summarization
  - LangGraph state integration

**Dependencies (add with `uv add`):**
```bash
uv add langchain-core
```

#### Step 4.2: Long-Term Memory
- [ ] Implement `src/memory/models.py`
  - SQLAlchemy models for conversation persistence
- [ ] Implement `src/memory/long_term.py`
  - Conversation CRUD operations
  - Auto-titling with LLM
  - Search functionality
  - Export/import capabilities

**Dependencies (add with `uv add`):**
```bash
uv add sqlalchemy aiosqlite
```

---

### Phase 5: Agentic Workflow Engine (Week 7-8)

#### Step 5.1: Base Agent Framework
- [ ] Implement `src/agents/base.py`
  - Abstract agent class
  - State management interface
  - Tool registration system

#### Step 5.2: Individual Agents
- [ ] Implement `src/agents/router.py` - Query routing
- [ ] Implement `src/agents/web_search.py` - Web search integration
- [ ] Implement `src/agents/rag.py` - RAG agent
- [ ] Implement `src/agents/summarizer.py` - Summarization
- [ ] Implement `src/agents/refiner.py` - Refinement/critique
- [ ] Implement `src/agents/final_answer.py` - Final synthesis

**Dependencies (add with `uv add`):**
```bash
uv add langgraph langchain
```

#### Step 5.3: Workflow Orchestrator
- [ ] Implement `src/agents/workflow.py`
  - LangGraph workflow definition
  - Edge routing logic from config
  - State persistence
  - Human-in-the-loop support (optional)

**Workflow State Schema:**
```python
class WorkflowState(TypedDict):
    query: str
    action: str
    web_results: Optional[List[str]]
    rag_results: Optional[List[str]]
    summary: Optional[str]
    refinement_needed: bool
    final_answer: Optional[str]
    conversation_history: List[dict]
    metadata: dict
```

---

### Phase 6: Observability (Week 9)

#### Step 6.1: Langfuse Integration
- [ ] Implement `src/observability/langfuse_handler.py`
  - Trace initialization
  - Span management
  - Metadata tagging
  - Cost tracking
  - Error logging

**Dependencies (add with `uv add`):**
```bash
uv add langfuse
```

#### Step 6.2: Tracing Utilities
- [ ] Implement `src/observability/tracing.py`
  - Decorator for automatic tracing
  - Context managers for spans
  - Integration with all modules

---

### Phase 7: CLI Commands & Integration (Week 10)

#### Step 7.1: Document Management Commands
- [ ] `ingest --path <path>` - Ingest documents
- [ ] `list-docs` - List indexed documents
- [ ] `remove-doc <doc_id>` - Remove document from index
- [ ] `reindex` - Rebuild entire index

#### Step 7.2: Conversation Management Commands
- [ ] `new-chat` - Start new conversation
- [ ] `list-chats` - List past conversations
- [ ] `load-chat <id>` - Load conversation
- [ ] `rename-chat <title>` - Rename current chat
- [ ] `delete-chat <id>` - Delete conversation
- [ ] `search-chats <query>` - Search conversation history

#### Step 7.3: Query Commands
- [ ] Interactive query mode
- [ ] Multi-line input support
- [ ] Source citation display
- [ ] Response streaming

---

### Phase 8: Testing & Documentation (Week 11-12)

#### Step 8.1: Unit Tests
- [ ] LLM module tests
- [ ] RAG module tests
- [ ] Agent tests with mocking
- [ ] Memory tests
- [ ] Configuration tests

#### Step 8.2: Integration Tests
- [ ] End-to-end workflow tests
- [ ] CLI command tests
- [ ] Configuration hot-reload tests

#### Step 8.3: Documentation
- [ ] API documentation (docstrings)
- [ ] Configuration guide
- [ ] Developer guide
- [ ] User guide

---

## 4. Module Dependencies Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                         CLI Layer                            │
│  (commands.py ← formatters.py)                              │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│                    Application Core                          │
│  (config.py, exceptions.py)                                 │
└────────┬───────────────┬────────────────┬──────────────────┘
         │               │                │
    ┌────▼────┐    ┌─────▼──────┐   ┌────▼─────┐
    │   LLM   │    │ Observability│  │  Memory  │
    │ Module  │    │  (Langfuse) │   │ Module   │
    └────┬────┘    └─────────────┘   └────┬─────┘
         │                                │
    ┌────▼────────────────────────────────▼─────┐
    │          Agentic Workflow Engine          │
    │  (LangGraph + Agents)                     │
    └────┬──────────────────────────────────────┘
         │
    ┌────▼────┐
    │   RAG   │
    │ Module  │
    └─────────┘
```

## 5. Key Interfaces

### 5.1 Configuration Interface

```python
from src.core.config import ConfigManager

# Initialize configuration
config = ConfigManager(
    config_dir="./config",
    env_file=".env"
)

# Access configuration
llm_config = config.get("llm")
app_config = config.get("app")

# Hot reload
config.reload()
```

### 5.2 LLM Interface

```python
from src.llm.factory import LLMFactory

# Create LLM instance
llm = LLMFactory.create(
    provider="openai",
    model="gpt-4",
    config=llm_config
)

# Generate response
response = await llm.generate(
    prompt="Explain quantum computing",
    temperature=0.7,
    max_tokens=1000
)

# Stream response
async for chunk in llm.stream(prompt):
    print(chunk, end="")
```

### 5.3 RAG Interface

```python
from src.rag.retriever import Retriever
from src.rag.document_processor import DocumentProcessor

# Process documents
processor = DocumentProcessor(config.rag)
await processor.ingest_directory("./docs")

# Retrieve relevant documents
retriever = Retriever(config.rag)
results = await retriever.retrieve(
    query="What is machine learning?",
    top_k=5
)
```

### 5.4 Workflow Interface

```python
from src.agents.workflow import WorkflowEngine

# Initialize workflow
engine = WorkflowEngine(
    config=config.agents,
    llm_factory=LLMFactory,
    retriever=retriever,
    memory=memory_manager
)

# Execute query
result = await engine.run(
    query="Compare AI vs ML",
    conversation_id="conv-123"
)
```

## 6. Configuration Best Practices

### 6.1 Environment Variable Substitution

All configuration files support `${VAR_NAME}` and `${VAR_NAME:default}` syntax:
- `${OPENAI_API_KEY}` - Required, will raise error if not set
- `${LLM_ROUTER_MODEL:gpt-3.5-turbo}` - Optional with default value

### 6.2 Configuration Validation

Use Pydantic models for schema validation:

```python
from pydantic import BaseModel, Field
from typing import List, Optional

class LLMProviderModel(BaseModel):
    name: str
    max_tokens: int = Field(gt=0)
    temperature: float = Field(ge=0, le=2)

class LLMProviderConfig(BaseModel):
    api_base: str
    api_key: str
    models: List[LLMProviderModel]

class LLMConfig(BaseModel):
    default_provider: str
    default_model: str
    providers: Dict[str, LLMProviderConfig]
    retry: RetryConfig
    rate_limit: RateLimitConfig
```

### 6.3 Configuration Hot Reload

```python
class ConfigManager:
    def __init__(self, config_dir: str, watch: bool = True):
        self._watcher = ConfigWatcher(config_dir, self.reload)
        if watch:
            self._watcher.start()
    
    def reload(self):
        # Reload all configuration files
        # Notify subscribers of changes
        pass
    
    def on_change(self, callback: Callable):
        # Register callback for config changes
        pass
```

## 7. Extension Points

### 7.1 Adding New LLM Provider

```python
# src/llm/custom_provider.py
from src.llm.base import BaseLLM

class CustomLLM(BaseLLM):
    async def generate(self, prompt: str, **kwargs) -> LLMResponse:
        # Implementation
        pass

# Register in factory
from src.llm.factory import LLMFactory
LLMFactory.register("custom", CustomLLM)
```

### 7.2 Adding New Agent

```python
# src/agents/custom_agent.py
from src.agents.base import BaseAgent

class CustomAgent(BaseAgent):
    async def execute(self, state: WorkflowState) -> WorkflowState:
        # Implementation
        return state

# Add to config/agents.yaml
```

### 7.3 Adding New Document Type

```python
# src/rag/document_processor.py
class DocumentProcessor:
    def _get_parser(self, file_type: str):
        parsers = {
            ".pdf": self._parse_pdf,
            ".txt": self._parse_txt,
            ".docx": self._parse_docx,  # New type
        }
        return parsers.get(file_type)
```

## 8. Development Timeline

| Phase | Duration | Deliverables |
|-------|----------|--------------|
| 1. Foundation | 2 weeks | Project structure, config module, CLI foundation |
| 2. LLM Layer | 1 week | LLM abstraction, OpenAI implementation, factory |
| 3. RAG Module | 2 weeks | Document processing, embeddings, vector store |
| 4. Memory | 1 week | Short-term and long-term memory |
| 5. Agents | 2 weeks | All agents and workflow orchestrator |
| 6. Observability | 1 week | Langfuse integration |
| 7. CLI Integration | 1 week | All CLI commands |
| 8. Testing & Docs | 2 weeks | Tests and documentation |

**Total: 12 weeks**

## 9. Risk Mitigation

| Risk | Mitigation |
|------|------------|
| LLM API rate limits | Implement robust retry with exponential backoff |
| Large document processing | Background indexing with progress tracking |
| Token limit exceeded | Automatic context summarization |
| Configuration errors | Schema validation with helpful error messages |
| Vector DB corruption | Regular backups and integrity checks |

## 10. Success Criteria

- [ ] All configuration externalized and validated
- [ ] LLM provider swappable without code changes
- [ ] Agent workflow configurable via YAML
- [ ] Document storage path configurable
- [ ] Full observability with Langfuse
- [ ] Conversation persistence working
- [ ] All CLI commands functional
- [ ] >80% test coverage
- [ ] Documentation complete

---

## 11. Complete UV Dependency Reference

### 11.1 All Dependencies by Phase

**Phase 1 - Foundation:**
```bash
uv add click pyyaml python-dotenv pydantic pydantic-settings
uv add --dev pytest pytest-asyncio pytest-cov ruff myky
```

**Phase 2 - LLM Layer:**
```bash
uv add openai tenacity tiktoken
```

**Phase 3 - RAG Module:**
```bash
uv add pypdf langchain-text-splitters sentence-transformers chromadb
uv add faiss-cpu  # Optional
```

**Phase 4 - Memory:**
```bash
uv add langchain-core sqlalchemy aiosqlite
```

**Phase 5 - Agents:**
```bash
uv add langgraph langchain
```

**Phase 6 - Observability:**
```bash
uv add langfuse
```

### 11.2 Complete pyproject.toml Example

```toml
[project]
name = "kbase"
version = "0.1.0"
description = "LLM-Powered Knowledge Exploration Application"
readme = "README.md"
requires-python = ">=3.10"
license = { text = "MIT" }
authors = [{ name = "Your Name", email = "your@email.com" }]

dependencies = [
    # CLI & Config (Phase 1)
    "click>=8.1.0",
    "pyyaml>=6.0",
    "python-dotenv>=1.0.0",
    "pydantic>=2.0.0",
    "pydantic-settings>=2.0.0",
    
    # LLM (Phase 2)
    "openai>=1.0.0",
    "tenacity>=8.0.0",
    "tiktoken>=0.5.0",
    
    # RAG (Phase 3)
    "pypdf>=3.0.0",
    "langchain-text-splitters>=0.0.1",
    "sentence-transformers>=2.0.0",
    "chromadb>=0.4.0",
    # "faiss-cpu>=1.7.0",  # Optional
    
    # Memory (Phase 4)
    "langchain-core>=0.1.0",
    "sqlalchemy>=2.0.0",
    "aiosqlite>=0.19.0",
    
    # Agents (Phase 5)
    "langgraph>=0.0.20",
    "langchain>=0.1.0",
    
    # Observability (Phase 6)
    "langfuse>=2.0.0",
]

[tool.uv]
dev-dependencies = [
    "pytest>=7.4.0",
    "pytest-asyncio>=0.21.0",
    "pytest-cov>=4.1.0",
    "ruff>=0.1.0",
    "mypy>=1.5.0",
]

[project.scripts]
kbase = "src.main:main"

[tool.ruff]
line-length = 100
target-version = "py310"

[tool.mypy]
python_version = "3.10"
strict = true

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
```

### 11.3 Common UV Commands

```bash
# Project initialization
uv init kbase
cd kbase

# Create/activate virtual environment
uv venv
source .venv/bin/activate

# Add dependencies
uv add <package>              # Add to main dependencies
uv add --dev <package>        # Add to dev dependencies
uv add --optional <group> <package>  # Add to optional group

# Install dependencies
uv sync                       # Install all dependencies from lock file
uv sync --dev                 # Install including dev dependencies

# Run commands
uv run python src/main.py     # Run Python script
uv run pytest                 # Run tests
uv run kbase                  # Run installed CLI command

# Update dependencies
uv lock --upgrade             # Update lock file
uv sync --upgrade             # Update and sync

# Remove dependencies
uv remove <package>

# Show dependency tree
uv tree

# Build/publish
uv build                      # Build distributions
uv publish                    # Publish to PyPI
```

### 11.4 UV vs pip Comparison

| Feature | pip | UV |
|---------|-----|-----|
| Install speed | Baseline | 10-100x faster |
| Lock file | Manual (pip-tools) | Automatic (`uv.lock`) |
| Virtual env | Separate tool | Built-in (`uv venv`) |
| Script running | Manual venv activation | `uv run` auto-handles |
| Dependency groups | extras (limited) | Native groups support |
| Tool installation | pip install (global) | `uv tool install` (isolated) |

### 11.5 Development Workflow

```bash
# Day-to-day development
uv run python src/main.py     # Run application
uv run pytest                 # Run tests
uv run ruff check .           # Lint code
uv run mypy src               # Type check

# Adding a new feature
uv add new-package            # Add dependency
uv run python -c "import new_package"  # Verify install

# Before committing
uv sync --dev                 # Ensure all deps installed
uv run pytest --cov          # Run tests with coverage
uv run ruff check --fix      # Lint and fix
```
