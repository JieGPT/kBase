# kBase - LLM-Powered Knowledge Exploration

A command-line interface (CLI) application for intelligent document querying using Retrieval-Augmented Generation (RAG). kBase allows you to index your local documents (PDF, TXT) and query them using natural language with the power of Large Language Models.

## Features

- **Document Ingestion**: Automatically process and index PDF and TXT files
- **Semantic Search**: Vector-based retrieval using ChromaDB and OpenAI embeddings
- **Conversation Memory**: Maintains context across multiple queries
- **Streaming Responses**: Real-time response generation from LLM
- **Configurable**: Environment variables and YAML configuration support
- **OpenAI-Compatible**: Works with OpenAI API and compatible endpoints (Ollama, etc.)

## Quick Start

### Prerequisites

- Python 3.10+
- [UV](https://github.com/astral-sh/uv) package manager
- OpenAI API key (or compatible API)

### Installation

1. **Clone the repository**:
```bash
git clone <repository-url>
cd kBase
```

2. **Install dependencies**:
```bash
uv sync
```

3. **Configure environment variables**:
```bash
cp .env.example .env
# Edit .env and add your API key
```

4. **Run the application**:
```bash
uv run python src/main.py
```

## Configuration

### Environment Variables

Create a `.env` file in the project root with the following variables:

```env
# Required: API Configuration
API_KEY=sk-your-api-key-here          # Or OPENAI_API_KEY
BASE_URL=https://api.openai.com/v1    # For OpenAI-compatible endpoints

# Optional: Model Configuration
MODEL=gpt-4o-mini                     # LLM model
TEMPERATURE=0.7                       # Response creativity (0-2)
MAX_TOKENS=2048                       # Max response length
EMBEDDING_MODEL=text-embedding-3-small

# Optional: Retrieval Configuration
TOP_K=5                               # Number of documents to retrieve

# Optional: Application Settings
DEBUG=false
```

### Configuration File

Edit `config/config.yaml` for persistent settings:

```yaml
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
```

## Usage

### Starting the Application

```bash
uv run python src/main.py
```

### Available Commands

Once in the CLI, you can use the following commands:

#### `help`
Show available commands and usage information.

```
>>> help

Available commands:
  help          Show this help message
  exit, quit    Exit the application
  ingest [path] Ingest documents from path
  list-docs     List all indexed documents
  clear         Clear conversation history
  <query>       Ask a question
```

#### `ingest [path]`
Process and index documents from a directory.

```
>>> ingest ./docs
Processing documents from: ./docs
Found 50 chunks. Generating embeddings...
  Processed 50/50 chunks
Done! Indexed 50 chunks.
```

If no path is provided, it uses the default from `config.yaml` (default: `./docs`).

#### `list-docs`
Display all indexed documents.

```
>>> list-docs
Total indexed chunks: 50

Documents:
  - report.pdf
  - notes.txt
  - manual.pdf
```

#### `clear`
Clear the current conversation history.

```
>>> clear
Conversation cleared.
```

#### `exit` or `quit`
Exit the application.

```
>>> exit
Goodbye!
```

### Querying Documents

Simply type your question to query the indexed documents:

```
>>> What is machine learning?
Assistant: Based on the documents, machine learning is a subset of artificial intelligence
that enables computers to learn and improve from experience without being explicitly programmed...

>>> How does it relate to neural networks?
Assistant: Neural networks are a specific approach to machine learning inspired by the
structure and function of biological neural networks...
```

The system maintains conversation context, so follow-up questions work naturally.

## Supported Document Types

- **PDF** (.pdf): Extracts text content from all pages
- **Text** (.txt): Plain text files

Documents are automatically:
1. Parsed and extracted
2. Split into overlapping chunks (default: 1000 chars with 200 char overlap)
3. Embedded using OpenAI embeddings
4. Stored in ChromaDB vector database

## Project Structure

```
kBase/
├── src/
│   ├── cli/
│   │   └── commands.py          # CLI interface and command handling
│   ├── core/
│   │   └── config.py            # Configuration management
│   ├── llm/
│   │   ├── base.py              # LLM interface
│   │   └── openai_client.py     # OpenAI-compatible client
│   ├── rag/
│   │   ├── document_processor.py # Document parsing and chunking
│   │   ├── embeddings.py         # Embedding generation
│   │   └── vector_store.py       # ChromaDB vector store
│   ├── memory/
│   │   └── conversation_buffer.py # Conversation memory
│   └── main.py                   # Application entry point
├── config/
│   ├── config.yaml              # Main configuration
│   └── .env                     # Environment variables (gitignored)
├── docs/                        # Document storage folder
├── data/                        # Vector database persistence
├── tests/                       # Test suite
├── pyproject.toml              # Project dependencies
└── README.md                   # This file
```

## Development

### Running Tests

```bash
# Run all tests
uv run pytest tests/ -v

# Run with coverage
uv run pytest --cov=src tests/
```

### Code Quality

```bash
# Lint code
uv run ruff check src/

# Fix linting issues
uv run ruff check src/ --fix
```

### Adding New Features

The codebase is modular and extensible:

- **New LLM Providers**: Implement `BaseLLM` interface in `src/llm/`
- **New Document Types**: Extend `DocumentProcessor` in `src/rag/document_processor.py`
- **New Vector Stores**: Implement vector store interface in `src/rag/`

## Using with Local Models (Ollama)

kBase supports any OpenAI-compatible API, including local models via Ollama:

1. **Install and run Ollama**:
```bash
# Install Ollama from https://ollama.ai
ollama pull llama3.2
ollama serve
```

2. **Configure kBase**:
```env
BASE_URL=http://localhost:11434/v1
API_KEY=ollama
MODEL=llama3.2
```

3. **Run kBase**:
```bash
uv run python src/main.py
```

## Troubleshooting

### "API key is required" Error
Make sure you've set the `API_KEY` or `OPENAI_API_KEY` environment variable in your `.env` file.

### No Documents Found
Ensure documents are in the `./docs` folder (or your configured path) and have `.pdf` or `.txt` extensions.

### Empty Responses
Check that documents were successfully ingested with `list-docs` command.

## License

MIT License

## Future Enhancements

- Hybrid retrieval (semantic + lexical search with Meilisearch)
- Re-ranking for improved retrieval quality
- Persistent conversation storage
- Web search integration
- Multiple LLM provider support
- Langfuse observability integration

See `MVP_Implementation_Plan.md` and `Implementation_Plan.md` for detailed roadmap.
