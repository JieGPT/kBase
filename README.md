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
