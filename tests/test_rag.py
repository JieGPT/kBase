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
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
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
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
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
    assert "Python" in results[0]["content"]
