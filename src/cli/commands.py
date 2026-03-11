import asyncio

import click
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from src.llm.openai_client import OpenAIClient
from src.rag.embeddings import EmbeddingGenerator
from src.rag.vector_store import VectorStore
from src.memory.conversation_buffer import ConversationBuffer
from src.rag.document_processor import DocumentProcessor


class CLI:
    def __init__(self, config):
        self.config = config
        self.session = PromptSession(history=FileHistory(".kbase_history"))
        self.running = True

        # Initialize components
        self.llm = OpenAIClient(
            model=config.config.llm.model,
            temperature=config.config.llm.temperature,
            max_tokens=config.config.llm.max_tokens,
            base_url=config.config.llm.base_url,
            api_key=config.config.llm.api_key,
        )
        self.embedder = EmbeddingGenerator(
            model=config.config.embedding.model,
            base_url=config.config.embedding.base_url,
            api_key=config.config.embedding.api_key,
        )
        self.vector_store = VectorStore(
            persist_path=config.config.vector_db.persist_path,
            collection_name=config.config.vector_db.collection_name,
        )
        self.memory = ConversationBuffer(max_turns=config.config.memory.max_turns)
        self.processor = DocumentProcessor(
            chunk_size=config.config.chunking.chunk_size,
            chunk_overlap=config.config.chunking.chunk_overlap,
        )

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
            asyncio.run(self.process_query(user_input))

    def show_help(self):
        help_text = """
Available commands:
  help          Show this help message
  exit, quit    Exit the application
  ingest [path] Ingest documents from path (uses config default if not specified)
  list-docs     List all indexed documents
  clear         Clear conversation history
  <query>       Ask a question (any other input)
"""
        click.echo(help_text)

    async def process_query(self, query: str):
        self.memory.add_user_message(query)

        # Retrieve relevant documents
        query_embedding = self.embedder.embed_query(query)
        results = self.vector_store.search(
            query_embedding, top_k=self.config.config.retrieval.top_k
        )

        # Build context
        if results:
            context = "\n\n".join(
                [
                    f"[{r['metadata'].get('source', 'Unknown')}, Page {r['metadata'].get('page', 'N/A')}]\n{r['content']}"
                    for r in results
                ]
            )
        else:
            context = "No relevant documents found."

        # Build prompt
        conversation_context = self.memory.get_context_string()
        prompt = f"""Based on the following documents, answer the user's question.
If the answer is not in the documents, say so.

Documents:\n{context}\n\nConversation History:\n{conversation_context}\n\nQuestion: {query}\n\nAnswer:"""

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
            doc_path, extensions=self.config.config.document_storage.supported_formats
        )

        if not documents:
            click.echo("No documents found.")
            return

        click.echo(f"Found {len(documents)} chunks. Generating embeddings...")

        # Generate embeddings in batches
        batch_size = 100
        for i in range(0, len(documents), batch_size):
            batch = documents[i : i + batch_size]
            embeddings = self.embedder.embed_texts([doc.content for doc in batch])
            self.vector_store.add_documents(batch, embeddings)
            click.echo(f"  Processed {min(i + batch_size, len(documents))}/{len(documents)} chunks")

        click.echo(f"Done! Indexed {len(documents)} chunks.")

    def list_documents(self):
        count = self.vector_store.count()
        click.echo(f"Total indexed chunks: {count}")

        # Get unique sources
        results = self.vector_store.collection.get(include=["metadatas"])
        sources = set()
        for meta in results.get("metadatas") or []:
            if meta and "source" in meta:
                sources.add(meta["source"])

        if sources:
            click.echo("\nDocuments:")
            for source in sorted(sources):
                click.echo(f"  - {source}")
