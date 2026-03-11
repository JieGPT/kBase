# Functional Specification: LLM-Powered Knowledge Exploration Application (CLI, Python Stack, Langfuse & Memory Management, with Meilisearch)

## 1. Introduction

This document outlines the functional requirements for an LLM-powered knowledge exploration application. The application aims to provide users with a robust platform for querying information, leveraging various Large Language Models (LLMs) through an OpenAI-compatible API, and employing agentic workflows for advanced tasks such as web searching, Retrieval Augmented Generation (RAG) from external documents, and intelligent summarization. The application will feature a Command Line Interface (CLI) and be built with a Python-centric technical stack, integrating Langfuse for observability and comprehensive memory management for both single and multiple conversations. **Meilisearch will be used to power efficient and fast keyword-based retrieval of document chunks from local files.**

## 2. Goals

*   To enable efficient and comprehensive knowledge exploration across diverse sources.
*   To provide flexibility in LLM choice via an OpenAI-compatible API.
*   To leverage agentic workflows for complex information retrieval and processing.
*   To support RAG from various document types (PDF, TXT) stored in a specified local folder.
*   To deliver concise and accurate summaries of key information.
*   To enhance user productivity and decision-making through intelligent information synthesis.
*   To provide deep observability, debugging, and evaluation for agentic workflows using Langfuse.
*   To maintain context and coherence within a single conversation session.
*   To manage and retrieve past conversations, enabling users to revisit and continue previous discussions.
*   **To enable fast and relevant keyword-based search within ingested local documents using Meilisearch.**

## 3. Core Features

### 3.1. User Interface (CLI)

The Command Line Interface (CLI) will be the primary interaction point for users.

*   **Query Input:**
    *   A clear command-line prompt for users to submit their queries.
    *   Support for multi-line input for complex questions (e.g., using a text editor or specific input commands).
*   **Response Display:**
    *   Formatted text output for LLM responses, including clear demarcation of different sections (e.g., "Web Search Results:", "RAG Documents:", "Summary:", "Final Answer:").
    *   Clear indication of the source of information (e.g., `[Web Search]`, `[Document: filename.pdf]`, `[LLM Synthesis]`).
    *   (Optional) Pagination for long responses.
*   **Document Management Commands:**
    *   Commands to trigger document ingestion from a specified folder (e.g., `ingest --path /path/to/docs`). This command will initiate the parsing, chunking, embedding, and indexing process for both the vector database and Meilisearch.
    *   Commands to list currently processed documents (e.g., `list-docs`).
    *   (Optional) Commands to remove documents from the RAG index and Meilisearch.
    *   **Command to re-index documents in Meilisearch (e.g., `reindex-docs`). This would rebuild the Meilisearch index from the currently ingested documents, useful for configuration changes or troubleshooting.**
*   **LLM Configuration Commands:**
    *   Commands to configure and select different OpenAI-compatible LLM endpoints (e.g., `config llm --provider openai --model gpt-4`).
    *   Commands to set API keys and other parameters.
*   **Conversation Management Commands:**
    *   Command to start a new conversation (e.g., `new-chat`).
    *   Command to list past conversations (e.g., `list-chats`).
    *   Command to load a specific past conversation by ID or title (e.g., `load-chat <id_or_title>`).
    *   Command to rename the current conversation (e.g., `rename-chat "New Title"`).
    *   Command to delete a conversation (e.g., `delete-chat <id>`).
    *   Clear display of the current conversation's history within the CLI session.

### 3.2. LLM Integration Layer

This layer will manage communication with various LLMs.

*   **OpenAI-Compatible API Wrapper:**
    *   A standardized interface to interact with any LLM that exposes an OpenAI-compatible API (e.g., OpenAI GPT models, custom fine-tuned models, local models via `ollama`, `vLLM`, etc.).
    *   Handles API key management, request serialization, and response deserialization.
*   **Model Selection:**
    *   Mechanism to dynamically select which LLM to use based on user configuration or agent directives.
*   **Error Handling & Retry Mechanisms:**
    *   Robust handling of API errors, rate limits, and network issues with appropriate retry logic.

### 3.3. Agentic Workflow Engine (LangGraph Powered)

This is the brain of your application, orchestrating complex tasks.

*   **Core Agent Orchestration:**
    *   Utilize LangGraph to define and manage stateful, multi-turn agentic workflows.
    *   Define different "nodes" for specific tasks (e.g., "Web Search," "RAG Retrieval," "Summarization," "Final Answer Generation").
    *   Define "edges" to dictate the flow between nodes based on intermediate results or conditions.
*   **Agent Roles/Capabilities:**
    *   **Router Agent:** Analyzes the initial query and determines the best course of action (e.g., direct LLM answer, web search, RAG, or a combination).
    *   **Web Search Agent:**
        *   Formulates search queries based on user input or intermediate agent thoughts.
        *   Executes web searches using a chosen search API (e.g., Google Search API, Brave Search API, custom scrapers).
        *   Parses and extracts relevant information from search results.
    *   **RAG Agent (Retrieval Augmented Generation):**
        *   **Document Ingestion (Detailed Meilisearch Integration):**
            *   Monitors a **specified local folder** (e.g., `docs/`) for new or updated PDF and TXT files.
            *   When a new or updated document is detected:
                1.  **Parsing:** Parses and extracts raw text content from the document (e.g., using `PyPDF2`, `pypdf`).
                2.  **Chunking:** Splits the extracted text into manageable, overlapping chunks. Each chunk will be assigned a unique ID, source filename, and page number/offset.
                3.  **Embedding:** Generates embeddings for each text chunk using a chosen embedding model. These embeddings are stored in the vector database.
                4.  **Meilisearch Indexing:**
                    *   For each chunk, a document is prepared for Meilisearch. This document will typically contain:
                        *   `id`: A unique identifier for the chunk (e.g., `filename_page_chunkindex`).
                        *   `content`: The actual text of the chunk.
                        *   `source_file`: The original filename (e.g., `report.pdf`).
                        *   `page_number`: The page number from which the chunk was extracted (if applicable).
                        *   (Optional) `metadata`: Any other relevant metadata (e.g., section title, author).
                    *   These prepared documents are then sent to the local Meilisearch instance for indexing. Meilisearch will automatically handle tokenization, stemming, and create its inverted index for fast keyword search.
            *   **Deletion/Update Handling:** When a document is removed or significantly altered in the local folder, the corresponding chunks and their entries in both the vector database and Meilisearch will be updated or deleted.
        *   **Retrieval (Detailed Meilisearch Integration):**
            *   When a user query is received:
                1.  **Semantic Search (Vector Database):** Generates embeddings for the user query and performs a similarity search against the vector database to retrieve `N` semantically relevant document chunks.
                2.  **Lexical Search (Meilisearch):** Performs a keyword search against the Meilisearch index using the user query. Meilisearch will return `M` document chunks that lexically match the query, ordered by relevance (Meilisearch's internal ranking).
                3.  **Hybrid Retrieval:**
                    *   Combines the results from both semantic and lexical searches.
                    *   Strategies for combination could include:
                        *   **Simple Union:** Take all unique chunks from both sets.
                        *   **Weighted Fusion:** Assign scores to chunks from both methods and combine them (e.g., using Reciprocal Rank Fusion - RRF).
                        *   **Re-ranking:** Use an LLM or a separate re-ranker model to score the combined set of chunks based on their relevance to the original query.
                    *   The goal is to get a comprehensive set of highly relevant chunks, leveraging the strengths of both retrieval methods.
        *   **Augmentation:**
            *   Passes the selected and potentially re-ranked retrieved chunks along with the original query and conversation history to an LLM for contextualized answer generation. The LLM will be instructed to synthesize information from these chunks.
    *   **Summarization Agent:**
        *   Identifies key information from web search results, retrieved documents, or LLM-generated text.
        *   Generates concise and accurate summaries at various levels of detail (e.g., bullet points, short paragraph, executive summary).
    *   **Refinement/Critique Agent (Optional but powerful):**
        *   Evaluates intermediate answers or search results.
        *   Identifies gaps, inconsistencies, or areas for further exploration.
        *   Guides the workflow to perform additional searches or RAG queries (including potentially more targeted Meilisearch queries).
    *   **Final Answer Generation Agent:**
        *   Synthesizes information from all sources (LLM, web, RAG) into a coherent and comprehensive final answer.
        *   Ensures the answer directly addresses the user's query.

### 3.4. Observability and Evaluation (Langfuse Integration)

This crucial component will provide insights into the performance and behavior of your LLM application.

*   **Trace Collection:**
    *   Automatically capture and send traces of all LLM calls, agent steps, and tool invocations to Langfuse.
    *   This includes input prompts, LLM responses, token usage, latency, and metadata for each step.
*   **Span Management:**
    *   Organize traces into hierarchical spans, representing individual operations within a larger workflow (e.g., a "Web Search" span, a "RAG Retrieval" span, an "LLM Call" span, **"Meilisearch Indexing" span, "Meilisearch Search" span**).
*   **Input/Output Logging:**
    *   Log the inputs and outputs of each agent step and LLM call for detailed debugging.
*   **Metadata Tagging:**
    *   Attach custom metadata (e.g., user ID, session ID, query type, document source, conversation ID, **Meilisearch query, Meilisearch hits/ranking, Meilisearch indexing status**) to traces and spans for filtering and analysis in Langfuse.
*   **Error Logging:**
    *   Automatically log any errors or exceptions that occur during agent execution or LLM calls to Langfuse, making it easier to identify and debug issues.
*   **Evaluation Hooks:**
    *   Provide hooks to integrate with Langfuse's evaluation capabilities, allowing for:
        *   **Manual Annotations:** Users or developers can manually label traces (e.g., "correct," "incorrect," "hallucination").
        *   **Automated Metrics:** Potentially integrate with automated evaluation metrics (e.g., RAGAS for RAG quality) by feeding data from Langfuse traces.
        *   **Dataset Creation:** Facilitate the creation of evaluation datasets from production traces.
*   **Cost Tracking:**
    *   Leverage Langfuse to track token usage and estimated costs associated with LLM calls, providing insights into operational expenses.

### 3.5. Memory Management

This section details how the application will maintain conversational context.

*   **3.5.1. Short-Term Memory (Single Conversation Context)**
    *   **Conversation Buffer:** Store a limited number of recent turns (user query + system response) for the current active conversation. This buffer will be passed to the LLM for context in subsequent turns.
    *   **Context Summarization (Optional but Recommended):** For longer conversations, employ an LLM to periodically summarize the conversation history, reducing the token count while retaining key information. This summary is then used as part of the prompt for new turns.
    *   **Session Management:** Each active conversation will have a unique session ID, allowing the backend to associate incoming queries with the correct ongoing discussion.
    *   **State Management:** LangGraph's state management will be utilized to pass relevant information (e.g., search results, retrieved documents, intermediate thoughts) between agent nodes within a single turn, and potentially persist key elements across turns if needed for complex multi-step reasoning.

*   **3.5.2. Long-Term Memory (Multiple Conversation Management)**
    *   **Conversation Persistence:**
        *   Store entire conversation histories (user queries and system responses) in a persistent database.
        *   Each conversation record will include a unique ID, user ID (if applicable for multi-user CLI), creation timestamp, last updated timestamp, and a (potentially auto-generated) title.
    *   **User-Specific Storage:** Conversations will be securely linked to individual user accounts (if implemented) or local profiles, ensuring privacy and personalized access.
    *   **Conversation Retrieval:**
        *   When a user selects a past conversation via a CLI command, the full history will be loaded from the database.
        *   The loaded history will be used to reconstruct the context for the LLM, allowing the user to seamlessly continue where they left off.
    *   **Auto-Titling:** An LLM agent will automatically generate a concise, descriptive title for new conversations based on the initial few turns. This title can be edited by the user via a CLI command.
    *   **Searchable History (Optional):** Allow users to search their past conversations by keywords within the messages or by conversation titles via CLI commands. This might involve indexing conversation content.
    *   **Archiving/Deletion:** Provide users with the ability to archive or permanently delete old conversations via CLI commands.

## 4. Technical Requirements

### 4.1. Architecture

*   **Application Core:** Python (e.g., `asyncio` for concurrent operations).
*   **CLI Framework:** Python (e.g., `Click`, `Typer`, `argparse`) for robust command-line interface development.
*   **LLM Orchestration:** `LangGraph` for defining and managing agentic workflows.
*   **LLM Integration:** `LangChain` (or similar libraries) for standardized LLM API calls and prompt management, compatible with OpenAI API.
*   **Embedding Models:** Python libraries for embedding generation (e.g., `HuggingFace Transformers`, `Sentence Transformers`).
*   **Vector Database:** A Python-native or easily integratable vector database (e.g., `ChromaDB`, `FAISS`, `Qdrant` client, `Weaviate` client) for storing and querying document embeddings. This database should be able to persist data locally.
*   **Lexical Search Engine:** **Meilisearch**.
    *   **Deployment:** Runs as a separate, lightweight process locally (e.g., started by the application or assumed to be running). It stores its data in a specified local directory.
    *   **Python Client:** `meilisearch-python-sdk` for interacting with the Meilisearch instance for indexing and searching.
*   **Search API Client:** Python client libraries for chosen web search APIs (e.g., `google-api-python-client`, `serpapi-python`).
*   **Document Parsers:** Python libraries for handling PDF and TXT parsing (e.g., `PyPDF2`, `pdfminer.six`, `pypdf`, `python-docx` if DOCX is added later).
*   **Langfuse SDK:** `langfuse-python` for observability and evaluation.
*   **Persistent Database (for Memory):** A lightweight, embedded, or easily deployable database for conversation histories and metadata (e.g., `SQLite` for local persistence, or a client for `PostgreSQL`/`MongoDB` if a separate server is desired for multi-user).
*   **Document Storage:** A **specified local folder** (e.g., `docs/`) where users place PDF and TXT files for RAG ingestion. The application will monitor this folder.

### 4.2. Performance & Scalability

*   Ability to handle sequential user queries efficiently within a CLI session.
*   Efficient processing of documents for RAG, possibly with background indexing.
*   Optimized LLM calls to minimize latency and cost.
*   Minimal overhead from Langfuse instrumentation on application performance.
*   Efficient storage and retrieval of conversation histories from the chosen persistent database.
*   Optimized context management strategies to balance coherence with token limits and latency.
*   **Fast indexing and search performance from Meilisearch for local documents, ensuring quick retrieval of relevant chunks.**

### 4.3. Security & Privacy

*   Secure handling of API keys (e.g., environment variables, configuration files).
*   Clear policies on data retention for uploaded documents and conversation histories.
*   Secure transmission and storage of trace data to Langfuse, ensuring no sensitive user data is inadvertently logged unless explicitly configured.
*   Local storage of documents and conversation history should be clearly communicated to the user, with options for secure deletion.
*   **Meilisearch instance should be configured securely, especially if exposed beyond the local machine (though for this CLI app, local-only access is expected). Access keys for Meilisearch should be managed securely.**

### 4.4. Extensibility

*   Modular Python codebase to easily integrate new LLMs, search APIs, or document types.
*   Configurable agent workflows to adapt to different use cases.
*   Easy configuration of Langfuse logging levels and data anonymization if required.
*   Flexible memory backend integration (e.g., different database types).
*   **Ability to swap out Meilisearch for other lexical search engines if needed in the future.**

## 5. Non-Functional Requirements

*   **Usability:** Intuitive command-line interface with clear instructions and helpful error messages.
*   **Reliability:** High stability and consistent performance.
*   **Maintainability:** Well-documented Python code, easy to update and troubleshoot.
*   **Performance:** Fast response times for queries and document processing.
*   **Portability:** Easily runnable across different operating systems where Python is supported.
*   **Observability:** Comprehensive logging and tracing provided by Langfuse for debugging and performance monitoring.
*   **Contextual Awareness:** The application should demonstrate an understanding of the ongoing conversation and past interactions.

## 6. Future Enhancements (Optional)

*   **Multi-modal input/output:** Support for image, audio queries (via CLI commands or file inputs).
*   **User Feedback Mechanism:** Allow users to rate answers and provide feedback via CLI commands, which can be sent to Langfuse for evaluation.
*   **Advanced Document Types:** Support for DOCX, CSV, HTML, etc., from the specified folder.
*   **Knowledge Graph Integration:** Build or query a knowledge graph for structured information.
*   **Custom Agent Creation:** Allow advanced users to define their own agent workflows via configuration files.
*   **Cost Monitoring:** Track LLM API usage and costs (enhanced by Langfuse's capabilities), potentially displayed in the CLI.
*   **A/B Testing:** Use Langfuse to compare different agent strategies or LLM configurations.
*   **Personalization:** Leverage long-term memory to tailor responses and proactively offer relevant information based on user history and preferences.
*   **Cross-Conversation Learning:** (Advanced) Develop mechanisms for the system to learn general patterns or preferences from multiple conversations to improve future interactions.
*   **Configuration File Management:** Use a configuration file (e.g., YAML, TOML) for easier management of LLM settings, API keys, and document paths.
*   **Hybrid Retrieval Tuning:** Implement advanced strategies for combining semantic (vector) and lexical (Meilisearch) search results, such as Reciprocal Rank Fusion (RRF), to optimize retrieval relevance.
*   **Meilisearch Configuration:** Allow CLI commands or configuration files to customize Meilisearch settings (e.g., searchable attributes, sortable attributes, stop words, synonyms).