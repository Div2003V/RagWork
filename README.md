# RagWork
basic rag...
# Flexible Local RAG Project

A fully local, privacy-focused Retrieval-Augmented Generation (RAG) system built with **LangChain**, **FAISS**, and **Ollama**. This project allows you to "chat" with your documents (PDFs, TXT) without any data leaving your device.


## Key Features
- **Privacy First:** 100% local execution. No API keys, no internet required after setup.
- **Flexible Data Extraction:** Automatically handles PDFs and Text files within a directory.
- **History-Aware:** Remembers previous parts of the conversation for a true chat experience.
- **State-of-the-Art Tools:** Uses Llama 3.2 (LLM) and Nomic-Embed-Text (Embeddings).

---


## Tool Stack
| Component | Technology |
| :--- | :--- |
| **LLM** | Llama 3.2 (via Ollama) |
| **Embeddings** | Nomic-Embed-Text (via Ollama) |
| **Orchestration** | LangChain |
| **Vector Database** | FAISS |
| **Environment** | Python 3.10+ |

---


## Getting Started

### 


1. Prerequisites
Install [Ollama](https://ollama.com/) and download the necessary models:
```bash
Ollama pull llama3.2
Ollama pull nomic-embed-text


2. Installation
Clone this repository and set up your Python environment:

# Create virtual environment
python -m venv venv

# Activate (Windows)
.\venv\Scripts\activate

# Install dependencies
python -m pip install --upgrade pip
python -m pip install langchain-ollama langchain-community langchain-text-splitters faiss-cpu pypdf docx2txt


3. Usage
Prepare Documents: Place your PDFs or Text files in the /docs folder.

Ingest Data: Run the retrieval pipeline to create your local vector database.
python retrieval_pipeline.py

Start Chatting: Launch the history-aware chat interface.
python answer_generator.py


How it Works
Ingestion: Documents are loaded and split into 1000-character chunks.

Embedding: nomic-embed-text converts text into high-dimensional vectors.

Storage: Vectors are stored in a local faiss_index folder.

Chatting: When you ask a question, the system retrieves the 3 most relevant chunks and "stuffs" them into the prompt for Llama 3.2.

Contextualization: The system uses chat history to rewrite follow-up questions into standalone queries.


Important Notes
Strictness: The model is currently configured as a "Helpful Assistant." To make it strictly stick to your PDF, modify the system_prompt in answer_generator.py.

Hardware: Performance depends on your local CPU/GPU. Ensure you have at least 8GB of RAM for a smooth experience.

Updating Docs: If you add new files to the /docs folder, you must re-run retrieval_pipeline.py to update the database.

