# RAG Document Q&A System

A simple but powerful Retrieval-Augmented Generation (RAG) system built with FastAPI, LangChain, and ChromaDB. Upload documents and ask questions about their content with cited sources.

## Features

- **Document Upload**: Support for PDF, DOCX, TXT, and MD files
- **Smart Chunking**: Intelligent text splitting for optimal retrieval
- **Vector Search**: Semantic search using sentence transformers
- **Source Attribution**: Get answers with document citations
- **Persistent Storage**: ChromaDB vector database with persistence

## Quick Start

### Prerequisites

- Python 3.13+
- Ollama with any compatible model (e.g., `llama3.2`, `mistral`, `qwen2.5`)

### Installation

1. Clone the repository
```bash
git clone <repository-url>
cd FastAPI-RAG-Chroma
```

2. Install dependencies
```bash
pip install .
# or with uv
uv sync
```

3. Start the server
```bash
uv run uvicorn main:app --reload
```

The API will be available at `http://localhost:8000`

## Usage

### Upload Documents

```bash
curl -X POST "http://localhost:8000/files/" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@your-document.pdf"
```

### Ask Questions

```bash
curl -X POST "http://localhost:8000/chat/What is the main topic discussed?"
```

### List Uploaded Documents

```bash
curl "http://localhost:8000/files/"
```

## API Endpoints

- `POST /files/` - Upload a document
- `GET /files/` - List all uploaded documents
- `POST /chat/{question}` - Ask a question about uploaded documents

## Project Structure

```
├── api/
│   └── v1/
│       ├── files.py    # File upload endpoints
│       └── chat.py     # Chat endpoints
├── db/
│   └── __init__.py     # Vector store configuration
├── services/
│   └── rag.py          # RAG processing logic
├── schemas/
│   └── files.py        # Pydantic models
└── main.py             # FastAPI application
```

## Configuration

The system uses Ollama's `gpt-oss:20b` model by default, but you can use any Ollama-compatible model. Update the model name in `services/rag.py` if needed:

```python
self.llm = ChatOllama(model="your-preferred-model", temperature=0)
```

Popular lightweight options:
```bash
# Small and fast
ollama pull llama3.2:1b
# Medium quality/speed balance  
ollama pull llama3.2:3b
# Good quality
ollama pull mistral:7b

ollama serve
```

## How It Works

1. **Document Processing**: Files are converted to text and split into semantic chunks
2. **Vectorization**: Chunks are embedded using sentence transformers and stored in ChromaDB
3. **Retrieval**: Questions are converted to embeddings and matched against stored chunks
4. **Generation**: Retrieved context is passed to the LLM for answer generation
5. **Attribution**: Sources are tracked and returned with each answer

## License

MIT License
