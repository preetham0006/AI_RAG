# AI Document Intelligence Platform

POC implementation based strictly on the supplied project specification.

## What it does

- Upload PDF, DOCX and TXT documents.
- Extract and clean text.
- Chunk documents.
- Generate embeddings.
- Store/index document chunks with document name, page number, section and chunk ID.
- Perform semantic retrieval.
- Use Retrieval-Augmented Generation (RAG) with an LLM.
- Answer questions with source/page references.
- Refuse unsupported questions when the requested information is not present in the uploaded documents.
- Support summarization.
- Support structured information extraction through natural-language prompts.
- Support comparison across multiple uploaded documents.
- Provide a simple frontend.

## Architecture

Document ingestion:

User upload -> Parser -> Text extraction -> Cleaning -> Chunking -> Embeddings -> LlamaIndex vector index

Question answering:

User question -> Query processing -> Query embedding/retrieval -> Relevant chunks -> Context construction -> LangChain LLM -> Answer + source references

## Technology

- Python
- FastAPI
- LangChain
- LlamaIndex
- Google Gemini
- Sentence Transformers / Hugging Face embeddings through LlamaIndex
- PyMuPDF
- python-docx
- Simple HTML/CSS/JavaScript frontend

No Docker, Redis, RabbitMQ, PostgreSQL, authentication, OCR, image understanding, chart understanding or cloud deployment is included because these are outside the supplied POC scope.

## Setup

Python 3.11+ is recommended.

```bash
cd backend
python -m venv .venv

# Windows PowerShell
.venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

Create `backend/.env`:

```env
GOOGLE_API_KEY=your_gemini_api_key
```

Run:

```bash
cd backend
uvicorn main:app --reload
```

Open:

http://127.0.0.1:8000

## Notes

The POC keeps the vector index in memory. Restarting the backend requires documents to be uploaded again. This is intentional for a simple demonstration and is not a production persistence layer.
