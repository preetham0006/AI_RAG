import os
import re
import shutil
import uuid
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from ingestion import ingest_file
from rag_service import DocumentRAG

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR.parent / "data" / "uploads"
FRONTEND_DIR = BASE_DIR.parent / "frontend"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="AI Document Intelligence Platform", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

rag = DocumentRAG()


class QuestionRequest(BaseModel):
    question: str


@app.get("/")
def home():
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/api/health")
def health():
    return {"status": "ok", "documents": rag.document_count}


@app.get("/api/documents")
def documents():
    return {"documents": rag.documents}


@app.post("/api/documents/upload")
async def upload_documents(files: List[UploadFile] = File(...)):
    allowed = {".pdf", ".docx", ".txt"}
    results = []

    for file in files:
        suffix = Path(file.filename or "").suffix.lower()
        if suffix not in allowed:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {suffix}. Only PDF, DOCX and TXT are supported."
            )

        document_id = str(uuid.uuid4())
        safe_name = re.sub(r"[^A-Za-z0-9._-]", "_", file.filename or "document")
        stored_path = UPLOAD_DIR / f"{document_id}_{safe_name}"

        with stored_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        try:
            metadata = ingest_file(stored_path, document_id, rag)
            results.append(metadata)
        except Exception as exc:
            stored_path.unlink(missing_ok=True)
            raise HTTPException(status_code=500, detail=str(exc))

    return {"message": "Documents processed successfully", "documents": results}


@app.post("/api/chat")
def chat(request: QuestionRequest):
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    if rag.document_count == 0:
        raise HTTPException(status_code=400, detail="Upload at least one document first.")

    try:
        return rag.answer(question)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
