from pathlib import Path
from typing import List, Dict

import fitz
from docx import Document as DocxDocument

from llama_index.core.schema import TextNode


def clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.strip() for line in text.split("\n")]
    text = "\n".join(line for line in lines if line)
    return text.strip()


def chunk_text(text: str, chunk_size: int = 900, overlap: int = 120) -> List[str]:
    if not text:
        return []

    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start = max(end - overlap, start + 1)

    return chunks


def parse_pdf(path: Path):
    pages = []
    with fitz.open(path) as pdf:
        for page_number, page in enumerate(pdf, start=1):
            pages.append((page_number, page.get_text("text")))
    return pages


def parse_docx(path: Path):
    doc = DocxDocument(path)
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    return [(1, "\n".join(paragraphs))]


def parse_txt(path: Path):
    return [(1, path.read_text(encoding="utf-8", errors="ignore"))]


def parse_file(path: Path):
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return parse_pdf(path)
    if suffix == ".docx":
        return parse_docx(path)
    if suffix == ".txt":
        return parse_txt(path)
    raise ValueError("Only PDF, DOCX and TXT files are supported.")


def ingest_file(path: Path, document_id: str, rag):
    pages = parse_file(path)
    nodes = []

    for page_number, raw_text in pages:
        cleaned = clean_text(raw_text)
        chunks = chunk_text(cleaned)

        for index, chunk in enumerate(chunks):
            node = TextNode(
                text=chunk,
                id_=f"{document_id}_page_{page_number}_chunk_{index}",
                metadata={
                    "document_id": document_id,
                    "document_name": path.name,
                    "page": page_number,
                    "section": "Document text",
                    "chunk_id": f"{document_id}_page_{page_number}_chunk_{index}",
                },
            )
            nodes.append(node)

    if not nodes:
        raise ValueError(f"No text could be extracted from {path.name}.")

    rag.add_nodes(nodes)

    return {
        "document_id": document_id,
        "document_name": path.name,
        "pages": len(pages),
        "chunks": len(nodes),
    }
