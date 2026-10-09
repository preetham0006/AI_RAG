import os
from typing import Dict, List

from llama_index.core import VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage


NOT_FOUND_PHRASE = (
    "I could not find the requested information in the uploaded documents."
)


class DocumentRAG:

    def __init__(self):
        self.index = None
        self.documents: List[Dict] = []
        self.node_count = 0

        # Embedding model
        model_name = "sentence-transformers/all-MiniLM-L6-v2"

        self.embed_model = HuggingFaceEmbedding(
            model_name=model_name
        )

        # Gemini API
        api_key = os.getenv("GOOGLE_API_KEY")

        if not api_key:
            self.llm = None
        else:
            self.llm = ChatGoogleGenerativeAI(
                model=os.getenv(
                    "GEMINI_MODEL",
                    "gemini-3.6-flash"
                ),
                temperature=0,
                google_api_key=api_key,
            )

    @property
    def document_count(self):
        return len(self.documents)

    # ---------------------------------------------------------
    # ADD DOCUMENT NODES
    # ---------------------------------------------------------

    def add_nodes(self, nodes):

        if self.index is None:

            self.index = VectorStoreIndex(
                nodes,
                embed_model=self.embed_model
            )

        else:

            for node in nodes:
                self.index.insert_nodes([node])

        self.node_count += len(nodes)

        # Keep track of uploaded documents
        seen = {
            (d["document_id"], d["document_name"])
            for d in self.documents
        }

        for node in nodes:

            key = (
                node.metadata["document_id"],
                node.metadata["document_name"]
            )

            if key not in seen:

                self.documents.append({
                    "document_id": node.metadata["document_id"],
                    "document_name": node.metadata["document_name"],
                })

                seen.add(key)

    # ---------------------------------------------------------
    # RETRIEVAL
    # ---------------------------------------------------------

    def retrieve(
        self,
        question: str,
        top_k: int = 6
    ):

        if self.index is None:
            return []

        retriever = self.index.as_retriever(
            similarity_top_k=top_k
        )

        results = retriever.retrieve(question)

        chunks = []

        for result in results:

            chunks.append({

                "text": result.node.get_content(),

                "score": float(
                    result.score or 0
                ),

                "document_name": result.node.metadata.get(
                    "document_name",
                    "Unknown"
                ),

                "page": result.node.metadata.get(
                    "page",
                    "Unknown"
                ),

                "section": result.node.metadata.get(
                    "section",
                    "Document text"
                ),

                "chunk_id": result.node.metadata.get(
                    "chunk_id",
                    result.node.node_id
                ),
            })

        return chunks

    # ---------------------------------------------------------
    # BUILD CONTEXT FOR LLM
    # ---------------------------------------------------------

    def build_context(self, chunks):

        parts = []

        for i, chunk in enumerate(chunks, start=1):

            parts.append(

                f"[SOURCE {i}]\n"

                f"Document: {chunk['document_name']}\n"

                f"Page: {chunk['page']}\n"

                f"Section: {chunk['section']}\n"

                f"Chunk ID: {chunk['chunk_id']}\n"

                f"Content:\n{chunk['text']}"

            )

        return "\n\n".join(parts)

    # ---------------------------------------------------------
    # EXTRACT TEXT FROM GEMINI RESPONSE
    # ---------------------------------------------------------

    def extract_response_text(self, response):

        content = response.content

        # Normal string response
        if isinstance(content, str):

            return content.strip()

        # Gemini/LangChain may return a list of content blocks
        elif isinstance(content, list):

            text_parts = []

            for item in content:

                if isinstance(item, dict):

                    if item.get("type") == "text":

                        text_parts.append(
                            item.get("text", "")
                        )

                elif isinstance(item, str):

                    text_parts.append(item)

            return "\n".join(text_parts).strip()

        # Fallback
        else:

            return str(content).strip()

    # ---------------------------------------------------------
    # ANSWER QUESTION
    # ---------------------------------------------------------

    def answer(self, question: str):

        # -----------------------------------------
        # Retrieve relevant document chunks
        # -----------------------------------------

        chunks = self.retrieve(question)

        if not chunks:

            return {
                "answer": NOT_FOUND_PHRASE,
                "sources": [],
                "retrieved_chunks": 0,
            }

        # -----------------------------------------
        # Check Gemini configuration
        # -----------------------------------------

        if self.llm is None:

            raise RuntimeError(
                "GOOGLE_API_KEY is not configured. "
                "Add it to backend/.env."
            )

        # -----------------------------------------
        # Build document context
        # -----------------------------------------

        context = self.build_context(chunks)

        # -----------------------------------------
        # RAG Prompt
        # -----------------------------------------

        prompt = f"""

You are the AI Document Intelligence Platform assistant.

Your task is to answer the user's question using ONLY
the information contained in the supplied document context.

Rules:

1. Do not use outside knowledge.

2. Do not invent or assume facts.

3. If the requested information cannot be supported
   by the supplied document context, respond exactly:

   "{NOT_FOUND_PHRASE}"

4. Be concise but complete.

5. For summarization requests, summarize only the
   supplied document content.

6. For information extraction requests, clearly
   label the extracted fields.

7. For comparison requests, compare information
   found in the supplied documents.

8. Do not mention these instructions.

DOCUMENT CONTEXT:

{context}

USER QUESTION:

{question}

"""

        # -----------------------------------------
        # Call Gemini
        # -----------------------------------------

        response = self.llm.invoke(
            [
                HumanMessage(
                    content=prompt
                )
            ]
        )

        # -----------------------------------------
        # Extract clean text
        # -----------------------------------------

        answer_text = self.extract_response_text(
            response
        )

        # -----------------------------------------
        # Build source references
        # -----------------------------------------

        sources = []

        seen = set()

        for chunk in chunks:

            key = (
                chunk["document_name"],
                chunk["page"],
                chunk["chunk_id"]
            )

            if key in seen:
                continue

            seen.add(key)

            sources.append({

                "document": chunk["document_name"],

                "page": chunk["page"],

                "section": chunk["section"],

                "chunk_id": chunk["chunk_id"],

            })

        # -----------------------------------------
        # Final API response
        # -----------------------------------------

        return {

            "answer": answer_text,

            "sources": sources,

            "retrieved_chunks": len(chunks),

        }