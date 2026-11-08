from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
app = FastAPI(title="AI Knowledge Assistant API")

from backend.chunking import chunk_pages
from backend.retrieval import SemanticRetriever
from backend.generation import AnswerGenerator
from backend.index_documents import load_documents

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


from functools import lru_cache

import pymupdf


@lru_cache(maxsize=1)
def get_retriever() -> SemanticRetriever:
    retriever = SemanticRetriever()
    retriever.index(load_documents())
    return retriever


@lru_cache(maxsize=1)
def get_generator() -> AnswerGenerator:
    return AnswerGenerator()

@app.get("/health")
def health_check():
    return {"status": "ok", "service": "document-processing"}


@app.post("/documents/extract")
async def extract_pdf(file: UploadFile = File(...)):
    if file.content_type != "application/pdf":
        raise HTTPException(
            status_code=415,
            detail="Only PDF files are supported",
        )

    content = await file.read()

    if not content:
        raise HTTPException(status_code=400, detail="Empty file")

    try:
        with pymupdf.open(stream=content, filetype="pdf") as document:
            if document.is_encrypted:
                raise HTTPException(
                    status_code=400,
                    detail="Encrypted PDFs are not supported",
                )

            pages = [
                {
                    "page": page.number + 1,
                    "text": page.get_text().strip(),
                }
                for page in document
            ]

    except HTTPException:
        raise
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Could not read this PDF",
        )

    return {
        "filename": file.filename,
        "page_count": len(pages),
        "pages": pages,
    }


@app.post("/documents/chunk")
async def chunk_pdf(file: UploadFile = File(...)):
    if file.content_type != "application/pdf":
        raise HTTPException(
            status_code=415,
            detail="Only PDF files are supported",
        )

    content = await file.read()

    if not content:
        raise HTTPException(status_code=400, detail="Empty file")

    try:
        with pymupdf.open(stream=content, filetype="pdf") as document:
            if document.is_encrypted:
                raise HTTPException(
                    status_code=400,
                    detail="Encrypted PDFs are not supported",
                )

            pages = [
                {
                    "page": page.number + 1,
                    "text": page.get_text().strip(),
                }
                for page in document
            ]
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Could not read this PDF",
        )

    chunks = chunk_pages(pages)

    return {
        "filename": file.filename,
        "page_count": len(pages),
        "chunk_count": len(chunks),
        "chunk_size": 800,
        "overlap": 120,
        "chunks": chunks,
    }

@app.post("/ask")
async def ask_question(payload: dict):
    question = payload.get("question", "")

    if not isinstance(question, str) or not question.strip():
        raise HTTPException(
            status_code=400,
            detail="A non-empty question is required",
        )

    retriever = get_retriever()
    
    passages = retriever.search_hybrid(
        question,
        top_k=5,
        candidate_k=50,
    )
    for index, passage in enumerate(passages, start=1):
        print(
            f"\n[{index}] "
            f"{passage['document']} | "
            f"PDF page {passage['page']}"
        )
        print(passage["text"][:500])

    generator = get_generator()
    result = generator.generate(question, passages)

    return result