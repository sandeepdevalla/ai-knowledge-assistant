
from pathlib import Path

import pymupdf

from backend.chunking import chunk_pages
from backend.retrieval import SemanticRetriever

DOCUMENTS_DIR = Path(__file__).resolve().parent.parent / "documents"
DOCUMENT_TITLES = {
    "NIST-SP-800-145.pdf": "The NIST Definition of Cloud Computing",
    "NIST-SP-800-207.pdf": "Zero Trust Architecture",
}

def load_documents() -> list[dict]:
    all_chunks = []

    for pdf_path in sorted(DOCUMENTS_DIR.glob("*.pdf")):
        with pymupdf.open(pdf_path) as document:
            pages = [
                {
                    "page": page.number + 1,
                    "text": page.get_text().strip(),
                }
                for page in document
            ]

        chunks = chunk_pages(pages)

        for chunk in chunks:
            chunk["document"] = pdf_path.name
            chunk["document_title"] = DOCUMENT_TITLES.get(
                pdf_path.name, pdf_path.stem
            )

        all_chunks.extend(chunks)
        print(f"{pdf_path.name}: {len(pages)} pages, {len(chunks)} chunks")

    if not all_chunks:
        raise RuntimeError(f"No PDF documents found in {DOCUMENTS_DIR}")

    return all_chunks


def main() -> None:
    chunks = load_documents()

    retriever = SemanticRetriever()
    retriever.index(chunks)

    question = "What are the essential characteristics of cloud computing?"
    results = retriever.search(question, top_k=3)

    print(f"\nQuestion: {question}")
    print("\nTop matching passages:")

    for rank, result in enumerate(results, start=1):
        print(f"\n--- Result {rank} ---")
        print(f"Document: {result['document']}")
        print(f"Page: {result['page']}")
        print(f"Similarity: {result['score']:.4f}")
        print(f"Text: {result['text']}")


if __name__ == "__main__":
    main()
