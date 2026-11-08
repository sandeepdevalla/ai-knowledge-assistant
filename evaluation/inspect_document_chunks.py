
from backend.index_documents import load_documents

chunks = load_documents()

search_terms = {
    "NIST-SP-800-145.pdf": [
        "essential characteristics",
        "on-demand self-service",
        "cloud computing",
    ],
    "NIST-SP-800-207.pdf": [
        "zero trust",
        "operative definition",
        "logical components",
    ],
}

for document, terms in search_terms.items():
    print(f"\n{'=' * 70}")
    print(document)

    for term in terms:
        matches = [
            chunk for chunk in chunks
            if chunk["document"] == document
            and term.lower() in chunk["text"].lower()
        ]

        print(f"\nSearch term: {term}")
        if not matches:
            print("No matching chunks found.")
            continue

        for chunk in matches[:2]:
            print(
                f"PDF page {chunk['page']} | "
                f"{len(chunk['text'])} characters"
            )
            print(chunk["text"][:700].replace("\n", " "))
