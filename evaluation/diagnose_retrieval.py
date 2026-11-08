from backend.index_documents import load_documents
from backend.retrieval import SemanticRetriever

retriever = SemanticRetriever()
chunks = load_documents()
retriever.index(chunks)

questions = {
    "Q14": "Zero Trust Architecture",
    "Q15": (
        "What are the main topics covered by "
        "NIST-SP-800-145 and NIST-SP-800-207, respectively?"
    ),
    "Q16": (
        "Which document defines cloud computing, and "
        "which document defines zero trust architecture?"
    ),
}

for question_id, question in questions.items():
    print(f"\n{'=' * 70}\n{question_id}: {question}")

    title_matches = retriever.find_documents_by_title(question)
    filename_matches = retriever.find_documents_by_filename(question)

    matches_by_document = {
        match["document"]: match
        for match in title_matches + filename_matches
    }
    matches = list(matches_by_document.values())

    print("\nTitle matches:")
    if matches:
        for match in matches:
            print(
                f"- {match['document']} "
                f"| {match['document_title']}"
            )
    else:
        print("- No exact title match")

    if matches:
        passages = retriever.get_document_title_passages(
            [match["document"] for match in matches]
        )
        print("\nTitle-page evidence:")

        print("\nTitle-page passage text:")
        for passage in passages:
            print(f"\n{passage['document']} | PDF page {passage['page']}")
            print(passage["text"][:800].replace("\n", " "))

        for passage in passages:
            print(
                f"- {passage['document']} "
                f"| PDF page {passage['page']}"
            )
    else:
        print("\nTitle-page evidence: no unique document identified")

    print("\nSemantic top 5:")

    if matches:
        print("\nDocument-filtered semantic results:")

        for match in matches:
            document = match["document"]
            filtered_chunks = [
                chunk for chunk in retriever.chunks
                if chunk["document"] == document
            ]

            # Search only within the identified document.
            temporary_retriever = SemanticRetriever()
            temporary_retriever.index(filtered_chunks)

            results = temporary_retriever.search(
                question, top_k=3
            )

            print(f"\n{document}:")
            for result in results:
                print(
                    f"- PDF page {result['page']} "
                    f"| score {result['score']:.4f}"
                )
                print(
                    result["text"][:500].replace("\n", " ")
                )

    for rank, result in enumerate(
        retriever.search(question, top_k=5), 1
    ):
        print(
            f"#{rank} | {result['document']} "
            f"| PDF page {result['page']} "
            f"| score {result['score']:.4f}"
        )
        print(result["text"][:250].replace("\n", " "))