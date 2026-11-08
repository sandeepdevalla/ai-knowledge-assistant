
import json
import time
from pathlib import Path

from backend.index_documents import load_documents
from backend.retrieval import SemanticRetriever


BASE_DIR = Path(__file__).resolve().parent.parent
QUESTIONS_FILE = BASE_DIR / "evaluation" / "questions.json"
RESULTS_FILE = BASE_DIR / "evaluation" / "retrieval_comparison.json"


def source_coverage(item, retrieved):
    expected = {
        (source["document"], source["page"])
        for source in item["sources"]
    }
    found = {
        (chunk["document"], chunk["page"])
        for chunk in retrieved
    }
    return len(expected & found) / len(expected) if expected else None


def document_aware_search(retriever, question, top_k=5):
    """Restrict hybrid retrieval when exactly one document is identified."""
    document_match = retriever.identify_document(question)

    if document_match:
        document = document_match["document"]
        filtered_chunks = [
            chunk for chunk in retriever.chunks
            if chunk["document"] == document
        ]

        filtered_retriever = SemanticRetriever()
        filtered_retriever.index(filtered_chunks)

        return (
            filtered_retriever.search_hybrid(
                question, top_k=top_k, candidate_k=50
            ),
            f"filtered:{document}",
        )

    return (
        retriever.search_hybrid(
            question, top_k=top_k, candidate_k=50
        ),
        "global_fallback",
    )


def main():
    questions = json.loads(QUESTIONS_FILE.read_text())

    print("Loading documents and building retrieval index...")
    chunks = load_documents()
    retriever = SemanticRetriever()
    retriever.index(chunks)

    results = []

    for item in questions:
        question = item["question"]
        result = {
            "id": item["id"],
            "category": item["category"],
            "question": question,
            "answerable": item["answerable"],
        }

        strategies = {
            "semantic": lambda: (
                retriever.search(question, top_k=5),
                "global",
            ),
            "hybrid": lambda: (
                retriever.search_hybrid(
                    question, top_k=5, candidate_k=50
                ),
                "global",
            ),
            "document_aware": lambda: document_aware_search(
                retriever, question, top_k=5
            ),
        }

        for strategy, run_search in strategies.items():
            started = time.perf_counter()
            retrieved, retrieval_mode = run_search()
            elapsed = time.perf_counter() - started

            result[f"{strategy}_mode"] = retrieval_mode
            result[f"{strategy}_top3_coverage"] = source_coverage(
                item, retrieved[:3]
            )
            result[f"{strategy}_top5_coverage"] = source_coverage(
                item, retrieved
            )
            result[f"{strategy}_seconds"] = round(elapsed, 4)
            result[f"{strategy}_top5"] = [
                {
                    "document": chunk["document"],
                    "page": chunk["page"],
                    "score": round(float(chunk["score"]), 6),
                    "text": chunk["text"],
                }
                for chunk in retrieved
            ]

        print(
            f"{item['id']}: top-3 coverage "
            f"semantic={result['semantic_top3_coverage']}, "
            f"hybrid={result['hybrid_top3_coverage']}, "
            f"document_aware={result['document_aware_top3_coverage']}"
        )
        results.append(result)

    RESULTS_FILE.write_text(json.dumps(results, indent=2))

    answerable = [r for r in results if r["answerable"]]

    def mean(rows, key):
        values = [r[key] for r in rows if r[key] is not None]
        return sum(values) / len(values) if values else None

    print("\n=== RETRIEVAL COMPARISON ===")
    print(f"Questions evaluated: {len(results)}")
    print(f"Answerable questions: {len(answerable)}")
    print(f"Unsupported questions: {len(results) - len(answerable)}")

    for k in (3, 5):
        key_suffix = f"_top{k}_coverage"
        for group_name, group in (
            ("Answerable", answerable),
            ("All questions with expected sources", results),
        ):
            print(f"\n{group_name} top-{k} mean source coverage:")
            for strategy in ("semantic", "hybrid", "document_aware"):
                value = mean(group, f"{strategy}{key_suffix}")
                if value is not None:
                    print(f"  {strategy}: {value:.3f}")

    print(f"\nDetailed results saved to: {RESULTS_FILE}")


if __name__ == "__main__":
    main()
