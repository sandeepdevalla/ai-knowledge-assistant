
import json
import time
from pathlib import Path

from backend.generation import AnswerGenerator
from backend.index_documents import load_documents
from backend.retrieval import SemanticRetriever


BASE_DIR = Path(__file__).resolve().parent.parent
QUESTIONS_FILE = BASE_DIR / "evaluation" / "questions.json"
RESULTS_FILE = BASE_DIR / "evaluation" / "results.json"


def main():
    questions = json.loads(QUESTIONS_FILE.read_text())

    chunks = load_documents()
    retriever = SemanticRetriever()
    retriever.index(chunks)
    generator = AnswerGenerator()

    results = []

    for item in questions:
        question = item["question"]
        started = time.perf_counter()

        retrieved = retriever.search(question, top_k=3)
        retrieval_seconds = time.perf_counter() - started

        expected_sources = {
            (source["document"], source["page"])
            for source in item["sources"]
        }
        retrieved_sources = {
            (chunk["document"], chunk["page"])
            for chunk in retrieved
        }

        matched_sources = expected_sources & retrieved_sources

        generation_started = time.perf_counter()
        try:
            generated = generator.generate(question, retrieved)
            generation_seconds = time.perf_counter() - generation_started

            answer = generated["answer"]
            usage = generated.get("usage", {})
            error = None
        except Exception as exc:
            generation_seconds = time.perf_counter() - generation_started
            answer = None
            usage = {}
            error = f"{type(exc).__name__}: {exc}"

        result = {
            "id": item["id"],
            "category": item["category"],
            "question": question,
            "answerable": item["answerable"],
            "expected_points": item["expected_points"],
            "expected_sources": [
                {"document": doc, "page": page}
                for doc, page in sorted(expected_sources)
            ],
            "retrieved": [
                {
                    "document": chunk["document"],
                    "page": chunk["page"],
                    "score": round(chunk["score"], 4),
                    "text": chunk["text"],
                }
                for chunk in retrieved
            ],
            "matched_source_count": len(matched_sources),
            "expected_source_count": len(expected_sources),
            "source_coverage": (
                len(matched_sources) / len(expected_sources)
                if expected_sources else None
            ),
            "answer": answer,
            "usage": usage,
            "retrieval_seconds": round(retrieval_seconds, 3),
            "generation_seconds": round(generation_seconds, 3),
            "error": error,
        }

        results.append(result)
        print(
            f"{item['id']} [{item['category']}] "
            f"source coverage={result['source_coverage']} "
            f"generation_error={error is not None}"
        )

    successful = [result for result in results if result["error"] is None]
    measurable = [
        result for result in results
        if result["source_coverage"] is not None
    ]

    summary = {
        "question_count": len(results),
        "generation_success_count": len(successful),
        "generation_error_count": len(results) - len(successful),
        "mean_source_coverage": (
            round(
                sum(r["source_coverage"] for r in measurable)
                / len(measurable),
                3,
            )
            if measurable else None
        ),
        "results": results,
    }

    RESULTS_FILE.write_text(json.dumps(summary, indent=2))
    print("\nSummary:")
    print(json.dumps({
        key: value for key, value in summary.items()
        if key != "results"
    }, indent=2))
    print(f"\nDetailed results saved to: {RESULTS_FILE}")


if __name__ == "__main__":
    main()
