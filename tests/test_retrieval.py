
from backend.retrieval import SemanticRetriever


def test_semantic_retrieval_returns_relevant_chunk():
    retriever = SemanticRetriever()

    chunks = [
        {
            "chunk_id": 0,
            "page": 1,
            "text": "Cloud computing provides on-demand access to shared computing resources.",
        },
        {
            "chunk_id": 1,
            "page": 2,
            "text": "Zero trust architecture requires explicit verification of access requests.",
        },
        {
            "chunk_id": 2,
            "page": 3,
            "text": "A hybrid cloud combines distinct cloud infrastructures.",
        },
    ]

    retriever.index(chunks)
    results = retriever.search(
        "What is cloud computing and how does it provide resources?",
        top_k=2,
    )

    assert len(results) == 2
    assert results[0]["chunk_id"] == 0
    assert results[0]["page"] == 1
    assert results[0]["score"] > results[1]["score"]


def test_empty_query_is_rejected():
    retriever = SemanticRetriever()

    try:
        retriever.search(" ")
        assert False, "Expected ValueError"
    except ValueError as exc:
        assert "Query cannot be empty" in str(exc)

def test_find_documents_by_title():
    retriever = SemanticRetriever()
    retriever.chunks = [
        {
            "document": "NIST-SP-800-145.pdf",
            "document_title": "The NIST Definition of Cloud Computing",
            "page": 1,
            "text": "Cloud computing title page",
        },
        {
            "document": "NIST-SP-800-207.pdf",
            "document_title": "Zero Trust Architecture",
            "page": 1,
            "text": "Zero trust title page",
        },
    ]

    assert retriever.find_documents_by_title(
        "Zero Trust Architecture"
    ) == [
        {
            "document": "NIST-SP-800-207.pdf",
            "document_title": "Zero Trust Architecture",
        }
    ]


def test_find_documents_by_title_returns_empty_for_unknown_title():
    retriever = SemanticRetriever()
    retriever.chunks = [
        {
            "document": "NIST-SP-800-145.pdf",
            "document_title": "The NIST Definition of Cloud Computing",
            "page": 1,
            "text": "Cloud computing title page",
        },
    ]

    assert retriever.find_documents_by_title("Unknown Document") == []

def test_identify_document_by_full_title():
    retriever = SemanticRetriever()
    retriever.chunks = [
        {
            "document": "NIST-SP-800-207.pdf",
            "document_title": "Zero Trust Architecture",
            "page": 1,
            "text": "Title page",
        }
    ]

    result = retriever.identify_document("Zero Trust Architecture")

    assert result == {
        "document": "NIST-SP-800-207.pdf",
        "document_title": "Zero Trust Architecture",
    }


def test_identify_document_returns_none_for_unknown_title():
    retriever = SemanticRetriever()
    retriever.chunks = [
        {
            "document": "NIST-SP-800-207.pdf",
            "document_title": "Zero Trust Architecture",
            "page": 1,
            "text": "Title page",
        }
    ]

    assert retriever.identify_document("Unknown Document") is None

def test_get_document_title_passage_returns_page_one():
    retriever = SemanticRetriever()
    retriever.chunks = [
        {
            "document": "NIST-SP-800-207.pdf",
            "document_title": "Zero Trust Architecture",
            "page": 1,
            "text": "NIST Special Publication 800-207 Zero Trust Architecture",
        },
        {
            "document": "NIST-SP-800-207.pdf",
            "document_title": "Zero Trust Architecture",
            "page": 2,
            "text": "Other content",
        },
    ]

    passages = retriever.get_document_title_passage(
        "NIST-SP-800-207.pdf"
    )

    assert len(passages) == 1
    assert passages[0]["page"] == 1
    assert "Zero Trust Architecture" in passages[0]["text"]
