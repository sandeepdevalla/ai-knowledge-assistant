
import pytest

from backend.chunking import chunk_pages


def test_chunk_size_and_overlap():
    pages = [{"page": 1, "text": "A" * 1000}]

    chunks = chunk_pages(pages)

    assert len(chunks) == 2
    assert len(chunks[0]["text"]) == 800
    assert len(chunks[1]["text"]) == 320
    assert chunks[1]["text"][:120] == chunks[0]["text"][-120:]


def test_page_metadata_is_preserved():
    pages = [
        {"page": 1, "text": "Cloud computing"},
        {"page": 2, "text": "Zero trust architecture"},
    ]

    chunks = chunk_pages(pages)

    assert [chunk["page"] for chunk in chunks] == [1, 2]
    assert [chunk["chunk_id"] for chunk in chunks] == [0, 1]


def test_empty_pages_are_skipped():
    pages = [
        {"page": 1, "text": ""},
        {"page": 2, "text": "   "},
        {"page": 3, "text": "Valid text"},
    ]

    chunks = chunk_pages(pages)

    assert len(chunks) == 1
    assert chunks[0]["page"] == 3


@pytest.mark.parametrize(
    ("chunk_size", "overlap"),
    [(0, 0), (-1, 0), (100, -1), (100, 100)],
)
def test_invalid_settings_are_rejected(chunk_size, overlap):
    with pytest.raises(ValueError):
        chunk_pages(
            [{"page": 1, "text": "Sample text"}],
            chunk_size=chunk_size,
            overlap=overlap,
        )


def test_chunking_preserves_page_metadata_for_cross_page_content():
    pages = [
        {
            "page": 6,
            "text": (
                "Platform as a Service (PaaS). The consumer deploys "
                "applications created using programming"
            ),
        },
        {
            "page": 7,
            "text": (
                "languages, libraries, services, and tools supported "
                "by the provider. Infrastructure as a Service (IaaS). "
                "The consumer provisions processing and storage."
            ),
        },
    ]

    chunks = chunk_pages(pages)

    page_six = [c for c in chunks if c["page"] == 6]
    page_seven = [c for c in chunks if c["page"] == 7]

    assert page_six
    assert page_seven
    assert any("programming" in c["text"] for c in page_six)
    assert any("languages, libraries" in c["text"] for c in page_seven)
    assert all(len(c["text"]) <= 800 for c in chunks)
