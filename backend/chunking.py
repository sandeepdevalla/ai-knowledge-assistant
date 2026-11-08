
import re
from typing import Any



def _split_long_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split oversized text into bounded segments with overlap."""
    chunks: list[str] = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))

        if end < len(text):
            boundary = text.rfind(" ", start + chunk_size // 2, end)
            if boundary > start:
                end = boundary

        content = text[start:end].strip()
        if content:
            chunks.append(content)

        if end >= len(text):
            break

        # Keep the requested overlap, but don't exceed the chunk limit.
        next_start = end - overlap
        if next_start <= start:
            next_start = end

        if text[next_start:next_start + 1] == " ":
            next_start += 1

        start = next_start

    return chunks



def chunk_pages(
    pages: list[dict[str, Any]],
    chunk_size: int = 800,
    overlap: int = 120,
) -> list[dict[str, Any]]:
    """Create bounded, overlapping chunks while preserving paragraph boundaries."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and smaller than chunk_size")

    chunks: list[dict[str, Any]] = []
    chunk_id = 0

    for page in pages:
        text = page.get("text", "").strip()
        page_number = page.get("page")

        if not text:
            continue

        # Preserve paragraph boundaries where the extracted text provides them.
        paragraphs = [
            part.strip()
            for part in re.split(r"\n\s*\n+", text)
            if part.strip()
        ]

        # Some PDF extractors provide line breaks but no blank lines.
        if not paragraphs:
            continue

        units: list[str] = []
        for paragraph in paragraphs:
            if len(paragraph) <= chunk_size:
                units.append(paragraph)
            else:
                units.extend(_split_long_text(paragraph, chunk_size, overlap))

        current = ""

        for unit in units:
            candidate = f"{current}\n\n{unit}" if current else unit

            if len(candidate) <= chunk_size:
                current = candidate
                continue

            if current:
                chunks.append(
                    {
                        "chunk_id": chunk_id,
                        "page": page_number,
                        "text": current,
                    }
                )
                chunk_id += 1


            # Oversized paragraphs have already been split with overlap.
            # Do not add a second overlap when packing their segments.
            tail = ""
            candidate = unit

            if len(candidate) <= chunk_size:
                current = candidate
            else:
                current = unit

        if current:
            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "page": page_number,
                    "text": current,
                }
            )
            chunk_id += 1

    return chunks
