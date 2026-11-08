import re
from collections import Counter

import numpy as np
from sentence_transformers import SentenceTransformer


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\b[a-z0-9]+\b", text.lower())


class SemanticRetriever:
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        self.model = SentenceTransformer(model_name)
        self.chunks: list[dict] = []
        self.embeddings: np.ndarray | None = None
        self.tokenized_chunks: list[list[str]] = []
        self.document_frequencies: Counter = Counter()
        self.average_document_length = 0.0

    def index(self, chunks: list[dict]) -> None:
        """Index chunks for semantic and keyword retrieval."""
        if not chunks:
            raise ValueError("Cannot index an empty chunk list")

        texts = [chunk["text"] for chunk in chunks]
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        tokenized = [_tokenize(text) for text in texts]
        document_frequencies = Counter()

        for tokens in tokenized:
            document_frequencies.update(set(tokens))

        self.chunks = chunks
        self.embeddings = embeddings
        self.tokenized_chunks = tokenized
        self.document_frequencies = document_frequencies
        self.average_document_length = (
            sum(map(len, tokenized)) / len(tokenized)
        )

    
    def find_documents_by_title(self, query: str) -> list[dict]:
        """Find indexed documents whose titles are explicitly mentioned."""
        query_terms = set(_tokenize(query))
        matches = []
        seen_documents = set()

        for chunk in self.chunks:
            document = chunk["document"]

            if document in seen_documents:
                continue

            seen_documents.add(document)
            title = chunk.get("document_title", "")
            title_terms = set(_tokenize(title))

            # Require at least two meaningful title terms to match.
            # Ignore common words that occur in ordinary questions.
            meaningful_terms = title_terms - {
                "the", "of", "and", "a", "an"
            }
            matched_terms = query_terms & meaningful_terms

            if len(matched_terms) >= 2:
                matches.append({
                    "document": document,
                    "document_title": title,
                })

        return matches


    def find_documents_by_filename(self, query: str) -> list[dict]:
        """Find indexed documents whose filenames are mentioned in a query."""
        query_lower = query.lower()
        matches = []
        seen_documents = set()

        for chunk in self.chunks:
            document = chunk["document"]

            if document in seen_documents:
                continue

            filename_without_extension = document.rsplit(".", 1)[0].lower()

            if (
                document.lower() in query_lower
                or filename_without_extension in query_lower
            ):
                seen_documents.add(document)
                matches.append({
                    "document": document,
                    "document_title": chunk.get("document_title", ""),
                })

        return matches


    def identify_document(self, query: str) -> dict | None:
        """Identify a document when the query contains its full title."""
        matches = self.find_documents_by_title(query)

        if len(matches) == 1:
            return matches[0]

        return None
    
    def get_document_title_passage(self, document: str) -> list[dict]:
        """Return the first-page passage for an identified document."""
        for chunk in self.chunks:
            if chunk["document"] == document and chunk["page"] == 1:
                return [chunk]

        return []
    
    def get_document_title_passages(
        self, documents: list[str]
    ) -> list[dict]:
        """Return title-page evidence for multiple identified documents."""
        passages = []

        for document in documents:
            passages.extend(self.get_document_title_passage(document))

        return passages
    
    def _validate_search(self, query: str, top_k: int) -> None:
        if not query.strip():
            raise ValueError("Query cannot be empty")
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")
        if self.embeddings is None:
            raise RuntimeError("Index documents before searching")

    def _semantic_ranking(self, query: str) -> list[int]:
        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )[0]
        scores = self.embeddings @ query_embedding
        return np.argsort(scores)[::-1].tolist()

    def _bm25_ranking(self, query: str) -> list[int]:
        """Rank chunks by BM25, excluding chunks with no keyword matches."""
        query_terms = _tokenize(query)
        total_docs = len(self.tokenized_chunks)
        k1, b = 1.5, 0.75
        scores = np.zeros(total_docs, dtype=float)

        for index, tokens in enumerate(self.tokenized_chunks):
            term_counts = Counter(tokens)
            doc_length = len(tokens)

            for term in query_terms:
                frequency = term_counts.get(term, 0)
                if frequency == 0:
                    continue

                df = self.document_frequencies[term]
                idf = np.log(
                    1 + (total_docs - df + 0.5) / (df + 0.5)
                )
                denominator = frequency + k1 * (
                    1 - b
                    + b * doc_length
                    / max(self.average_document_length, 1)
                )
                scores[index] += (
                    idf * frequency * (k1 + 1) / denominator
                )

        # Only include chunks with at least one matching query term.
        matched_indices = np.flatnonzero(scores > 0)

        return sorted(
            matched_indices.tolist(),
            key=lambda index: scores[index],
            reverse=True,
        )

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        """Original semantic-only baseline."""
        self._validate_search(query, top_k)
        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )[0]
        scores = self.embeddings @ query_embedding
        ranked_indices = np.argsort(scores)[::-1][:top_k]

        return [
            {**self.chunks[int(i)], "score": float(scores[i])}
            for i in ranked_indices
        ]

    def search_hybrid(
        self,
        query: str,
        top_k: int = 3,
        candidate_k: int = 50,
        rrf_k: int = 60,
    ) -> list[dict]:
        """Fuse semantic and BM25 rankings using reciprocal rank fusion."""
        self._validate_search(query, top_k)

        if candidate_k <= 0 or rrf_k <= 0:
            raise ValueError("candidate_k and rrf_k must be positive")

        semantic = self._semantic_ranking(query)[:candidate_k]
        lexical = self._bm25_ranking(query)[:candidate_k]

        fused_scores: dict[int, float] = {}

        for ranking in (semantic, lexical):
            for rank, index in enumerate(ranking, start=1):
                fused_scores[index] = (
                    fused_scores.get(index, 0.0)
                    + 1.0 / (rrf_k + rank)
                )

        ranked_indices = sorted(
            fused_scores,
            key=fused_scores.get,
            reverse=True,
        )[:top_k]

        return [
            {
                **self.chunks[index],
                "score": fused_scores[index],
            }
            for index in ranked_indices
        ]