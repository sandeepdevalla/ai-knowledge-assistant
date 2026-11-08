import boto3
import re


MODEL_ID = "amazon.nova-micro-v1:0"
AWS_REGION = "us-east-1"


class AnswerGenerator:
    def __init__(self):
        self.client = boto3.client(
            "bedrock-runtime",
            region_name=AWS_REGION,
        )

    @staticmethod
    def validate_citations(answer: str, source_count: int) -> str:
        """Remove citation references that don't map to supplied evidence."""

        def replace_citation(match: re.Match) -> str:
            citation_number = int(match.group(1))

            if 1 <= citation_number <= source_count:
                return match.group(0)

            return ""

        return re.sub(r"\[(\d+)\]", replace_citation, answer)

    def generate(self, question: str, passages: list[dict]) -> dict:
        if not question.strip():
            raise ValueError("Question cannot be empty")

        if not passages:
            return {
                "answer": "I couldn't find sufficient evidence in the indexed documents to answer this question.",
                "sources": [],
            }

        evidence = []
        sources = []

        for passage in passages:
            source = {
                "document": passage["document"],
                "page": passage["page"],
            }
            sources.append(source)
            evidence.append(
                f"[{len(evidence) + 1}] "
                f"Document: {source['document']}, "
                f"page: {source['page']}\n"
                f"{passage['text']}"
            )

        prompt = f"""
Answer the question using only the evidence below.
If the evidence does not support an answer, say so.
Do not invent facts or citations.
Cite supporting evidence using [1], [2], etc.

Evidence:
{chr(10).join(evidence)}

Question: {question}
"""

        response = self.client.converse(
            modelId=MODEL_ID,
            messages=[
                {
                    "role": "user",
                    "content": [{"text": prompt}],
                }
            ],
            inferenceConfig={
                "maxTokens": 500,
                "temperature": 0.1,
            },
        )

        answer = response["output"]["message"]["content"][0]["text"]

        answer = self.validate_citations(
            answer,
            source_count=len(sources),
        )

        # Extract valid citation IDs in the order they appear.
        citation_ids = list(
            dict.fromkeys(
                int(match)
                for match in re.findall(r"\[(\d+)\]", answer)
            )
        )

        # Return only sources referenced by valid citations.
        cited_sources = [
            sources[citation_id - 1]
            for citation_id in citation_ids
            if 1 <= citation_id <= len(sources)
        ]

        return {
            "answer": answer,
            "sources": cited_sources,
            "model": MODEL_ID,
            "usage": response.get("usage", {}),
        }
