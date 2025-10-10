from typing import List, Tuple

from openai import OpenAIError
from sentence_transformers import CrossEncoder


class CrossEncoderReranker:
    """
    Cross-encoder reranker using sentence-transformers.
    Scores query-document pairs for relevance.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """
        Initialize cross-encoder reranker.

        Args:
            model_name: HuggingFace cross-encoder model
        """
        self.model_name = model_name
        self.model = CrossEncoder(model_name)
        print(f"✓ CrossEncoderReranker loaded: {model_name}")

    def rerank(
        self, query: str, documents: List[dict], top_k: int = None
    ) -> List[Tuple[dict, float]]:
        """
        Rerank documents using cross-encoder.

        Args:
            query: Search query
            documents: List of documents (with 'text' field)
            top_k: Return top K after reranking (None = all)

        Returns:
            List of (document, rerank_score) tuples, sorted by score
        """
        if not documents:
            return []

        # Prepare query-document pairs
        pairs = [[query, doc.get("text", "")] for doc in documents]

        # Score all pairs
        scores = self.model.predict(pairs)

        # Combine documents with scores
        doc_scores = list(zip(documents, scores))

        # Sort by score (descending)
        doc_scores.sort(key=lambda x: x[1], reverse=True)

        # Return top_k
        if top_k:
            return doc_scores[:top_k]
        return doc_scores


class LLMReranker:
    """
    LLM-based reranker using relevance prompting.
    More expensive but can handle complex relevance.
    """

    def __init__(self, llm_model: str = "gpt-4o-mini"):
        """Initialize LLM reranker."""
        from openai import OpenAI

        self.client = OpenAI()
        self.model = llm_model
        print(f"✓ LLMReranker initialized: {llm_model}")

    def rerank(
        self, query: str, documents: List[dict], top_k: int = None
    ) -> List[Tuple[dict, float]]:
        """
        Rerank documents using LLM relevance scoring.

        Args:
            query: Search query
            documents: List of documents (with 'text' field)
            top_k: Return top K after reranking (None = all)

        Returns:
            List of (document, rerank_score) tuples, sorted by score
        """
        if not documents:
            return []

        scores = []
        for doc in documents:
            score = self._score_document(query, doc)
            scores.append(score)

        # Combine and sort
        doc_scores = list(zip(documents, scores))
        doc_scores.sort(key=lambda x: x[1], reverse=True)

        if top_k:
            return doc_scores[:top_k]
        return doc_scores

    def _score_document(self, query: str, document: dict) -> float:
        """Score a single document for relevance (0-10)."""
        text = document.get("text", "")[:500]  # Limit length

        prompt = f"""Rate the relevance of this document to the query on a scale of 0-10.

Query: {query}

Document: {text}

Respond with ONLY a number between 0 and 10, where:
- 0 = completely irrelevant
- 10 = perfectly relevant

Relevance score:"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=5,
            )

            score_text = response.choices[0].message.content.strip()
            score = float(score_text)
            return max(0, min(10, score)) / 10  # Normalize to 0-1

        except (OpenAIError, ValueError, IndexError, AttributeError) as e:
            print("Scoring error:", e)
            return 0.5
