from langchain.schema import Document, BaseRetriever

from langchain.retrievers import ContextualCompressionRetriever
from langchain_core.documents import BaseDocumentCompressor

from typing import List, Tuple, Any

from openai import OpenAIError
from sentence_transformers import CrossEncoder


class CrossEncoderReranker:
    """
    Cross-encoder reranker working directly with LangChain Documents using Huggingface model.
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
        self, query: str, documents: List[Document], top_k: int = None
    ) -> List[Tuple[Document, float]]:
        """
        Rerank Langchain documents using cross-encoder.

        Args:
            query: Search query
            documents: List of Langchain Documents
            top_k: Return top K after reranking (None = all)

        Returns:
            List of (Document, rerank_score) tuples, sorted by score
        """
        if not documents:
            return []

        # Prepare query-document pairs
        pairs = [[query, doc.page_content] for doc in documents]

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
    LLM-based reranker using relevance prompting working directly with LangChain Documents.
    More expensive but can handle complex relevance.
    """

    def __init__(self, llm_model: str = "gpt-4o-mini"):
        """Initialize LLM reranker."""
        from openai import OpenAI

        self.client = OpenAI()
        self.model = llm_model
        print(f"✓ LLMReranker initialized: {llm_model}")

    def rerank(
        self, query: str, documents: List[Document], top_k: int = None
    ) -> List[Tuple[Document, float]]:
        """
        Rerank Langchain documents using LLM relevance scoring.

        Args:
            query: Search query
            documents: List of Langchain Documents
            top_k: Return top K after reranking (None = all)

        Returns:
            List of (Document, rerank_score) tuples, sorted by score
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

    def _score_document(self, query: str, document: Document) -> float:
        """Score a single Langchain document for relevance (0-1)."""
        text = document.page_content[:500]  # Limit length

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


class CustomCompressor(BaseDocumentCompressor):
    """
    LangChain-compatible compressor using our custom rerankers.
    Works directly with LangChain Documents - no conversion needed.
    """

    # Declare fields as class attributes (Pydantic requirement)
    reranker: Any
    top_k: int = 5

    class Config:
        arbitrary_types_allowed = True  # Allow custom types like our reranker

    def __init__(self, reranker, top_k: int = 5):
        """
        Initialize compressor.

        Args:
            reranker: Custom reranker instance
            top_k: Number of documents to return after reranking
        """
        # Initialize Pydantic model with fields
        super().__init__(reranker=reranker, top_k=top_k)

    def compress_documents(
        self, documents: List[Document], query: str, callbacks=None
    ) -> List[Document]:
        """
        Rerank and filter documents.

        Args:
            documents: Retrieved documents
            query: User query
            callbacks: LangChain callbacks

        Returns:
            Reranked and filtered documents
        """
        if not documents:
            return []

        # Rerank using our reranker
        reranked = self.reranker.rerank(query, documents, top_k=self.top_k)

        # Add rerank scores to metadata
        compressed = []
        for doc, score in reranked:
            # Create new document with updated metadata
            new_doc = Document(
                page_content=doc.page_content,
                metadata={
                    **doc.metadata,
                    "rerank_score": float(score),
                },
            )
            compressed.append(new_doc)

        return compressed


def create_reranking_retriever(
    base_retriever: BaseRetriever,
    top_k: int = 5,
    cfg: dict = {},
) -> ContextualCompressionRetriever:
    """
    Factory function to create reranking retriever, a wrapper around a base retriever.

    Args:
        base_retriever: base retriever instance
        top_k: top K to return after reranking
        cfg: config dict for reranker (e.g. re-ranker type, model name)

    Returns:
        ContextualCompressionRetriever with reranking
    """
    reranker_type = cfg.get("type", "cross-encoder")

    if reranker_type == "cross-encoder":
        model_name = cfg.get("model_name", "cross-encoder/ms-marco-MiniLM-L-6-v2")
        reranker = CrossEncoderReranker(model_name=model_name)
        compressor = CustomCompressor(reranker, top_k=top_k)

        print(f"✓ Created cross-encoder reranking retriever (top_k={top_k})")

    elif reranker_type == "cohere":
        from langchain_cohere import CohereRerank

        compressor = CohereRerank(top_n=top_k)
        print(f"✓ Created Cohere reranking retriever (top_k={top_k})")

    elif reranker_type == "llm":
        llm_model = cfg.get("llm_model", "gpt-4o-mini")
        reranker = LLMReranker(llm_model=llm_model)
        compressor = CustomCompressor(reranker, top_k=top_k)

        print(f"✓ Created LLM reranking retriever (top_k={top_k})")

    else:
        raise ValueError(f"Unknown reranker_type: {reranker_type}")

    # Create compression retriever
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=base_retriever
    )

    return compression_retriever
