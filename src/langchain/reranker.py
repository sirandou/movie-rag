from typing import List, Any
from langchain.schema import Document, BaseRetriever

from langchain.retrievers import ContextualCompressionRetriever
from langchain_core.documents import BaseDocumentCompressor


class CustomCompressor(BaseDocumentCompressor):
    """
    Wrapper for our custom rerankers to be used as a DocumentCompressor in LangChain.
    Works with any reranker that has a `rerank(query: str, docs: List[dict], top_k: int) -> List[Tuple[dict, float]]` method.
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

        # Convert LangChain Documents to our format
        custom_docs = [
            {"text": doc.page_content, "metadata": doc.metadata} for doc in documents
        ]

        # Rerank using our reranker
        reranked = self.reranker.rerank(query, custom_docs, top_k=self.top_k)

        # Convert back to LangChain Documents with scores
        compressed = []
        for doc, score in reranked:
            lc_doc = Document(
                page_content=doc["text"],
                metadata={
                    **doc[
                        "metadata"
                    ],  # original score might be here if retriever was custom
                    "rerank_score": float(score),
                },
            )
            compressed.append(lc_doc)

        return compressed


def create_reranking_retriever(
    base_retriever: BaseRetriever,
    top_k: int = 5,
    cfg: dict = {},
) -> ContextualCompressionRetriever:
    """
    Factory function to create reranking retriever.

    Args:
        base_retriever: base retriever instance
        top_k: top K to return after reranking
        cfg: config dict for reranker (e.g. re-ranker type, model name)

    Returns:
        ContextualCompressionRetriever with reranking
    """
    reranker_type = cfg.get("type", "cross-encoder")

    if reranker_type == "cross-encoder":
        from src.retrievers.reranker import CrossEncoderReranker

        model_name = cfg.get("model_name", "cross-encoder/ms-marco-MiniLM-L-6-v2")
        reranker = CrossEncoderReranker(model_name=model_name)
        compressor = CustomCompressor(reranker, top_k=top_k)

        print(f"✓ Created cross-encoder reranking retriever (top_k={top_k})")

    elif reranker_type == "cohere":
        from langchain_cohere import CohereRerank

        compressor = CohereRerank(top_n=top_k)
        print(f"✓ Created Cohere reranking retriever (top_k={top_k})")

    elif reranker_type == "llm":
        from src.retrievers.reranker import LLMReranker

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
