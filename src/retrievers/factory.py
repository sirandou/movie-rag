from typing import Any

from src.retrievers.dense_retriever import FaissDenseRetriever
from src.retrievers.sparse_retriever import BM25SparseRetriever
from src.retrievers.hybrid_retriever import HybridRetriever
from src.retrievers.base import CustomBaseRetriever


def create_text_retriever(cfg: dict[str, Any] | None = None) -> CustomBaseRetriever:
    """
    Create text retriever from config dictionary.

    Args:
        cfg: Configuration dict with 'type' and component configs

    Returns:
        Retriever instance
    """
    if cfg is None:
        cfg = {}

    retriever_type = cfg.get("type", "hybrid")

    if retriever_type == "dense":
        return FaissDenseRetriever(
            embedding_model=cfg.get("embedding_model", "text-embedding-3-small"),
            embedding_provider=cfg.get("embedding_provider", "openai"),
            index_type=cfg.get("index_type", "flat"),
        )

    elif retriever_type == "sparse":
        return BM25SparseRetriever(k1=cfg.get("k1", 1.5), b=cfg.get("b", 0.75))

    elif retriever_type == "hybrid":
        return HybridRetriever(
            strategy=cfg.get("hybrid_strategy", "hybrid"),
            dense_backend=cfg.get("dense_backend", "faiss"),
            faiss_index_type=cfg.get("faiss_index_type", "flat"),
            embedding_model=cfg.get("embedding_model", "text-embedding-3-small"),
            embedding_provider=cfg.get("embedding_provider", "openai"),
            bm25_k1=cfg.get("bm25_k1", 1.5),
            bm25_b=cfg.get("bm25_b", 0.75),
            hybrid_alpha=cfg.get("hybrid_alpha", 0.5),
        )

    else:
        raise ValueError(f"Unknown retriever type: {retriever_type}")
