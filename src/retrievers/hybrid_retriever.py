from typing import List, Tuple, Literal, Any
from pathlib import Path
import pickle

from src.retrievers.base import CustomBaseRetriever
from src.retrievers.dense_retriever import FaissDenseRetriever
from src.retrievers.in_memory_dense_retriever import InMemoryDenseRetriever
from src.retrievers.sparse_retriever import BM25SparseRetriever
from src.utils.llm import SimpleLLM


class HybridRetriever(CustomBaseRetriever):
    """
    Hybrid retriever that mixes dense and sparse retrieval.
    It can also switch between dense, sparse, and hybrid strategies at any point.
    """

    def __init__(
        self,
        strategy: Literal["dense", "sparse", "hybrid"] = "hybrid",
        dense_backend: Literal["faiss", "in_memory"] = "faiss",
        faiss_index_type: str = "flat",  # Only for FAISS backend
        embedding_model: str = "all-MiniLM-L6-v2",
        embedding_provider: Literal[
            "sentence-transformers", "openai"
        ] = "sentence-transformers",
        bm25_k1: float = 1.5,
        bm25_b: float = 0.75,
        hybrid_alpha: float = 0.5,  # Weight for dense vs sparse (0=all sparse, 1=all dense)
    ):
        """
        Initialize hybrid retriever.

        Args:
            strategy: "dense", "sparse", or "hybrid"
            dense_backend: "faiss" or "in_memory" for dense retrieval
            embedding_model: Model name for embeddings
            embedding_provider: "sentence-transformers" or "openai"
            hybrid_alpha: Weight for combining dense and sparse scores
        """
        self.strategy = strategy
        self.dense_backend = dense_backend
        self.hybrid_alpha = hybrid_alpha
        self.chunks = []

        # Initialize dense retriever based on backend choice
        if strategy in ["dense", "hybrid"]:
            if dense_backend == "faiss":
                self.dense_retriever = FaissDenseRetriever(
                    embedding_model=embedding_model,
                    embedding_provider=embedding_provider,
                    index_type=faiss_index_type,
                )
            else:  # in_memory
                self.dense_retriever = InMemoryDenseRetriever(
                    embedding_model=embedding_model,
                    embedding_provider=embedding_provider,
                )
        else:
            self.dense_retriever = None

        # Initialize sparse retriever for sparse/hybrid strategies
        if strategy in ["sparse", "hybrid"]:
            self.sparse_retriever = BM25SparseRetriever(bm25_k1, bm25_b)
        else:
            self.sparse_retriever = None

        print("✓ HybridRetriever initialized")
        print(f"  Strategy: {strategy}")
        print(f"  Dense backend: {dense_backend if strategy != 'sparse' else 'N/A'}")
        print(f"  Hybrid alpha: {hybrid_alpha if strategy == 'hybrid' else 'N/A'}")
        print(f"  Dense weight: {hybrid_alpha:.2f}")
        print(f"  Sparse weight: {1 - hybrid_alpha:.2f}")

    def add_documents(self, documents: List[dict]) -> None:
        """Add documents to all active retrievers."""
        self.chunks = documents

        # Add to dense retriever
        if self.dense_retriever:
            self.dense_retriever.add_documents(documents)

        # Add to sparse retriever
        if self.sparse_retriever:
            self.sparse_retriever.add_documents(documents)

        print(f"✓ Added {len(documents)} documents to HybridRetriever")

    def search(self, query: str, k: int = 5) -> List[Tuple[dict, float]]:
        """
        Search using the configured strategy.

        Args:
            query: Search query
            k: Number of results

        Returns:
            List of (document, score) tuples
        """
        if self.strategy == "dense":
            return self._search_dense(query, k)
        elif self.strategy == "sparse":
            return self._search_sparse(query, k)
        elif self.strategy == "hybrid":
            return self._search_hybrid(query, k)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

    def _search_dense(self, query: str, k: int) -> List[Tuple[dict, float]]:
        """Dense search using embeddings."""
        return self.dense_retriever.search(query, k)

    def _search_sparse(self, query: str, k: int) -> List[Tuple[dict, float]]:
        """Sparse search using BM25."""
        return self.sparse_retriever.search(query, k)

    def _search_hybrid(self, query: str, k: int) -> List[Tuple[dict, float]]:
        """
        Hybrid search combining dense and sparse results.
        Uses Reciprocal Rank Fusion (RRF) to combine rankings.
        """
        # Get results from both retrievers
        dense_results = self.dense_retriever.search(query, k * 5)
        sparse_results = self.sparse_retriever.search(query, k * 5)

        # Create score dictionaries (doc_id -> score)
        dense_scores: dict[str, float] = {}
        sparse_scores: dict[str, float] = {}

        # Process dense results
        all_scores = [score for _, score in dense_results]
        min_s, max_s = min(all_scores), max(all_scores)
        for i, (doc, score) in enumerate(dense_results):
            doc_id = f"doc{doc['metadata'].get('doc_id')}_chunk{doc['metadata'].get('chunk_id')}"
            # normalize score
            dense_scores[doc_id] = score - min_s / (max_s - min_s + 1e-9)

        # Process sparse results
        all_scores = [score for _, score in sparse_results]
        min_s, max_s = min(all_scores), max(all_scores)
        for i, (doc, score) in enumerate(sparse_results):
            doc_id = f"doc{doc['metadata'].get('doc_id')}_chunk{doc['metadata'].get('chunk_id')}"
            # normalize score
            sparse_scores[doc_id] = score - min_s / (max_s - min_s + 1e-9)

        # Combine scores
        combined_scores = {}
        all_docs = {}

        # Add dense results
        for doc, _ in dense_results:
            doc_id = f"doc{doc['metadata'].get('doc_id')}_chunk{doc['metadata'].get('chunk_id')}"
            all_docs[doc_id] = doc
            dense_score = dense_scores.get(doc_id, 0)
            sparse_score = sparse_scores.get(doc_id, 0)
            combined_scores[doc_id] = (
                self.hybrid_alpha * dense_score + (1 - self.hybrid_alpha) * sparse_score
            )

        # Add sparse results not in dense
        for doc, _ in sparse_results:
            doc_id = f"doc{doc['metadata'].get('doc_id')}_chunk{doc['metadata'].get('chunk_id')}"
            if doc_id not in all_docs:
                all_docs[doc_id] = doc
                dense_score = dense_scores.get(doc_id, 0)
                sparse_score = sparse_scores.get(doc_id, 0)
                combined_scores[doc_id] = (
                    self.hybrid_alpha * dense_score
                    + (1 - self.hybrid_alpha) * sparse_score
                )

        # Sort by combined score
        sorted_results = sorted(
            combined_scores.items(), key=lambda x: x[1], reverse=True
        )[:k]

        # Return documents with scores
        return [(all_docs[doc_id], score) for doc_id, score in sorted_results]

    def switch_strategy(
        self, strategy: Literal["dense", "sparse", "hybrid"], cfg: dict[str, Any] = {}
    ):
        """Switch retrieval strategy on the fly."""
        old_strategy = self.strategy
        self.strategy = strategy

        # Initialize retrievers if needed
        if strategy in ["dense", "hybrid"] and not self.dense_retriever:
            if self.dense_backend == "faiss":
                self.dense_retriever = FaissDenseRetriever(
                    embedding_model=cfg.get("embedding_model", "all-MiniLM-L6-v2"),
                    embedding_provider=cfg.get(
                        "embedding_provider", "sentence-transformers"
                    ),
                    index_type=cfg.get("faiss_index_type", "flat"),
                )
            else:
                self.dense_retriever = InMemoryDenseRetriever(
                    embedding_model=cfg.get("embedding_model", "all-MiniLM-L6-v2"),
                    embedding_provider=cfg.get(
                        "embedding_provider", "sentence-transformers"
                    ),
                )

            # Re-add documents if we have them
            if self.chunks:
                self.dense_retriever.add_documents(self.chunks)

        if strategy in ["sparse", "hybrid"] and not self.sparse_retriever:
            self.sparse_retriever = BM25SparseRetriever(
                cfg.get("bm25_k1", 1.5), cfg.get("bm25_b", 0.75)
            )
            if self.chunks:
                self.sparse_retriever.add_documents(self.chunks)

        print(f"✓ Switched strategy from {old_strategy} to {strategy}")

    def set_hybrid_weight(self, alpha: float):
        """Adjust the weight for hybrid search (0=all sparse, 1=all dense)."""
        self.hybrid_alpha = alpha
        print(f"✓ Set hybrid alpha to {alpha}")

    def generate(
        self,
        query: str,
        results: list[tuple[dict, float]],
        llm_model: str = "gpt-4o-mini",
        temperature: float = 0.7,
    ) -> dict[str, Any]:
        """
        Generate answer based on query (full RAG).

        Args:
            query: User query
            results: List of (document, score) tuples
            llm_model: LLM model name
            temperature: Sampling temperature for LLM

        Returns:
            Dict with 'answer', 'query', 'sources'
        """
        llm = SimpleLLM(model=llm_model)
        response = llm.generate_llm_answer(query, results, temperature)

        return response

    def save(self, path: str) -> None:
        """Save retriever state."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save metadata
        metadata = {
            "strategy": self.strategy,
            "dense_backend": self.dense_backend,
            "hybrid_alpha": self.hybrid_alpha,
            "num_chunks": len(self.chunks),
        }

        with open(path / "hybrid_metadata.pkl", "wb") as f:
            pickle.dump(metadata, f)

        # Save sub-retrievers
        if self.dense_retriever:
            self.dense_retriever.save(str(path / "dense"))
        if self.sparse_retriever:
            self.sparse_retriever.save(str(path / "sparse"))

        print(f"✓ Saved HybridRetriever to {path}")

    def load(self, path: str) -> None:
        """Load retriever state."""
        path = Path(path)

        # Load metadata
        with open(path / "hybrid_metadata.pkl", "rb") as f:
            metadata = pickle.load(f)

        self.strategy = metadata["strategy"]
        self.dense_backend = metadata["dense_backend"]
        self.hybrid_alpha = metadata["hybrid_alpha"]

        # Load sub-retrievers
        if (path / "dense").exists():
            if self.dense_backend == "faiss":
                self.dense_retriever = FaissDenseRetriever()
            else:
                self.dense_retriever = InMemoryDenseRetriever()
            self.dense_retriever.load(str(path / "dense"))

        if (path / "sparse").exists():
            self.sparse_retriever = BM25SparseRetriever()
            self.sparse_retriever.load(str(path / "sparse"))

        print(f"✓ Loaded HybridRetriever from {path}")

    def __repr__(self) -> str:
        return (
            f"HybridRetriever(strategy={self.strategy}, backend={self.dense_backend})"
        )
