from rank_bm25 import BM25Okapi
import pickle
from typing import List, Tuple, Any
from pathlib import Path
import string

from src.retrievers.base import BaseRetriever
from src.utils.llm import SimpleLLM


class BM25SparseRetriever(BaseRetriever):
    """BM25-based sparse retriever for keyword search."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        """
        Initialize sparse retriever.

        Args:
            k1: BM25 k1 parameter (term frequency saturation)
            b: BM25 b parameter (length normalization)
        """
        self.k1 = k1
        self.b = b
        self.bm25 = None
        self.chunks: list[dict] = []
        self.tokenized_corpus: list[list[str]] = []

        print(f"✓ SparseRetriever initialized (k1={k1}, b={b})")

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization (lowercase + remove punctuation + split + remove short)."""
        return [
            t
            for t in text.lower()
            .replace("\n", " ")
            .translate(str.maketrans("", "", string.punctuation))
            .split()
            if len(t) > 2
        ]

    def add_documents(self, documents: List[dict]) -> None:
        """
        Add documents to BM25 index.

        Args:
            documents: List of dicts with 'text' and 'metadata'
        """
        if not documents:
            raise ValueError("No documents provided")

        self.chunks = documents

        # Tokenize all documents
        print(f"Tokenizing {len(documents)} documents...")
        self.tokenized_corpus = [self._tokenize(doc["text"]) for doc in documents]

        # Create BM25 index
        print("Building BM25 index...")
        self.bm25 = BM25Okapi(self.tokenized_corpus, k1=self.k1, b=self.b)

        print(f"✓ Added {len(documents)} documents to BM25 index")

    def search(self, query: str, k: int = 5) -> List[Tuple[dict, float]]:
        """
        Search for relevant documents using BM25.

        Args:
            query: Search query
            k: Number of results

        Returns:
            List of (document, score) tuples
        """
        if self.bm25 is None:
            raise ValueError("No index created. Call add_documents() first.")

        # Tokenize query
        query_tokens = self._tokenize(query)

        # Get BM25 scores for all documents
        scores = self.bm25.get_scores(query_tokens)

        # Get top-k indices
        top_k_indices = scores.argsort()[-k:][::-1]

        # Prepare results
        results = []
        for idx in top_k_indices:
            doc = self.chunks[idx]
            score = float(scores[idx])
            results.append((doc, score))

        return results

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
            k: Number of chunks to retrieve
            llm_model: LLM model name

        Returns:
            Dict with 'answer', 'query', 'sources'
        """
        llm = SimpleLLM(model=llm_model)
        response = llm.generate_llm_answer(query, results, temperature)

        return response

    def save(self, path: str) -> None:
        """Save BM25 index to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        save_data = {
            "bm25": self.bm25,
            "chunks": self.chunks,
            "tokenized_corpus": self.tokenized_corpus,
            "k1": self.k1,
            "b": self.b,
        }

        with open(path / "bm25.pkl", "wb") as f:
            pickle.dump(save_data, f)

        print(f"✓ Saved SparseRetriever to {path}")

    def load(self, path: str) -> None:
        """Load BM25 index from disk."""
        path = Path(path)

        with open(path / "bm25.pkl", "rb") as f:
            save_data = pickle.load(f)

        self.bm25 = save_data["bm25"]
        self.chunks = save_data["chunks"]
        self.tokenized_corpus = save_data["tokenized_corpus"]
        self.k1 = save_data["k1"]
        self.b = save_data["b"]

        print(f"✓ Loaded SparseRetriever from {path}")
        print(f"  Documents: {len(self.chunks)}")

    def __repr__(self) -> str:
        num_docs = len(self.chunks) if self.chunks else 0
        return f"BM25SparseRetriever(k1={self.k1}, b={self.b}, docs={num_docs})"
