from typing import Literal, List, Tuple, Any

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from src.retrievers.base import CustomBaseRetriever
from src.utils.embeddings import EmbeddingModel
from src.utils.llm import SimpleLLM


class InMemoryDenseRetriever(CustomBaseRetriever):
    """Simple in-memory vector store using cosine similarity."""

    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        embedding_provider: Literal[
            "sentence-transformers", "openai"
        ] = "sentence-transformers",
    ):
        self.embedding_model = EmbeddingModel(
            model_name=embedding_model, provider=embedding_provider
        )
        self.chunks: list[dict] = []
        self.embeddings = None

    def add_documents(self, documents: List[dict]) -> None:
        """
        Add chunks and generate embeddings.

        Args:
            documents: List of dicts with 'text' and 'metadata'
        """
        if not documents:
            raise ValueError("No documents provided")

        # Extract texts
        texts = [doc["text"] for doc in documents]

        # Generate embeddings
        print(f"Generating embeddings for {len(texts)} documents...")
        self.embeddings = self.embedding_model.encode(texts, show_progress=True)
        self.chunks = documents

        print(f"✓ Added {len(documents)} chunks to in-memory store")

    def search(self, query: str, k: int = 5) -> List[Tuple[dict, float]]:
        """
        Search for relevant documents.

        Args:
            query: Search query
            k: Number of results to return
            **kwargs: Additional retriever-specific parameters

        Returns:
            List of (document/chunk, score) tuples
        """
        # Embed query
        query_embedding = self.embedding_model.encode(query, show_progress=False)

        # Compute cosine similarities
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]

        # Get top k
        top_k_indices = np.argsort(similarities)[-k:][::-1]

        # Return results
        results = []
        for idx in top_k_indices:
            results.append((self.chunks[idx], float(similarities[idx])))

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
        pass

    def load(self, path: str) -> None:
        pass
