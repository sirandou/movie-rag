import faiss
import pickle
from typing import List, Tuple, Literal, Any
from pathlib import Path

from src.retrievers.base import BaseRetriever
from src.utils.embeddings import EmbeddingModel
from src.utils.llm import SimpleLLM


class FaissDenseRetriever(BaseRetriever):
    """FAISS-based dense retriever using embeddings."""

    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        embedding_provider: Literal[
            "sentence-transformers", "openai"
        ] = "sentence-transformers",
        index_type: str = "flat",
    ):
        """
        Initialize dense retriever.

        Args:
            embedding_model: Name of embedding model
            embedding_provider: "sentence-transformers" or "openai"
            index_type: Type of FAISS index ("flat" or "ivf")
        """
        self.embedding_model = EmbeddingModel(
            model_name=embedding_model, provider=embedding_provider
        )
        self.index_type = index_type
        self.index = None
        self.chunks: list[dict] = []

        print(f"✓ FaissDenseRetriever initialized (index_type={index_type})")

    def add_documents(self, documents: List[dict]) -> None:
        """
        Add documents to FAISS index.

        Args:
            documents: List of dicts with 'text' and 'metadata'
        """
        if not documents:
            raise ValueError("No documents provided")

        # Extract texts
        texts = [doc["text"] for doc in documents]

        # Generate embeddings
        print(f"Generating embeddings for {len(texts)} documents...")
        embeddings = self.embedding_model.encode(texts, show_progress=True)
        print("Embeddings generated")

        # Create FAISS index
        dimension = self.embedding_model.get_dimension()

        print("Saving index...")
        if self.index_type == "flat":
            self.index = faiss.IndexFlatL2(dimension)
        else:
            raise ValueError(f"Unknown index_type: {self.index_type}")

        # Add embeddings to index
        self.index.add(embeddings)

        # Store documents
        self.chunks = documents

        print(f"✓ Added {len(documents)} documents to FAISS index")
        print(f"  Index size: {self.index.ntotal}")

    def search(self, query: str, k: int = 5) -> List[Tuple[dict, float]]:
        """
        Search for similar documents.

        Args:
            query: Search query
            k: Number of results

        Returns:
            List of (document, score) tuples. Here score is negative L2 distance (the lower the closer).
        """
        if self.index is None:
            raise ValueError("No index created. Call add_documents() first.")

        # Embed query
        query_embedding = self.embedding_model.encode(query, show_progress=False)

        # Search
        distances, indices = self.index.search(query_embedding, k)

        # Prepare results
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            doc = self.chunks[idx]
            score = -float(distance)
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
        """Save index and documents to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save FAISS index
        index_path = path / "faiss.index"
        faiss.write_index(self.index, str(index_path))

        # Save documents
        docs_path = path / "documents.pkl"
        with open(docs_path, "wb") as f:
            pickle.dump(self.chunks, f)

        # Save metadata
        meta_path = path / "metadata.pkl"
        metadata = {
            "model_name": self.embedding_model.model_name,
            "provider": self.embedding_model.provider,
            "index_type": self.index_type,
            "num_docs": len(self.chunks),
        }
        with open(meta_path, "wb") as f:
            pickle.dump(metadata, f)

        print(f"✓ Saved FaissDenseRetriever to {path}")

    def load(self, path: str) -> None:
        """Load index and documents from disk."""
        path = Path(path)

        # Load metadata
        meta_path = path / "metadata.pkl"
        with open(meta_path, "rb") as f:
            metadata = pickle.load(f)

        # Reinitialize embedding model
        self.embedding_model = EmbeddingModel(
            model_name=metadata["model_name"], provider=metadata["provider"]
        )
        self.index_type = metadata["index_type"]

        # Load FAISS index
        index_path = path / "faiss.index"
        self.index = faiss.read_index(str(index_path))

        # Load documents
        docs_path = path / "documents.pkl"
        with open(docs_path, "rb") as f:
            self.chunks = pickle.load(f)

        print(f"✓ Loaded FaissDenseRetriever from {path}")
        print(f"  Documents: {len(self.chunks)}")

    def __repr__(self) -> str:
        num_docs = len(self.chunks) if self.chunks else 0
        return f"FaissDenseRetriever(model={self.embedding_model.model_name}, docs={num_docs})"
