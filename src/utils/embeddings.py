import time

from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Union, Literal
from openai import OpenAI


class EmbeddingModel:
    """Unified wrapper for embedding models (Sentence-Transformers and OpenAI)."""

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        provider: Literal["sentence-transformers", "openai"] = "sentence-transformers",
    ):
        """
        Initialize embedding model.

        Args:
            model_name: Name of the model
                - For sentence-transformers: "all-MiniLM-L6-v2", "all-mpnet-base-v2", etc.
                - For OpenAI: "text-embedding-3-small", "text-embedding-3-large", "text-embedding-ada-002"
            provider: Which provider to use
        """
        self.model_name = model_name
        self.provider = provider
        self.model = None
        self.dimension = None

        print(f"Loading embedding model: {model_name} (provider: {provider})")

        if provider == "sentence-transformers":
            self._init_sentence_transformers()
        elif provider == "openai":
            self._init_openai()
        else:
            raise ValueError(f"Unknown provider: {provider}")

        print(f"✓ Model loaded (dimension: {self.dimension})")

    def _init_sentence_transformers(self):
        """Initialize Sentence-Transformers model."""
        self.model = SentenceTransformer(self.model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()

    def _init_openai(self):
        """Initialize OpenAI client."""
        self.client = OpenAI()

        # Set dimension based on model
        dimension_map = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536,
        }

        self.dimension = dimension_map.get(self.model_name)
        if self.dimension is None:
            raise ValueError(f"Unknown OpenAI model: {self.model_name}")

    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 32,
        show_progress: bool = True,
    ) -> np.ndarray:
        """
        Encode texts into embeddings.

        Args:
            texts: Single text or list of texts
            batch_size: Batch size for encoding (used for sentence-transformers)
            show_progress: Show progress bar (used for sentence-transformers)

        Returns:
            Numpy array of embeddings
        """
        if isinstance(texts, str):
            texts = [texts]

        if self.provider == "sentence-transformers":
            return self._encode_sentence_transformers(texts, batch_size, show_progress)
        elif self.provider == "openai":
            return self._encode_openai(texts)

    def _encode_sentence_transformers(
        self, texts: List[str], batch_size: int, show_progress: bool
    ) -> np.ndarray:
        """Encode using Sentence-Transformers."""
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        )
        return embeddings.astype("float32")

    def _encode_openai(self, texts: List[str]) -> np.ndarray:
        """Encode using OpenAI API."""
        # OpenAI has a limit of ~8k tokens per request and 2048 texts per batch
        # For simplicity, we'll batch by number of texts
        max_batch_size = 1500
        all_embeddings = []

        for i in range(0, len(texts), max_batch_size):
            batch = texts[i : i + max_batch_size]
            response = self.client.embeddings.create(input=batch, model=self.model_name)

            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)
            time.sleep(0.1)

        return np.array(all_embeddings, dtype="float32")

    def get_dimension(self) -> int:
        """Get embedding dimension."""
        return self.dimension

    def __repr__(self) -> str:
        return f"EmbeddingModel(provider={self.provider}, model={self.model_name}, dim={self.dimension})"
