from abc import ABC, abstractmethod
from typing import List, Tuple


class CustomBaseRetriever(ABC):
    """Abstract base class for all retrievers."""

    @abstractmethod
    def add_documents(self, documents: List[dict]) -> None:
        """Add documents to the retriever."""
        pass

    @abstractmethod
    def search(self, query: str, k: int = 5) -> List[Tuple[dict, float]]:
        """
        Search for relevant documents.

        Args:
            query: Search query
            k: Number of results to return

        Returns:
            List of (document/chunk, score) tuples
        """
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        """Save retriever state to disk."""
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        """Load retriever state from disk."""
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
