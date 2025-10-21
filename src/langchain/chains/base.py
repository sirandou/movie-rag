from abc import ABC, abstractmethod
from typing import Any


class BaseChain(ABC):
    """
    Abstract base class for a flexible RAG chain.
    Subclasses must implement core pipeline steps.
    Important: self.retriever must be set in build(). self.llm must be set in __init__().
    """

    def __init__(self):
        self.llm = None
        self.retriever = None

    @abstractmethod
    def build(self):
        """
        Build the RAG pipeline. Must set self.retriever (can be None initially).
        """
        pass

    @abstractmethod
    def query(self, question: str, return_sources: bool = True) -> dict[str, Any]:
        """
        Query the RAG chain.

        Args:
            question: User question
            return_sources: If True, include source documents in output

        Returns:
            Dict with 'answer' (str), 'question' (str), optional 'sources' (list of dicts with 'content' and
            'metadata'), and optional other keys.
        """
        pass
