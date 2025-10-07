from typing import List
from langchain.schema import BaseRetriever, Document
from langchain.callbacks.manager import CallbackManagerForRetrieverRun

from src.retrievers.base import CustomBaseRetriever
from src.retrievers.visual_retriever import VisualRetriever


class TextRetrieverWrapper(BaseRetriever):
    """
    Universal wrapper for custom text retrievers to make them langchain retrievers.
    Works with all retrievers in src/retrievers except for VisualRetriever.
    """

    retriever: CustomBaseRetriever
    k: int = 5

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, retriever: CustomBaseRetriever, k: int = 5):
        """
        Wrap any custom retriever for LangChain compatibility.

        Args:
            retriever (CustomBaseRetriever): An already initialized custom retriever instance
            k: Number of results
        """
        super().__init__(retriever=retriever, k=k)

    def add_documents(self, documents: List[Document]) -> None:
        """
        Add LangChain Documents to custom retriever.
        Handles format conversion: LangChain → Custom.

        Args:
            documents: LangChain Documents (from chunker)
        """
        # Convert LangChain Documents → Custom format
        custom_chunks = []
        for doc in documents:
            custom_chunk = {"text": doc.page_content, "metadata": doc.metadata}
            custom_chunks.append(custom_chunk)

        # Add to underlying custom retriever
        self.retriever.add_documents(custom_chunks)

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun = None
    ) -> List[Document]:
        """Get top k documents from custom retriever and change them into langchain documents."""
        results = self.retriever.search(query, k=self.k)

        # Convert to LangChain Documents
        documents = []
        for doc, score in results:
            lc_doc = Document(
                page_content=doc["text"],
                metadata={**doc["metadata"], "score": score},
            )
            documents.append(lc_doc)

        return documents


class VisualRetrieverWrapper(BaseRetriever):
    """
    Wrapper for VisualRetriever (different document structure).
    """

    retriever: VisualRetriever
    k: int = 5

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, retriever: VisualRetriever, k: int = 5):
        """
        Wrap custom visual retriever for LangChain compatibility.

        Args:
            retriever (VisualRetriever): An already initialized visual retriever instance
            k: Number of results
        """
        super().__init__(retriever=retriever, k=k)

    def add_documents(self, documents: list[dict]) -> None:
        """
        Add custom documents to visual retriever.
        Important: documents must be the outcome of custom visual document creator
        """
        self.retriever.add_documents(documents)

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun = None
    ) -> List[Document]:
        """Get poster documents."""
        results = self.retriever.search(query, k=self.k)

        documents = []
        for doc, score in results:
            lc_doc = Document(
                page_content=doc["text_content"],
                metadata={
                    **doc["metadata"],
                    "poster_path": doc["poster_path"],
                    "score": score,
                },
            )
            documents.append(lc_doc)

        return documents
