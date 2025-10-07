from langchain.schema import Document
from typing import List, Dict, Any
from src.data.chunk import chunk


def chunk_documents(
    docs: List[Document],
    chunking_strategy: str = "sentence",
    cfg: Dict[str, Any] | None = None,
) -> List[Document]:
    """
    Apply your custom chunking to LangChain Documents and return chunked Documents.

    Args:
        docs: List of LangChain Document objects.
        chunking_strategy: One of "fixed", "sentence", or "semantic".
        cfg: Optional chunking config, e.g. {"chunk_size": 2000, "overlap": 200}.

    Returns:
        List[Document]: Chunked LangChain Documents.
    """
    # Convert to the dict format expected by existing chunk() function
    raw_docs = [{"page_content": d.page_content, "metadata": d.metadata} for d in docs]

    chunked = chunk(chunking_strategy, raw_docs, cfg)

    # Convert back to LangChain Document objects
    chunked_docs = [
        Document(page_content=item["text"], metadata=item["metadata"])
        for item in chunked
    ]

    print(
        f"Chunked {len(docs)} docs → {len(chunked_docs)} chunks "
        f"using '{chunking_strategy}' strategy."
    )

    return chunked_docs
