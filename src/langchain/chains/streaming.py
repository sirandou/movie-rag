from typing import Any, Set
from langchain.schema import Document
import re

from langchain_openai import ChatOpenAI

from src.langchain.chains.base import BaseChain
from src.langchain.prompts import STREAM_PROMPT


class CitationStreamingWrapper:
    """
    Wraps a LangChain chain for clean streaming with citations at the end.

    Example Output:
        Question: What makes Inception great?

        Answer: Inception is great [1] with stunning visuals [2]...

        ============================================================
        Citations:
        ============================================================
        [1] Inception, Source: Review, Preview: Inception features...
        [2] Inception, Source: Review, Preview: The visual effects...
    """

    def __init__(
        self,
        chain: BaseChain,
        citation_pattern: str = r"\[(\d+)\]",
        verbose: bool = True,
    ) -> None:
        """
        Args:
            chain: BaseChain instance
            citation_pattern: Regex to detect citations (default: [1], [2])
            verbose: If True, print to console
        """
        # Replace original qa chain with streaming version
        self.chain = chain
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0, streaming=True)

        self.citation_pattern = re.compile(citation_pattern)
        self.verbose = verbose

    def invoke(self, question: str, return_sources: bool = True) -> dict[str, Any]:
        """Non-streaming invoke (pass-through to original chain)."""
        return self.chain.query(question, return_sources=return_sources)

    def stream(self, question: str, **kwargs) -> dict[str, Any]:
        """Stream the chain output with clean citation tracking."""
        self.cited_indices: Set[int] = set()  # Track cited indices

        # Print question
        if self.verbose:
            print(f"Question: {question}\n")
            print("Answer: ", end="", flush=True)

        full_answer = []
        buffer = ""  # Buffer to detect citations across chunk boundaries

        # Stream the chain (manual chain to control numbering context)
        docs = self.chain.retriever.get_relevant_documents(question)
        context = "\n\n".join(
            [f"[{i + 1}] {doc.page_content}" for i, doc in enumerate(docs)]
        )

        for chunk in self.llm.stream(
            STREAM_PROMPT.format(context=context, question=question)
        ):
            # Extract token
            token = chunk.content
            full_answer.append(token)
            buffer += token

            # Track citations in buffer
            self._track_citations(buffer)

            # Print token
            if self.verbose:
                print(token, end="", flush=True)

            # Keep only recent buffer (for split citations like [1])
            if len(buffer) > 10:
                buffer = buffer[-5:]

        # Print citations at the end
        if self.verbose:
            print("\n")  # Newline after answer
            self._print_citations(docs)

        # Return result
        return {
            "answer": "".join(full_answer),
            "question": question,
            "citations_used": sorted(list(self.cited_indices)),
            "sources": [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                }
                for doc in docs
            ],
        }

    def _track_citations(self, text: str) -> None:
        """Track which citations appear in the text."""
        matches = self.citation_pattern.finditer(text)
        for match in matches:
            cite_num = int(match.group(1))
            self.cited_indices.add(cite_num)

    def _print_citations(self, docs: list[Document]) -> None:
        """Print clean citation list at the end."""
        print("=" * 60)
        print("Citations:")
        print("=" * 60)

        # Show only cited sources (in citation order)
        for cite_num in sorted(self.cited_indices):
            if cite_num <= len(docs):
                doc = docs[cite_num - 1]
                self._print_single_citation(cite_num, doc)

    def _print_single_citation(self, cite_num: int, doc: Document) -> None:
        """Print a single citation in clean format."""
        # Extract metadata
        title = doc.metadata.get("movie_title", "Unknown")
        source_type = doc.metadata.get("source", "Unknown")  # e.g., 'Review', 'Plot'

        # Get preview (first 350 chars)
        preview = doc.page_content[:1000]
        if len(doc.page_content) > 1000:
            preview += "..."

        # Print citation
        print(f"\n[{cite_num}] {title}, Source: {source_type}, Preview: {preview}")
