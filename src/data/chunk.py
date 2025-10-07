"""Chunking strategies for movie reviews and plots."""

from typing import Dict, List, Any

import tiktoken
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class MovieReviewChunker:
    def __init__(self):
        """Initialize chunking tools"""
        # For token counting
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

        # For semantic chunking
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")

    def chunk_fixed_size_tokens(
        self, text: str, chunk_size: int = 200, overlap: int = 50, verbose: bool = False
    ) -> List[Dict]:
        """
        Fixed-size chunking using tiktoken
        """
        chunks = []

        # Tokenize the entire text
        tokens = self.tokenizer.encode(text)

        # Create chunks with overlap
        start = 0
        chunk_id = 0

        while start < len(tokens):
            end = min(start + chunk_size, len(tokens))

            # Get the chunk tokens
            chunk_tokens = tokens[start:end]

            # Decode back to text
            chunk_text = self.tokenizer.decode(chunk_tokens)

            chunks.append(chunk_text)

            # Move forward with overlap
            start += chunk_size - overlap
            chunk_id += 1

        if verbose:
            print(f"Created {len(chunks)} fixed-size chunks")
        return chunks

    def chunk_by_sentences(
        self, text: str, sentences_per_chunk: int = 5, verbose: bool = False
    ) -> List[Dict]:
        """
        Chunk by sentence boundaries (better for reviews)
        """
        chunks = []

        # Split into sentences (simple approach)
        # For reviews, consider each review as a sentence. metadata will be a separate chunk
        if "Review:" in text:
            reviews = text.split("Review: ")

            # Group reviews into chunks
            chunks.append(reviews[0].strip())
            for i in range(1, len(reviews), sentences_per_chunk):
                chunk_reviews = reviews[i : i + sentences_per_chunk]
                chunk_text = "Review: " + "Review: ".join(chunk_reviews)

                chunks.append(chunk_text.strip())
        else:
            # Standard sentence splitting
            sentences = sent_tokenize(text)
            sentences = [s.strip() for s in sentences if s.strip()]

            for i in range(0, len(sentences), sentences_per_chunk):
                chunk_sentences = sentences[i : i + sentences_per_chunk]
                chunk_text = " ".join(chunk_sentences)

                chunks.append(chunk_text)

        if verbose:
            print(f"Created {len(chunks)} sentence-based chunks")
        return chunks

    def chunk_by_semantic_similarity(
        self,
        text: str,
        threshold: float = 0.7,
        min_chunk_size: int = 100,
        verbose: bool = False,
    ) -> List[Dict]:
        """
        Semantic chunking using sentence-transformers
        """
        chunks = []

        # Split into sentences first
        if "Review: " in text:
            segments = text.split("Review: ")
            segments[1:] = ["Review: " + s for s in segments[1:]]
        else:
            segments = sent_tokenize(text)

        segments = [s.strip() for s in segments if len(s.strip()) > 20]

        if not segments:
            return [{"chunk_id": 0, "text": text}]

        # Embed all segments
        embeddings = self.embedder.encode(segments)

        # Group similar segments
        current_chunk = [segments[0]]
        current_embedding = embeddings[0]
        chunk_id = 0

        for i in range(1, len(segments)):
            # Calculate similarity with current chunk's centroid
            similarity = cosine_similarity([current_embedding], [embeddings[i]])[0][0]

            # Check if we should add to current chunk or start new one
            current_chunk_text = " ".join(current_chunk)

            if similarity > threshold and len(current_chunk_text) < 2000:
                # Add to current chunk
                current_chunk.append(segments[i])
                # Update centroid (simple average)
                current_embedding = (current_embedding + embeddings[i]) / 2
            else:
                # Save current chunk and start new one
                chunks.append(" ".join(current_chunk))

                current_chunk = [segments[i]]
                current_embedding = embeddings[i]
                chunk_id += 1

        # Don't forget the last chunk
        if current_chunk:
            chunks.append(" ".join(current_chunk))

        if verbose:
            print(f"Created {len(chunks)} semantic chunks")
        return chunks


def chunk(
    chunking_strategy: str, all_docs: list[dict], cfg: dict[str, Any] | None = None
) -> list[dict]:
    """Get list of documents and return list of chunks with added chunk metadata.
    This serves like a basic factory function for different chunking strategies."""
    # Options: "fixed", "sentence", "semantic"
    if cfg is None:
        cfg = {}

    print("\nChunking documents...")

    chunker = MovieReviewChunker()
    all_chunks = []
    for doc_idx, doc in enumerate(all_docs):
        # Apply chosen chunking strategy
        if chunking_strategy == "fixed":
            chunks = chunker.chunk_fixed_size_tokens(
                doc["page_content"],
                chunk_size=cfg.get("chunk_size", 200),
                overlap=cfg.get("overlap", 50),
            )
        elif chunking_strategy == "sentence":
            chunks = chunker.chunk_by_sentences(
                doc["page_content"],
                sentences_per_chunk=cfg.get("sentences_per_chunk", 5),
            )
        elif chunking_strategy == "semantic":
            chunks = chunker.chunk_by_semantic_similarity(
                doc["page_content"], threshold=cfg.get("threshold", 0.7)
            )
        else:
            raise ValueError(f"Unknown chunking strategy: {chunking_strategy}")

        # Create chunk objects with metadata
        for chunk_idx, chunk_text in enumerate(chunks):
            chunk_obj = {
                "text": chunk_text,
                "metadata": {
                    **doc["metadata"],  # Include all document metadata
                    "doc_id": doc_idx,
                    "chunk_id": chunk_idx,
                    "total_chunks": len(chunks),
                    "chunk_len": len(chunk_text),
                    "chunking_strategy": chunking_strategy,
                },
            }
            all_chunks.append(chunk_obj)
    return all_chunks
