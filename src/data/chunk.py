"""Chunking strategies for movie reviews and plots."""

from typing import Dict, List

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
        self, text: str, chunk_size: int = 200, overlap: int = 50
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

            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "text": chunk_text,
                    "start_token": start,
                    "end_token": end,
                    "token_count": len(chunk_tokens),
                }
            )

            # Move forward with overlap
            start += chunk_size - overlap
            chunk_id += 1

        print(f"Created {len(chunks)} fixed-size chunks")
        return chunks

    def chunk_by_sentences(self, text: str, sentences_per_chunk: int = 5) -> List[Dict]:
        """
        Chunk by sentence boundaries (better for reviews)
        """
        chunks = []

        # Split into sentences (simple approach)
        # For reviews, consider each review as a sentence. metadata will be a separate chunk
        if "Review:" in text:
            reviews = text.split("Review: ")

            # Group reviews into chunks
            chunks.append(
                {
                    "chunk_id": 0,
                    "text": reviews[0].strip(),
                    "sentence_count": len(sent_tokenize(reviews[0])),
                }
            )
            for i in range(1, len(reviews), sentences_per_chunk):
                chunk_reviews = reviews[i : i + sentences_per_chunk]
                chunk_text = "Review: " + "Review: ".join(chunk_reviews)

                chunks.append(
                    {
                        "chunk_id": i // sentences_per_chunk,
                        "text": chunk_text.strip(),
                        "review_count": len(chunk_reviews),
                    }
                )
        else:
            # Standard sentence splitting
            sentences = sent_tokenize(text)
            sentences = [s.strip() for s in sentences if s.strip()]

            for i in range(0, len(sentences), sentences_per_chunk):
                chunk_sentences = sentences[i : i + sentences_per_chunk]
                chunk_text = " ".join(chunk_sentences)

                chunks.append(
                    {
                        "chunk_id": i // sentences_per_chunk,
                        "text": chunk_text,
                        "sentence_count": len(chunk_sentences),
                    }
                )

        print(f"Created {len(chunks)} sentence-based chunks")
        return chunks

    def chunk_by_semantic_similarity(
        self, text: str, threshold: float = 0.7, min_chunk_size: int = 100
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
                chunks.append(
                    {
                        "chunk_id": chunk_id,
                        "text": " ".join(current_chunk),
                        "segment_count": len(current_chunk),
                    }
                )

                current_chunk = [segments[i]]
                current_embedding = embeddings[i]
                chunk_id += 1

        # Don't forget the last chunk
        if current_chunk:
            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "text": " ".join(current_chunk),
                    "segment_count": len(current_chunk),
                }
            )

        print(f"Created {len(chunks)} semantic chunks")
        return chunks


def chunk(chunking_strategy: str, all_docs: list[dict]) -> list[dict]:
    """Get list of documents and return list of chunks with added chunk metadata"""
    # Options: "fixed", "sentence", "semantic"
    print("\n3. Chunking documents...")

    chunker = MovieReviewChunker()
    all_chunks = []
    for doc_idx, doc in enumerate(all_docs):
        # Apply chosen chunking strategy
        if chunking_strategy == "fixed":
            chunks = chunker.chunk_fixed_size_tokens(
                doc["page_content"], chunk_size=200, overlap=50
            )
        elif chunking_strategy == "sentence":
            chunks = chunker.chunk_by_sentences(
                doc["page_content"], sentences_per_chunk=5
            )
        elif chunking_strategy == "semantic":
            chunks = chunker.chunk_by_semantic_similarity(
                doc["page_content"], threshold=0.7
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
                    "chunking_strategy": chunking_strategy,
                },
            }
            all_chunks.append(chunk_obj)
    return all_chunks
