from openai import OpenAI
from typing import List, Tuple, Any


class SimpleLLM:
    """Simple wrapper for OpenAI chat completions."""

    def __init__(self, model: str = "gpt-4o-mini"):
        self.client = OpenAI()
        self.model = model

    def generate_llm_answer(
        self,
        query: str,
        retrieved_chunks: List[Tuple[dict, float]],
        temperature: float = 0.7,
    ) -> dict[str, Any]:
        """
        Generate answer from retrieved chunks.

        Args:
            query: User query
            retrieved_chunks: List of (chunk, score) from retriever
            temperature: Sampling temperature

        Returns:
            Dict with 'answer' and 'sources'
        """
        # Format context from top chunks
        context_parts = []
        sources = []

        for i, (chunk, score) in enumerate(retrieved_chunks, 1):
            text = chunk.get("text", "")
            metadata = chunk.get("metadata", {})

            # to create context for llm
            context_parts.append(f"[Source {i}]\n{text}")

            # just to return
            sources.append(
                {
                    "rank": i,
                    "score": score,
                    "movie": metadata.get("movie_title", "Unknown"),
                    "text": text[:150] + "...",
                }
            )

        context = "\n\n".join(context_parts)

        # Create prompt
        prompt = f"""You are a helpful movie assistant. Answer the user's question based on the provided movie information.

Retrieved Information:
{context}

User Question: {query}

Instructions:
- Answer based ONLY on the provided information
- Mention specific movies when relevant
- If information is insufficient, say so
- Be concise but informative

Answer:"""

        # Generate
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
        )

        answer = response.choices[0].message.content

        return {
            "answer": answer,
            "query": query,
            "sources": sources,
            "num_chunks_used": len(sources),
        }
