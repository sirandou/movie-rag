from typing import Dict, Any, Literal

from src.langchain.chains.base import BaseChain
from src.langchain.prompts import ROUTER_PROMPT, VISUAL_RAG_PROMPT, COMBINED_RAG_PROMPT
from src.retrievers.visual_retriever import VisualRetriever


class MultiModalMovieRouter:
    """
    Routes queries to text, visual, or both.
    Combines results intelligently.
    """

    def __init__(
        self,
        text_chain: BaseChain,
        visual_retriever: VisualRetriever,
        classifier: Literal["heuristic", "llm"] = "heuristic",
    ) -> None:
        """
        Args:
            text_chain (BaseChain) instance, for example a MovieRAGChain instance
            visual_retriever (VisualRetriever): Visual retriever instance
        """
        self.text_chain = text_chain
        self.visual_retriever = visual_retriever
        self.llm = text_chain.llm
        self.classifier = classifier

    def query(self, question: str, k: int = 5) -> Dict[str, Any]:
        """
        Multi-modal query with automatic routing.

        Returns:
            Dict with modality, answer, and results
        """
        # Classify query
        modality = self._classify(question)

        print(f"Modality: {modality}")

        if modality == "text":
            return self._text_only(question, k)
        elif modality == "visual":
            return self._visual_only(question, k)
        else:
            return self._combined(question, k)

    def _classify(self, query: str) -> str:
        """A simple heuristic or LLM-based classifier to route queries."""
        if self.classifier == "heuristic":
            query_lower = query.lower()

            # Visual keywords: only have to be about posters
            visual_words = [
                "poster",
                "image",
                "see",
                "visual",
                "look",
                "dark",
                "bright",
                "color",
                "colour",
                "style",
                "aesthetic",
                "tone",
                "cinematography",
                "scene",
                "scenery",
                "landscape",
                "vibrant",
                "neon",
                "gloomy",
                "moody",
                "text",
                "logo",
            ]
            has_visual = any(w in query_lower for w in visual_words)

            # Since it's harder to define text keywords, we will always retrieve text with heuristic approach
            if has_visual:
                route = "both"
            else:
                route = "text"
        else:
            router_prompt = ROUTER_PROMPT.format(query=query)
            route = self.llm.invoke(router_prompt).content.strip().lower()

        if "both" in route:
            return "both"
        elif "visual" in route:
            return "visual"
        else:
            return "text"

    def _text_only(self, question: str, k: int) -> Dict[str, Any]:
        """Text-only query."""
        result = self.text_chain.query(question)

        print(result["answer"])

        return {
            "modality": "text",
            "question": question,
            "answer": result["answer"],
            "sources": result.get("sources", [])[:k],
        }

    def _visual_only(self, question: str, k: int) -> Dict[str, Any]:
        """Visual-only query."""
        results = self.visual_retriever.search(question, k=k)

        movies = [
            {
                "title": doc["metadata"].get("movie_title", "Unknown"),
                "score": float(score),
                "year": doc["metadata"].get("release_year", "N/A"),
                "poster_text": doc.get(
                    "text_content", ""
                ),  # Poster text itself includes name and year
                "poster_path": doc["poster_path"],
            }
            for doc, score in results
        ]

        movies_desc = "\n".join([f"- {movie['poster_text']}" for movie in movies])

        prompt = VISUAL_RAG_PROMPT.format(question=question, movies_desc=movies_desc)
        answer = self.llm.invoke(prompt).content

        print(answer)

        return {
            "modality": "visual",
            "question": question,
            "answer": answer,
            "movies": movies,
        }

    def _combined(self, question: str, k: int) -> Dict[str, Any]:
        """Combined text + visual query."""
        # Get text results
        text_result = self.text_chain.query(question)

        # Get visual results
        visual_results = self.visual_retriever.search(question, k=k * 2)
        visual_movies = [
            {
                "title": doc["metadata"].get("movie_title", "Unknown"),
                "score": float(score),
                "year": doc["metadata"].get("release_year", "N/A"),
                "poster_text": doc["text_content"],
                "poster_path": doc["poster_path"],
            }
            for doc, score in visual_results
        ]

        movies_desc = "\n".join([movie["poster_text"] for movie in visual_movies])

        prompt = COMBINED_RAG_PROMPT.format(
            question=question,
            text_answer=text_result["answer"],
            movies_desc=movies_desc,
        )
        answer = self.llm.invoke(prompt).content
        print(answer)

        return {
            "modality": "both",
            "question": question,
            "answer": answer,
            "text_sources": text_result.get("sources", [])[:k],
            "visual_sources": visual_movies,
        }
