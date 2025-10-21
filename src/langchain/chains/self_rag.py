from typing import Any

from src.langchain.chains.base import BaseChain
from src.langchain.prompts import SELF_RAG_CRITIQUE_PROMPT, SELF_RAG_REFINE_PROMPT


class SelfRAGWrapper:
    """
    Self-critique with refinement using same documents.
    Simple and effective for most cases.
    """

    def __init__(self, base_chain: BaseChain) -> None:
        """
        Args:
            base_chain: Any BaseChain instance (e.g., standard RAG chain)
        """
        self.chain = base_chain
        self.llm = base_chain.llm

    def query(self, question: str) -> dict[str, Any]:
        """Query with self-critique and refinement.
        Steps:
          1. Get initial answer (chain retrieves + generates)
          2. Critique answer quality (GOOD/BAD)
          3. If BAD, refine using SAME sources
        """
        # 1. Initial answer
        result = self.chain.query(question)

        print(f"\n{'=' * 60}")
        print(f"Question: {question}")
        print("=" * 60)
        print(f"\nInitial answer:\n{result['answer']}")

        # 2. Critique
        critique = self._critique(question, result["answer"])
        print(f"\nCritique: {critique}")

        # 3. Refine if needed (using SAME sources)
        if "BAD" in critique.upper():
            print("\nRefining using existing sources...")

            improved = self._refine(question, result["answer"], result["sources"])

            result["original_answer"] = result["answer"]
            result["answer"] = improved
            result["refined"] = True

            print(f"\nRefined answer:\n{improved}...")
        else:
            print("\nAnswer is good, no refinement needed")
            result["refined"] = False

        return result

    def _critique(self, question: str, answer: str) -> str:
        """Critique answer quality."""
        prompt = SELF_RAG_CRITIQUE_PROMPT.format(question=question, answer=answer)

        return self.llm.predict(prompt).strip()

    def _refine(self, question: str, answer: str, sources: list[dict[str, Any]]) -> str:
        """Refine answer using existing sources."""
        # Format sources
        sources_text = "\n\n".join(
            [f"Source {i}: {s['content']}" for i, s in enumerate(sources[:5], 1)]
        )

        prompt = SELF_RAG_REFINE_PROMPT.format(
            question=question, answer=answer, sources_text=sources_text
        )

        return self.llm.predict(prompt).strip()
