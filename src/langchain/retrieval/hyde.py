from typing import List
from langchain.schema import Document, BaseRetriever
from langchain_openai import ChatOpenAI
from langchain.callbacks.manager import CallbackManagerForRetrieverRun
from langchain.prompts import PromptTemplate

from src.langchain.prompts import HYDE_PROMPT


class HyDERetriever(BaseRetriever):
    """
    HyDE (Hypothetical Document Embeddings): Generate hypothetical perfect answer first,
    embed it, then retrieve documents close to this hypothetical `answer` rather than the query.

    A wrapper around any BaseRetriever (e.g., vector store).

    Flow:
    1. User query → LLM generates hypothetical perfect answer
    2. Embed hypothetical answer
    3. Search with that embedding
    4. Return actual documents
    """

    base_retriever: BaseRetriever
    llm: ChatOpenAI
    hyde_prompt: PromptTemplate
    k: int = 5

    class Config:
        arbitrary_types_allowed = True

    def __init__(
        self,
        base_retriever: BaseRetriever,
        llm_model: str = "gpt-4o-mini",
        hyde_prompt: PromptTemplate = None,
        k: int = 5,
    ):
        """
        Initialize HyDE retriever.

        Args:
            base_retriever: Underlying retriever to search with
            llm_model: Model for generating hypothetical documents
            hyde_prompt: Custom prompt (optional)
            k: Number of results
        """
        # Default HyDE prompt
        if hyde_prompt is None:
            hyde_prompt = (HYDE_PROMPT,)

        llm = ChatOpenAI(model=llm_model, temperature=0.7)

        super().__init__(
            base_retriever=base_retriever, llm=llm, hyde_prompt=hyde_prompt, k=k
        )

        print(f"✓ HyDERetriever initialized (model={llm_model})")

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun = None
    ) -> List[Document]:
        """
        Get relevant documents using HyDE.

        Args:
            query: User question
            run_manager: Callback manager

        Returns:
            Retrieved documents
        """
        # 1. Generate hypothetical document
        hypothetical_doc = self._generate_hypothetical_doc(query)

        # 2. Search using hypothetical document instead of query
        documents = self.base_retriever.get_relevant_documents(hypothetical_doc)

        # 3. Add HyDE metadata
        for doc in documents:
            doc.metadata["hyde_query"] = hypothetical_doc

        return documents[: self.k]

    def _generate_hypothetical_doc(self, query: str) -> str:
        """Generate hypothetical perfect document (answer)."""
        prompt_value = self.hyde_prompt.format(pre_hyde_query=query)

        response = self.llm.predict(prompt_value)

        return response.strip()
