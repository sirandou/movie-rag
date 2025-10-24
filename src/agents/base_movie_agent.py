from abc import abstractmethod, ABC
from typing import Dict, Any

from langchain_openai import ChatOpenAI

from src.agents.tools.collaborative_filtering import CollaborativeFilteringTool
from src.agents.tools.sql_tool import SQLMovieTool
from src.agents.tools.multimodal_rag import (
    TextMovieTool,
    VisualMovieTool,
    CombinedMovieTool,
)
from src.langchain.chains.movie_rag import MovieRAGChain
from src.retrievers.visual_retriever import VisualRetriever


class MovieAgent(ABC):
    def __init__(
        self,
        text_chain: MovieRAGChain,
        visual_retriever: VisualRetriever,
        sql_database_path: str,
        reviews_csv_path: str,
        llm_model: str = "gpt-4o-mini",
        llm_temperature: float = 0.0,
    ) -> None:
        """
        Agent with text, visual, sql, and search/recommendation tools.
        Args:
            - text_chain: MovieRAGChain instance
            - visual_retriever: VisualRetriever instance
            - sql_database_path: path to SQLite database for SQL tool
            - reviews_csv_path: path to reviews CSV for collaborative filtering tool
            - llm_model: language model name
            - llm_temperature: llm temperature
        """
        self.agent = None

        # Create tools
        print("Creating tools...")
        text_tool = TextMovieTool(text_chain).get_tool()
        visual_tool = VisualMovieTool(
            visual_retriever, llm_model, llm_temperature
        ).get_tool()
        combined_tool = CombinedMovieTool(
            text_chain, visual_retriever, llm_model, llm_temperature
        ).get_tool()
        sql_tool = SQLMovieTool(
            sql_database_path, llm_model, llm_temperature
        ).get_tool()
        item_based_cf_tool = CollaborativeFilteringTool(
            reviews_path=reviews_csv_path
        ).get_tool()
        self.tools = [
            text_tool,
            visual_tool,
            combined_tool,
            sql_tool,
            item_based_cf_tool,
        ]

        # Bind tools to LLM
        llm = ChatOpenAI(model=llm_model, temperature=llm_temperature)
        self.llm = llm
        self.llm_with_tools = llm.bind_tools(self.tools)

        # Build graph
        print("Building agent graph...")
        self._build_agent()
        print("✓ Agent ready!")

    @abstractmethod
    def _build_agent(self) -> None:
        pass

    @abstractmethod
    def query(self, question: str, verbose: bool = True) -> Dict[str, Any]:
        """
        Query the agent.

        Args:
            question: User question
            verbose: Print reasoning

        Returns:
            Agent response
        """
        pass
