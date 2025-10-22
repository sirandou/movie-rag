from functools import partial
from typing import TypedDict, Annotated, Dict, Any

from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

from src.agents.tools.collaborative_filtering import CollaborativeFilteringTool
from src.agents.tools.sql_tool import SQLMovieTool
from src.agents.tools.multimodal_rag import (
    TextMovieTool,
    VisualMovieTool,
    CombinedMovieTool,
)
from src.langchain.chains.movie_rag import MovieRAGChain
from src.retrievers.visual_retriever import VisualRetriever


class AgentState(TypedDict):
    """State for the agent.
    State is a dictionary, where `messages` key is a list that keeps being appended to."""

    messages: Annotated[list, add_messages]


class MovieAgent:
    def __init__(
        self,
        text_chain: MovieRAGChain,
        visual_retriever: VisualRetriever,
        llm_model: str = "gpt-4o-mini",
        llm_temperature: float = 0.0,
    ) -> None:
        """
        ReAct Agent with text and visual search tools, built with StateGraph.
        Args:
            - text_chain: MovieRAGChain instance
            - visual_retriever: VisualRetriever instance
            - llm_model: language model name
            - llm_temperature: llm temperature
        """
        # Create tools
        print("Creating tools...")
        text_tool = TextMovieTool(text_chain).get_tool()
        visual_tool = VisualMovieTool(
            visual_retriever, llm_model, llm_temperature
        ).get_tool()
        combined_tool = CombinedMovieTool(
            text_chain, visual_retriever, llm_model, llm_temperature
        ).get_tool()
        sql_tool = SQLMovieTool(llm_model, llm_temperature).get_tool()
        item_based_cf_tool = CollaborativeFilteringTool().get_tool()
        self.tools = [
            text_tool,
            visual_tool,
            combined_tool,
            sql_tool,
            item_based_cf_tool,
        ]

        # Bind tools to LLM
        llm = ChatOpenAI(model=llm_model, temperature=llm_temperature)
        self.llm_with_tools = llm.bind_tools(self.tools)

        # Build graph
        print("Building agent graph...")
        self._build_agent()
        print("✓ Agent ready!")

    def _build_agent(self) -> None:
        """Build the agent graph manually."""
        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node(
            "agent", partial(self._call_model, llm_with_tools=self.llm_with_tools)
        )
        workflow.add_node("tools", ToolNode(self.tools))

        # Add edges
        workflow.add_edge(START, "agent")
        workflow.add_conditional_edges(
            "agent", self._should_continue, {"continue": "tools", "end": END}
        )
        workflow.add_edge("tools", "agent")

        self.agent = workflow.compile()

    @staticmethod
    def _call_model(state: AgentState, llm_with_tools: ChatOpenAI) -> Dict[str, Any]:
        """Call the LLM with tools."""
        messages = state["messages"]
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    def _should_continue(self, state: AgentState) -> str:
        """Decide if we should continue or end."""
        messages = state["messages"]
        last_message = messages[-1]

        # If there are tool calls, continue
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "continue"
        else:
            return "end"

    def query(self, question: str, verbose: bool = True) -> Dict[str, Any]:
        """
        Query the agent.

        Args:
            question: User question
            verbose: Print reasoning

        Returns:
            Agent response
        """
        # Stream the agent's process
        final_answer = None

        for chunk in self.agent.stream(
            {"messages": [HumanMessage(content=question)]},
            stream_mode="values",  # Stream state values
        ):
            if verbose:
                # Pretty print last message
                chunk["messages"][-1].pretty_print()

            final_answer = chunk["messages"][-1]

        if verbose:
            print(f"\n{'=' * 60}")
            print("Done!")
            print("=" * 60)

        return {"answer": final_answer.content, "messages": chunk["messages"]}
