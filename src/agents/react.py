from functools import partial
from typing import TypedDict, Annotated, Dict, Any

from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

from src.agents.base_movie_agent import MovieAgent


class ReactState(TypedDict):
    """Define state for the agent (input to nodes and conditional edges routing functions,
    output of nodes append to it).
    State is a dictionary, where `messages` key is a list that keeps being appended to."""

    messages: Annotated[list, add_messages]


class ReactAgent(MovieAgent):
    """ReAct Agent, built with StateGraph."""

    def _build_agent(self) -> None:
        """Build the agent graph manually."""
        workflow = StateGraph(ReactState)

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
    def _call_model(state: ReactState, llm_with_tools: ChatOpenAI) -> Dict[str, Any]:
        """Call the LLM with tools."""
        messages = state["messages"]
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    def _should_continue(self, state: ReactState) -> str:
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
        final_state = None
        msg_len = 0

        for chunk in self.agent.stream(
            {"messages": [HumanMessage(content=question)]},
            stream_mode="values",  # Stream state values
        ):
            if verbose:
                if len(chunk["messages"]) > msg_len:  # avoid duplicate prints
                    chunk["messages"][-1].pretty_print()
                    msg_len = len(chunk["messages"])

            final_state = chunk

        if verbose:
            print(f"\n{'=' * 60}")
            print("Done!")
            print("=" * 60)

        if final_state:
            return {
                "answer": final_state["messages"][-1].content,
                "messages": final_state["messages"],
            }
        else:
            return {"answer": "", "messages": []}
