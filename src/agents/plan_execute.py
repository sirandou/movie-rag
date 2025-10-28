from typing import TypedDict, Annotated, List, Dict, Any
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_core.messages import HumanMessage, SystemMessage

from src.agents.base_movie_agent import MovieAgent
from src.langchain.prompts import PLANNING_PROMPT, SYNTHESIZE_PROMPT


class PlanExecuteState(TypedDict):
    """State for plan-execute agent."""

    messages: Annotated[list, add_messages]
    plan: List[str]
    step_number: int
    step_results: List[str]


class PlanExecuteAgent(MovieAgent):
    """Agent that plans before executing. Better for complex multistep queries.
    START
      ↓
    planner     # creates plan with steps
      ↓
    executor   # prepares next step (step_number++)
        ↓
        ├── model  (if there’s a step left)  # creates tool calls for current step
        │      ↓
        │    tools
        │      ↓
        │  updater  # updates step_results with tool call results
        │      ↓
        │  executor  (loop)
        │
        └── synthesizer  (if all steps done or step_number > len(plan))
                ↓
                END
    """

    def _build_agent(self) -> None:
        """Build plan-execute graph."""
        workflow = StateGraph(PlanExecuteState)

        # Add nodes
        workflow.add_node("planner", self._plan)
        workflow.add_node("executor", self._execute_step)
        workflow.add_node("model", self._call_model)
        workflow.add_node("tools", ToolNode(self.tools))
        workflow.add_node("updater", self._update_results)
        workflow.add_node("synthesizer", self._synthesize)

        # Add edges
        workflow.add_edge(START, "planner")
        workflow.add_edge("planner", "executor")

        # Routing from executor → either model to generate tool call, or we're at synthesize step → synthesizer
        workflow.add_conditional_edges(
            "executor",
            self._should_continue_execution,
            {"model": "model", "synthesizer": "synthesizer"},
        )

        workflow.add_edge("model", "tools")
        workflow.add_edge("tools", "updater")
        workflow.add_edge("updater", "executor")
        workflow.add_edge("synthesizer", END)

        self.agent = workflow.compile()

    def _plan(self, state: PlanExecuteState) -> Dict[str, Any]:
        """Create execution plan from user query."""
        question = state["messages"][0].content

        # Create planning prompt
        plan_prompt = PLANNING_PROMPT.format(question=question)

        plan_message = [
            SystemMessage(content="You are a planning agent."),
            HumanMessage(content=plan_prompt),
        ]

        response = self.llm.invoke(plan_message)

        # Parse plan into steps
        plan_text = response.content
        steps = [
            line.strip()
            for line in plan_text.split("\n")
            if line.strip() and line.strip()[0].isdigit()
        ]

        print(f"\nPlan created ({len(steps)} steps):")
        for step in steps:
            print(f"  {step}")

        return {"plan": steps, "step_number": 0, "step_results": []}

    def _execute_step(self, state: PlanExecuteState) -> Dict[str, Any]:
        """Update the step number ahead of execution."""
        step_number = state["step_number"]
        return {"step_number": step_number + 1}

    def _should_continue_execution(self, state: PlanExecuteState) -> str:
        plan = state["plan"]
        step_number = state["step_number"]

        if step_number >= len(plan):  # since last step is synthesizer
            return "synthesizer"
        else:
            return "model"

    def _call_model(self, state: PlanExecuteState) -> Dict[str, Any]:
        plan = state["plan"]
        step_number = state["step_number"]
        step_results = state["step_results"]

        # Get current step
        step = plan[step_number - 1]  # step_number is 1-indexed

        print(f"\nExecuting step {step_number}/{len(plan)}: {step}")

        # Build context with previous results
        context = "\n".join(
            [f"Step {i + 1} result: {result}" for i, result in enumerate(step_results)]
        )

        # Execute step
        step_message = [
            SystemMessage(
                content=f"Original question: {state['messages'][0].content}\n"
                + (f"Previous steps:\n{context}" if context else "First step.")
            ),
            HumanMessage(content=f"Execute this step: {step}"),
        ]

        response = self.llm_with_tools.invoke(step_message)

        return {"messages": [response]}

    def _update_results(self, state: PlanExecuteState) -> Dict[str, Any]:
        """Called after tool execution to update step results."""
        step_results = state["step_results"]
        step_results.append(state["messages"][-1].content)
        return {"step_results": step_results}

    def _synthesize(self, state: PlanExecuteState) -> Dict[str, Any]:
        """Synthesize all step results into final answer."""
        plan = state["plan"]
        step_results = state["step_results"]
        original_question = state["messages"][0].content

        print("\nSynthesizing results...")

        # Build synthesis prompt
        results_text = "\n\n".join(
            [
                f"Step {i + 1} ({plan[i]}): {result}"
                for i, result in enumerate(step_results)
            ]
        )

        synthesis_prompt = SYNTHESIZE_PROMPT.format(
            original_question=original_question, results_text=results_text
        )

        response = self.llm.invoke([HumanMessage(content=synthesis_prompt)])

        return {"messages": [response]}

    def query(self, question: str, verbose: bool = True) -> Dict[str, Any]:
        """Query with plan-execute pattern."""

        # Stream the agent's process
        final_state = None
        msg_len = 0

        for chunk in self.agent.stream(
            {
                "messages": [HumanMessage(content=question)],
                "plan": [],
                "step_number": 0,
                "step_results": [],
            },
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
                "plan": final_state["plan"],
                "step_results": final_state["step_results"],
            }
        else:
            return {"answer": "", "messages": [], "plan": [], "step_results": []}
