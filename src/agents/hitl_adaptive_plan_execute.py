from functools import partial
from typing import TypedDict, Annotated, List, Dict, Any
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages, Messages
from langgraph.prebuilt import ToolNode
from langchain_core.messages import HumanMessage, SystemMessage

from src.agents.base_movie_agent import MovieAgent
from src.langchain.prompts import (
    PLANNING_PROMPT,
    REPLAN_DECISION_PROMPT,
    SYNTHESIZE_PROMPT,
    FALLBACK_EVALUATION_PROMPT,
)


def _get_new_tool_messages(messages: List[Messages]) -> List[Messages]:
    """Return list of ToolMessages generated after the last AI message."""
    last_ai_idx = max((i for i, m in enumerate(messages) if m.type == "ai"), default=-1)
    return [m for i, m in enumerate(messages) if m.type == "tool" and i > last_ai_idx]


class AdaptHITLPlanExecState(TypedDict):
    """State for human in the loop plan-execute agent."""

    messages: Annotated[list, add_messages]
    plan: List[str]
    step_number: int
    step_results: List[str]
    last_tools_results: Dict[str, list]
    retry_count: int
    replan_count: int
    failed_steps: List[Dict[str, Any]]
    fallback_used: bool
    replanning_needed: bool
    human_interaction_count: int
    human_choice: str


class AdaptHITLPlanExecAgent(MovieAgent):
    """Adaptive agent that can replan when steps fail with human in the loop capabilities.

    START
       ↓
    planner
       ↓
    executor (step+=1)
       ↓
    decide_replan (ask LLM, prompt includes original plan + completed steps)
    ├─ replanning_needed & replan_count < max → replanner (give failed context, only keep replan_count in new state) → executor
    ├─ step_number >= len(plan) → synthesizer → END
    └─ otherwise → model(retry/fallback aware) → tools → evaluator (based on keywords. Decides if retry-only once or fallback time. Adds failure history)
                                                            │
                                                            ├─ continue → updater (update step_result only if no failure; ie if fallback failed we skip) → executor
                                                            ├─ retry → model
                                                            ├─ fallback → model
                                                            └─ human_intervention (fallback fail, below max interv.) → human
                                                                                                                        │
                                                                                                                        ├─ model (with modified step)
                                                                                                                        ├─ updater (skip) → executor
                                                                                                                        ├─ updater (accept and continue) → executor
                                                                                                                        ├─ replanner → executor
                                                                                                                        └─ END
    """

    def _build_agent(self) -> None:
        self.max_tries_per_step = 2
        self.max_replanning_attempts = 2
        self.max_human_interaction = 3

        """Build plan-execute graph."""
        workflow = StateGraph(AdaptHITLPlanExecState)

        # Add nodes
        workflow.add_node("planner", self._plan)
        workflow.add_node("executor", self._execute_step)
        workflow.add_node("decide_replan", self._decide_replan)
        workflow.add_node(
            "model",
            partial(self._call_model, max_tries_per_step=self.max_tries_per_step),
        )
        workflow.add_node("tools", ToolNode(self.tools))
        workflow.add_node(
            "evaluator",
            partial(
                self._evaluate_step_result,
                max_tries_per_step=self.max_tries_per_step,
            ),
        )
        workflow.add_node("replanner", self._replan)

        workflow.add_node(
            "updater",
            partial(self._update_results, max_tries_per_step=self.max_tries_per_step),
        )
        workflow.add_node("synthesizer", self._synthesize)
        workflow.add_node(
            "human",
            partial(self._human, max_replanning_attempts=self.max_replanning_attempts),
        )

        # Add edges
        workflow.add_edge(START, "planner")
        workflow.add_edge("planner", "executor")

        # Routing from executor → either model to generate tool call, or we're at synthesize step → synthesizer,
        # Or need to replan due to too many failures
        workflow.add_edge("executor", "decide_replan")
        workflow.add_conditional_edges(
            "decide_replan",
            partial(
                self._should_continue_execution,
                max_replanning_attempts=self.max_replanning_attempts,
            ),
            {"model": "model", "synthesizer": "synthesizer", "replanner": "replanner"},
        )
        workflow.add_edge("model", "tools")
        workflow.add_edge("tools", "evaluator")
        # Evaluator decides if result is good or needs retry/replanning
        workflow.add_conditional_edges(
            "evaluator",
            partial(
                self._handle_step_result,
                max_tries_per_step=self.max_tries_per_step,
                max_human_interaction=self.max_human_interaction,
            ),
            {
                "continue": "updater",  # Success, continue
                "retry": "model",  # Retry same step
                "fallback": "model",  # Try with fallback
                "human_intervention": "human",  # fallback failed, ask for human help
            },
        )
        workflow.add_conditional_edges(
            "human",
            self._analyze_human,
            {
                "model": "model",  # model with modified step
                "continue": "updater",  # continue/skip
                "end": END,  # stop execution
                "replan": "replanner",  # human asked for replanning
            },
        )

        workflow.add_edge("updater", "executor")
        workflow.add_edge("replanner", "executor")
        workflow.add_edge("synthesizer", END)

        self.agent = workflow.compile()

    def _plan(self, state: AdaptHITLPlanExecState) -> Dict[str, Any]:
        """Create execution plan from user query."""
        question = state["messages"][0].content
        failed_steps = state.get("failed_steps", [])

        # Create planning prompt, including failed steps if any
        failure_context = ""
        if failed_steps:
            failure_context += "\n\nPrevious failed steps:\n" + "\n".join(
                [f"- {fs['step']}: {fs['error']}" for fs in failed_steps]
            )
            failure_context += (
                "\n\nPrevious successful steps:\n"
                + "\n".join([f"- {sr}" for sr in state.get("step_results", [])])
                + "\n\nPlease create a new plan that avoids failed approaches and only if needed, acknowledges successful steps."
            )

        plan_prompt = PLANNING_PROMPT.format(question=question) + failure_context

        plan_message = [
            SystemMessage(
                content="You are an adaptive planning agent. Create robust plans with fallback options."
            ),
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

        # Reset state
        return {
            "plan": steps,
            "step_number": 0,
            "step_results": [],
            "last_tools_results": {"successes": [], "failures": []},
            "retry_count": 0,
            "replan_count": 0,
            "failed_steps": [],
            "fallback_used": False,
            "replanning_needed": False,
            "human_interaction_count": 0,
            "human_choice": "",
        }

    def _execute_step(self, state: AdaptHITLPlanExecState) -> Dict[str, Any]:
        """Update the step number ahead of execution."""
        step_number = state["step_number"]
        return {"step_number": step_number + 1}

    def _decide_replan(self, state: AdaptHITLPlanExecState) -> Dict[str, Any]:
        """
        Ask LLM to analyze failures and decide if replanning is needed.
        """
        plan = state["plan"]
        step_results = state["step_results"]
        step_number = state["step_number"]
        original_query = state["messages"][0].content if state["messages"] else ""

        # Build the decision prompt
        current_plan_str = chr(10).join(
            f"{i + 1}. {step}" for i, step in enumerate(plan)
        )
        completed_str = (
            chr(10).join(
                f"Step {i + 1}: {result[:200]}..."
                for i, result in enumerate(step_results)
            )
            if step_results
            else "None yet"
        )
        current_step_str = plan[step_number - 1]
        if len(state["failed_steps"]) > 0:
            last_failure_str = f"{state['failed_steps'][-1]['step']}: {state['failed_steps'][-1]['error']}"
        else:
            last_failure_str = "None"

        prompt = REPLAN_DECISION_PROMPT.format(
            original_query=original_query,
            current_plan_str=current_plan_str,
            completed_str=completed_str,
            current_step_str=current_step_str,
            last_failure_str=last_failure_str,
        )

        try:
            response = self.llm.invoke(
                [
                    SystemMessage(
                        content="You are a strategic planning analyst. Give decisive recommendation for plan execution."
                    ),
                    HumanMessage(content=prompt),
                ]
            )

            decision = response.content.strip().upper()

            # Parse response
            if "REPLAN" in decision:
                return {
                    "replanning_needed": True,
                    "replan_count": state["replan_count"] + 1,
                }
            else:
                return {"replanning_needed": False}

        except Exception as e:
            print(f"Error in LLM decision: {e}, defaulting to continue")
            return {"replanning_needed": False}

    def _should_continue_execution(
        self, state: AdaptHITLPlanExecState, max_replanning_attempts: int
    ) -> str:
        plan = state["plan"]
        step_number = state["step_number"]

        if (
            state["replanning_needed"]
            and state["replan_count"] < max_replanning_attempts
        ):
            return "replanner"
        elif step_number >= len(plan):  # since last step is synthesizer
            return "synthesizer"
        else:
            return "model"

    def _call_model(
        self, state: AdaptHITLPlanExecState, max_tries_per_step: int
    ) -> Dict[str, Any]:
        plan = state["plan"]
        step_number = state["step_number"]
        step_results = state["step_results"]
        retry_count = state.get("retry_count", 0)
        fallback_used = state.get("fallback_used", False)
        last_failed_results = state["last_tools_results"]["failures"]

        # Get current step
        step = plan[step_number - 1]  # step_number is 1-indexed

        print(f"\nExecuting step {step_number}/{len(plan)}: {step}")
        if fallback_used:
            print("  (Using web search fallback)")
        elif retry_count > 0:
            print("  (Retry)")

        # Build context with previous results
        context = "\n".join([f"Step result: {result}" for result in step_results])

        # Modify instructions based on retry/fallback status
        system_content = f"Original question: {state['messages'][0].content}\n" + (
            f"Previous successful steps results:\n{context}\n"
            if context
            else "First step.\n"
        )

        human_content = f"Execute this step: {step}"

        if fallback_used:
            system_content += "Local data not found or some tools have failed. Use search_web_for_movies tool instead of the failed tools to find information online.\n"
            human_content = f" Previously decided step: {step}.\n Failed tools: {'; '.join([f'{f["name"]}: {f["content"]}' for f in last_failed_results])}.\nIMPORTANT: ONLY Use search_web_for_movies tool instead of the failed tools to execute."
        elif retry_count > 0:
            last_errors = "; ".join(
                [f"{f['name']}: {f['content']}" for f in last_failed_results]
            )
            if last_errors:
                system_content += f"Previous attempt failed with error: {last_errors}. Try a different approach or search query.\n"
            else:
                system_content += "Previous attempt failed. Try a different approach or search query.\n"

        # Execute step
        step_message = [
            SystemMessage(content=system_content),
            HumanMessage(content=human_content),
        ]

        response = self.llm_with_tools.invoke(step_message)

        return {"messages": [response]}

    def _evaluate_step_result(
        self, state: AdaptHITLPlanExecState, max_tries_per_step: int
    ) -> Dict[str, Any]:
        """Evaluate if step succeeded or failed."""
        retry_count = state.get("retry_count", 0)
        step_number = state["step_number"]
        plan = state["plan"]
        messages = state["messages"]
        new_tool_messages = _get_new_tool_messages(messages)
        tool_failures = []
        tool_successes = []

        # evaluate non-fallback and fallback differently
        if not state["fallback_used"]:
            failure_indicators = [
                "not found",
                "no results",
                "error",
                "failed",
                "unable to",
                "couldn't find",
                "no matches",
                "cannot provide",
                "not include",
                "not exist",
                "no movies found",
                "i don't have information",
                "no data available",
                "no information found",
                "I'm sorry",
                "unfortunately",
            ]

            for m in new_tool_messages:
                content_lower = str(m.content).lower()
                if any(ind in content_lower for ind in failure_indicators):
                    tool_failures.append({"name": m.name, "content": m.content[:200]})
                else:
                    tool_successes.append({"name": m.name, "content": m.content[:200]})

            # failure if any tool has failed
            is_failure = len(tool_failures) > 0
        else:
            prompt = FALLBACK_EVALUATION_PROMPT.format(
                original_question=state["messages"][0].content,
                current_step=plan[step_number - 1],
                fallback_result="\n".join([m.content for m in new_tool_messages]),
            )
            response = self.llm.invoke(
                [
                    SystemMessage(
                        content="You are a verifier that checks whether a search result satisfies an execution step."
                    ),
                    HumanMessage(content=prompt),
                ]
            )
            is_failure = "FAIL" in response.content.strip().upper()
            # since it's fallback, consider all tool messages as failures if failed
            tool_failures = [
                {"name": m.name, "content": m.content[:200]} for m in new_tool_messages
            ]

        if is_failure:
            print(
                f"\nStep failed: {'; '.join([f' - {f["content"][:100]}...' for f in tool_failures])}"
            )

            # Record failure
            failed_step = {
                "step": plan[step_number - 1],
                "error": "; ".join(
                    [f"{f['name']}: {f['content'][:200]}..." for f in tool_failures]
                ),
                "retry_count": retry_count,
            }

            failed_steps = state.get("failed_steps", [])
            failed_steps.append(failed_step)

            updates = {"failed_steps": failed_steps, "retry_count": retry_count + 1}
            if updates["retry_count"] >= max_tries_per_step:
                updates["fallback_used"] = True
            updates["last_tools_results"] = {
                "successes": tool_successes,
                "failures": tool_failures,
            }
            return updates

        # Success - reset retry count
        return {
            "retry_count": 0,
            "fallback_used": False,
            "last_tools_results": {
                "successes": tool_successes,
                "failures": tool_failures,
            },
        }

    def _handle_step_result(
        self,
        state: AdaptHITLPlanExecState,
        max_tries_per_step: int,
        max_human_interaction: int,
    ) -> str:
        # not failed
        if state["retry_count"] == 0:
            return "continue"

        # if fallback has already been used and still failing, ask for human intervention if under limit, else continue
        elif state["retry_count"] > max_tries_per_step:
            if state["human_interaction_count"] < max_human_interaction:
                return "human_intervention"
            else:
                return "continue"

        elif state["fallback_used"]:
            return "fallback"

        else:
            return "retry"

    def _human(
        self, state: AdaptHITLPlanExecState, max_replanning_attempts: int
    ) -> Dict[str, Any]:
        updates = {"human_interaction_count": state["human_interaction_count"] + 1}

        print(f"\n{'=' * 26} \033[1mHUMAN INTERVENTION REQUIRED\033[0m {'=' * 25}")
        print("\nThe step that failed with fallback: ")
        print(state["plan"][state["step_number"] - 1])
        print("\nThe failed tool calls:\n")
        print(
            "\n\n".join(
                [
                    f" {f['name']}: {f['content']}"
                    for f in state["last_tools_results"]["failures"]
                ]
            )
        )
        print("\n")
        choice = input(
            "1=Refine query, 2=Skip step, 3=Continue with this result, 4=Replan, 5=Stop: "
        ).strip()

        if choice == "1":
            new_query = input("How should we refine the query? ")
            plan_mod = state["plan"]
            plan_mod[state["step_number"] - 1] = new_query
            updates = {
                **updates,
                "plan": plan_mod,
                "human_choice": "model",
                "retry_count": 0,
                "fallback_used": False,
                "last_tools_results": {"successes": [], "failures": []},
            }
            print("\n Continuing with refined step as requested.")
            return updates

        if choice == "2":
            updates = {**updates, "human_choice": "continue"}
            print("\n Skipping to next steps as requested.")
            return updates

        if choice == "3":
            updates = {
                **updates,
                "human_choice": "continue",
                "retry_count": 0,
                "fallback_used": False,
                "last_tools_results": {"successes": [], "failures": []},
            }
            print("\n Continuing with current results as requested.")
            return updates

        if choice == "4":
            if state["replan_count"] < max_replanning_attempts:
                updates = {
                    **updates,
                    "replanning_needed": True,
                    "replan_count": state["replan_count"] + 1,
                    "human_choice": "replan",
                }
                print("\n Replanning as requested.")
                return updates
            else:
                updates = {**updates, "human_choice": "continue"}
                print("\n max replanning attempts reached, continuing to next step.")
                return updates

        if choice == "5":
            updates = {**updates, "human_choice": "end"}
            print("\n ending execution as requested.")
            return updates

        updates = {**updates, "human_choice": "continue"}
        print("\n choice not recognized, continuing to next step.")
        return updates

    def _analyze_human(self, state: AdaptHITLPlanExecState) -> str:
        return state["human_choice"]

    def _update_results(
        self, state: AdaptHITLPlanExecState, max_tries_per_step: int
    ) -> Dict[str, Any]:
        """Update step results after successful execution."""
        step_results = state["step_results"]
        messages = state["messages"]
        new_tool_messages = [m.content for m in _get_new_tool_messages(messages)]

        # Don't append if even fallback failed
        if state["retry_count"] <= max_tries_per_step:
            step_results.extend(new_tool_messages)

        return {
            "step_results": step_results,
            "retry_count": 0,
            "fallback_used": False,
            "last_tools_results": {"successes": [], "failures": []},
        }

    def _replan(self, state: AdaptHITLPlanExecState) -> Dict[str, Any]:
        """Create a new plan based on failures."""
        print("\n Replanning based on encountered issues...")

        # Create new plan
        new_state = self._plan(state)

        # Keep from last state
        new_state["replan_count"] = state["replan_count"]
        new_state["failed_steps"] = state["failed_steps"]
        new_state["messages"] = state["messages"]
        new_state["step_results"] = state["step_results"]

        return new_state

    def _synthesize(self, state: AdaptHITLPlanExecState) -> Dict[str, Any]:
        """Synthesize all step results into final answer."""
        plan = state["plan"]
        step_results = state["step_results"]
        failed_steps = state.get("failed_steps", [])
        original_question = state["messages"][0].content

        print("\nSynthesizing results...")

        # Build synthesis prompt
        results_text = "\n\n".join(
            [f"Step result: {result}" for result in step_results]
        )
        if failed_steps:
            results_text += "\n\nNote: Some steps could not be completed:\n"
            for failure in failed_steps[-3:]:  # Show last 3 failures
                results_text += f"- {failure['step']}\n"

        plan_text = "\n".join([f"{i + 1}. {step}" for i, step in enumerate(plan)])

        synthesis_prompt = SYNTHESIZE_PROMPT.format(
            original_question=original_question,
            results_text=results_text,
            plan_text=plan_text,
        )

        if failed_steps:
            synthesis_prompt += (
                "\n\nProvide the best answer possible with the available information."
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
                "last_tools_results": {"successes": [], "failures": []},
                "retry_count": 0,
                "replan_count": 0,
                "failed_steps": [],
                "fallback_used": False,
                "replanning_needed": False,
                "human_interaction_count": 0,
                "human_choice": "",
            },
            {"recursion_limit": 100},
            stream_mode="values",  # Stream state values
        ):
            if verbose:
                if len(chunk["messages"]) > msg_len:  # avoid duplicate prints
                    new_messages_num = len(chunk["messages"]) - msg_len
                    for msg in chunk["messages"][-new_messages_num:]:
                        msg.pretty_print()
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
