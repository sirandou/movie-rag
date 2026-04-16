import pytest
from langchain.prompts import FewShotPromptTemplate, PromptTemplate

from src.langchain.prompts import (
    COMBINED_RAG_PROMPT,
    FALLBACK_EVALUATION_PROMPT,
    FEW_SHOT_QA_PROMPT,
    HYDE_PROMPT,
    PLANNING_PROMPT,
    REPLAN_DECISION_PROMPT,
    ROUTER_PROMPT,
    SELF_RAG_CRITIQUE_PROMPT,
    SELF_RAG_REFINE_PROMPT,
    SQL_TOOL_PROMPT,
    STREAM_PROMPT,
    SYNTHESIZE_PROMPT,
    VISUAL_RAG_PROMPT,
    ZERO_SHOT_QA_PROMPT,
)


# --- types ---


def test_zero_shot_is_prompt_template():
    assert isinstance(ZERO_SHOT_QA_PROMPT, PromptTemplate)


def test_few_shot_is_few_shot_template():
    assert isinstance(FEW_SHOT_QA_PROMPT, FewShotPromptTemplate)


# --- input variables ---


@pytest.mark.parametrize(
    "prompt, expected_vars",
    [
        (ZERO_SHOT_QA_PROMPT, {"context", "question"}),
        (FEW_SHOT_QA_PROMPT, {"context", "question"}),
        (STREAM_PROMPT, {"context", "question"}),
        (HYDE_PROMPT, {"pre_hyde_query"}),
        (SELF_RAG_CRITIQUE_PROMPT, {"question", "answer"}),
        (SELF_RAG_REFINE_PROMPT, {"question", "answer", "sources_text"}),
        (ROUTER_PROMPT, {"query"}),
        (VISUAL_RAG_PROMPT, {"question", "movies_desc"}),
        (COMBINED_RAG_PROMPT, {"question", "text_answer", "movies_desc"}),
        (SQL_TOOL_PROMPT, {"question"}),
        (PLANNING_PROMPT, {"question"}),
        (SYNTHESIZE_PROMPT, {"original_question", "results_text", "plan_text"}),
        (
            REPLAN_DECISION_PROMPT,
            {
                "original_query",
                "current_plan_str",
                "completed_str",
                "current_step_str",
                "last_failure_str",
            },
        ),
        (FALLBACK_EVALUATION_PROMPT, {"original_question", "current_step", "fallback_result"}),
    ],
)
def test_input_variables(prompt, expected_vars):
    assert set(prompt.input_variables) == expected_vars


# --- format() works with the expected variables ---


def test_zero_shot_format():
    out = ZERO_SHOT_QA_PROMPT.format(context="Some movies.", question="What is good?")
    assert "Some movies." in out
    assert "What is good?" in out


def test_hyde_format():
    out = HYDE_PROMPT.format(pre_hyde_query="Best sci-fi films?")
    assert "Best sci-fi films?" in out


def test_sql_format():
    out = SQL_TOOL_PROMPT.format(question="Top 10 movies by rating")
    assert "Top 10 movies by rating" in out


def test_router_format():
    out = ROUTER_PROMPT.format(query="dark moody films")
    assert "dark moody films" in out


def test_self_rag_critique_format():
    out = SELF_RAG_CRITIQUE_PROMPT.format(question="Q?", answer="A.")
    assert "Q?" in out
    assert "A." in out


def test_synthesize_format():
    out = SYNTHESIZE_PROMPT.format(
        original_question="What is best?",
        results_text="Result 1",
        plan_text="Step 1",
    )
    assert "What is best?" in out


def test_replan_format():
    out = REPLAN_DECISION_PROMPT.format(
        original_query="Q",
        current_plan_str="Plan",
        completed_str="Done",
        current_step_str="Step",
        last_failure_str="Fail",
    )
    assert "Q" in out


# --- templates are non-empty ---


@pytest.mark.parametrize(
    "prompt",
    [
        ZERO_SHOT_QA_PROMPT,
        STREAM_PROMPT,
        HYDE_PROMPT,
        SQL_TOOL_PROMPT,
        PLANNING_PROMPT,
        SELF_RAG_CRITIQUE_PROMPT,
        ROUTER_PROMPT,
    ],
)
def test_template_non_empty(prompt):
    assert len(prompt.template) > 20
