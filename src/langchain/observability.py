import os

from langsmith import tracing_context


def setup_langsmith(project_name: str = "movie-rag") -> None:
    """
    Setup LangSmith tracing.

    Args:
        project_name: Project name in LangSmith
    """
    # Enable tracing
    os.environ["LANGCHAIN_PROJECT"] = project_name

    print("✓ LangSmith tracing enabled")
    print(f"  Project: {project_name}")
    print("  Dashboard: https://smith.langchain.com/")


# Usage example
def trace_query(chain, query: str, run_name: str = None):
    """Execute query with tracing."""
    with tracing_context("movie-rag", tags=[run_name] if run_name else None):
        result = chain.query(query)
    return result
