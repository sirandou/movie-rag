import os
from typing import Optional


def setup_langsmith(project_name: str = "movie-rag") -> None:
    """
    Setup LangSmith tracing.

    Args:
        project_name: Project name in LangSmith
    """
    # Enable tracing
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = project_name

    print("✓ LangSmith tracing enabled")
    print(f"  Project: {project_name}")
    print("  Dashboard: https://smith.langchain.com/")


class TracingContext:
    """Context manager for LangSmith tracing."""

    def __init__(self, project_name: str, run_name: Optional[str] = None):
        self.project_name = project_name
        self.run_name = run_name
        self.original_project = os.getenv("LANGCHAIN_PROJECT")

    def __enter__(self):
        os.environ["LANGCHAIN_PROJECT"] = self.project_name
        if self.run_name:
            print(f"Starting traced run: {self.run_name}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.original_project:
            os.environ["LANGCHAIN_PROJECT"] = self.original_project
        if self.run_name:
            print(f"Completed run: {self.run_name}")


# Usage example
def trace_query(chain, query: str, run_name: str = None):
    """Execute query with tracing."""
    with TracingContext("movie-rag", run_name):
        result = chain.query(query)
    return result
