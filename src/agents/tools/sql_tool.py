from langchain_core.tools import BaseTool, tool
from langchain_openai import ChatOpenAI

from src.data.sqlite_database import MovieDatabase
from src.langchain.prompts import SQL_TOOL_PROMPT


class SQLMovieTool:
    def __init__(
        self, llm_model: str = "gpt-4o-mini", llm_temperature: float = 0.0
    ) -> None:
        """
        SQL tool for structured queries about movies.
        Args:
            llm_model: language model name
            llm_temperature: llm temperature
        """
        self.db = MovieDatabase()
        self.db.connect()
        self.llm = ChatOpenAI(model=llm_model, temperature=llm_temperature)

    def get_tool(self) -> BaseTool:
        db = self.db
        llm = self.llm

        @tool
        def search_movies_by_metadata(question: str) -> str:
            """
            Search movies using structured metadata (ratings, years, counts).

            - Counts and statistics ("How many movies...", "What percentage...")
            - Ratings and scores ("movies rated above 8")
            - Years and dates ("movies from the 1990s")
            - Sorting, filtering, and ranking ("Top 10 highest rated")
            - Aggregations ("Average rating of sci-fi movies")
            - Comparisons ("Which has more reviews, X or Y?")

            DO NOT use for:
            - Plot summaries, story content, review sentiments and review contents (use text tool)
            - Visual style queries (use visual tool)

            Args:
                question: Question about structured movie data

            Returns:
                Structured query result
            """
            # Generate SQL from natural language
            sql_prompt = SQL_TOOL_PROMPT.format(question=question)

            sql_query = llm.invoke(sql_prompt).content.strip()

            # Clean up SQL (remove markdown, etc.)
            sql_query = sql_query.replace("```sql", "").replace("```", "").strip()
            # print(sql_query)

            # Execute query with error handling
            try:
                result = db.query(sql_query)

                if result.empty:
                    return f"No movies found matching: {question}. Tried query: SQL: {sql_query}"

                if len(result) > 20:
                    # Summarize large results
                    summary = f"Found {len(result)} results. Top 10:\n\n"
                    summary += result.head(10).to_string(index=False)
                    return summary
                else:
                    return result.to_string(index=False)

            except Exception as e:
                # Try to fix common SQL errors
                if "no such column" in str(e):
                    return f"Database doesn't have that information. Error executing query: {str(e)}\nSQL: {sql_query}"
                return f"Error: {e}. Error executing query: {str(e)}\nSQL: {sql_query}"

        return search_movies_by_metadata
