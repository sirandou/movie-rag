from langchain.tools import tool
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI

from src.data.sqlite_database import MovieDatabase
from src.langchain.chains.movie_rag import MovieRAGChain
from src.langchain.prompts import (
    VISUAL_RAG_PROMPT,
    COMBINED_RAG_PROMPT,
    SQL_TOOL_PROMPT,
)
from src.retrievers.visual_retriever import VisualRetriever


class TextMovieTool:
    def __init__(self, text_chain: MovieRAGChain) -> None:
        """
        Text based movie search tool using MovieRAGChain.
        Searches plots and reviews containing some textual metadata.
        Args:
            text_chain: MovieRAGChain instance
        """
        self.chain = text_chain

    def get_tool(self) -> BaseTool:
        chain = self.chain

        @tool
        def search_movies_by_content(query: str) -> str:
            """
            Search for movies by plot, themes, story, or reviews from plots and reviews with metadata documents.

            Use this tool when the query is about:
            - Movie plots or storylines ("What is [movie] about?")
            - Themes and messages ("What themes are in [movie]?")
            - Critical reception ("What do critics say about [movie]?")
            - Story elements ("Movies about time travel")
            - Character or director analysis
            - Recommendations based on plot, genre, staff, or reviews

            DO NOT use for visual queries (colors, style, aesthetics).

            Args:
                query: User query about movie plots or reviews

            Returns:
                Answer based on movie plots and reviews
            """
            # unused for now, sources, scores, etc
            result = chain.query(query)

            return result["answer"]

        return search_movies_by_content


class VisualMovieTool:
    def __init__(
        self,
        visual_retriever: VisualRetriever,
        llm_model: str = "gpt-4o-mini",
        llm_temperature: float = 0.0,
    ) -> None:
        """
        Poster-based visual movie search (visual retriever itself might be multimodal, also using movie captions).
        Does the same task as MultiModalMovieRouter._visual_only.
        Args:
            visual_retriever: VisualRetriever instance
            llm_model: language model name
            llm_temperature: llm temperature
        """
        self.visual_retriever = visual_retriever
        self.llm = ChatOpenAI(model=llm_model, temperature=llm_temperature)

    def get_tool(self) -> BaseTool:
        retriever = self.visual_retriever
        llm = self.llm

        @tool
        def search_movies_by_visual(query: str) -> str:
            """
            Search for movies by visual style, aesthetics, colors, or poster appearance.

            Use this tool when the query mentions::
            - Visual style ("dark moody films", "bright colorful movies")
            - Colors or tone ("movies with neon colors", "gloomy atmosphere")
            - Aesthetics ("visually similar to Blade Runner")
            - Cinematography or visual elements
            - Poster appearance

            DO NOT use for plot or story questions.

            Args:
                query: Visual description or style query

            Returns:
                Answer based on movie posters and visual style (+ basic metadata)
            """
            # Search with visual retriever
            results = retriever.search(query, k=5)

            # unused for now, scores, titles, years, etc
            movies_desc = "\n".join(
                [f"- {doc.get('text_content', '')}" for doc, score in results]
            )

            prompt = VISUAL_RAG_PROMPT.format(question=query, movies_desc=movies_desc)
            answer = llm.invoke(prompt).content

            return answer

        return search_movies_by_visual


class CombinedMovieTool:
    def __init__(
        self,
        text_chain: MovieRAGChain,
        visual_retriever: VisualRetriever,
        llm_model: str = "gpt-4o-mini",
        llm_temperature: float = 0.0,
    ) -> None:
        """
        Combined text + visual retrieval.
        Extracts the _combined logic from MultiModalMovieRouter.
        Args:
            text_chain: MovieRAGChain instance
            visual_retriever: VisualRetriever instance
            llm_model: language model name
            llm_temperature: llm temperature
        """
        self.text_chain = text_chain
        self.visual_retriever = visual_retriever
        self.llm = ChatOpenAI(model=llm_model, temperature=llm_temperature)

    def get_tool(self) -> BaseTool:
        text_chain = self.text_chain
        visual_retriever = self.visual_retriever
        llm = self.llm

        @tool
        def search_movies_by_content_and_visual(query: str) -> str:
            """
            Search movies using BOTH text (plot/reviews) AND visual (poster style) data.

            Use this for complex queries that need BOTH:
            - Content understanding (themes, plot, reviews)
            - Visual understanding (style, aesthetics, colors)

            This tool ALWAYS retrieves from both modalities and intelligently combines them.

            Best for:
            - "Sci-fi movies that LOOK like Blade Runner" (visual + genre)
            - "Dark themed films with moody visuals" (theme + visual)
            - "Movies similar to X in both story and style"
            - Queries explicitly mentioning BOTH content AND visual aspects

            Args:
                query: Query needing combined text + visual analysis

            Returns:
                Answer combining both modalities
            """
            # 1. Get text results
            text_result = text_chain.query(query)

            # 2. Get visual results
            visual_results = visual_retriever.search(query, k=10)

            # unused for now, scores, titles, years, etc
            movies_desc = "\n".join(
                [f"- {doc.get('text_content', '')}" for doc, score in visual_results]
            )

            # 3. Combine with LLM (your _combined prompt logic)
            prompt = COMBINED_RAG_PROMPT.format(
                question=query,
                text_answer=text_result["answer"],
                movies_desc=movies_desc,
            )
            answer = llm.invoke(prompt).content

            return answer

        return search_movies_by_content_and_visual


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
            print(sql_query)
            # Execute query, retry logic
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
