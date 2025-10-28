from langchain_tavily import TavilySearch
from langchain.tools import tool


class WebSearchTool:
    def __init__(self, max_results: int = 3):
        """
        Web search for current movie information.
        Handles queries that need real-time data or information missing from the documents.
        Args:
            max_results: Maximum search results to return
        """
        self.max_results = max_results

    def get_tool(self):
        max_results = self.max_results

        @tool
        def search_web_for_movies(query: str) -> str:
            """
            Search the web for movie information.

            Use this tool when:
            - User asks about upcoming releases or future movies
            - Need current information not in the database (recent reviews, news)
            - Questions about what directors/actors are currently working on
            - Real-time information (box office, trending movies)
            - Recent events or announcements
            - When the answer is not found in existing movie documents

            ALWAYS use this tool for general movie questions where the answer is not in the documents instead of trying to guess or fabricate an answer.

            Examples:
            - "What movies are coming out next month?"
            - "Latest reviews for Dune Part 3"
            - "Is Christopher Nolan working on a new film?"
            - "Top grossing movies this year"
            - any general question where the answer is not found in the existing movie documents

            Args:
                query: Search query about movies

            Returns:
                Search results with current information
            """
            try:
                # Create search tool
                search = TavilySearch(max_results=max_results)

                # Execute search
                results = search.invoke({"query": query})

                # Format results
                if not results:
                    return "No results found."

                formatted = []
                for i, result in enumerate(results["results"], 1):
                    title = result.get("title", "No title")
                    content = result.get("content", "No content")
                    url = result.get("url", "")

                    formatted.append(f"Result {i}: {title}\n{content}\nSource: {url}\n")

                return "\n".join(formatted)

            except Exception as e:
                return f"Search error: {str(e)}"

        return search_web_for_movies
