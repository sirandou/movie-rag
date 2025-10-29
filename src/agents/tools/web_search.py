from langchain_core.prompts import PromptTemplate
from langchain_tavily import TavilySearch
from langchain.tools import tool
from langchain_openai import ChatOpenAI


class WebSearchTool:
    def __init__(
        self,
        max_results: int = 3,
        llm_model: str = "gpt-4o-mini",
        llm_temperature: float = 0.0,
    ):
        """
        Web search for current movie information.
        Handles queries that need real-time data or information missing from the documents.
        Args:
            max_results: Maximum search results to return
            llm_model: language model name
            llm_temperature: llm temperature
        """
        self.max_results = max_results
        self.llm = ChatOpenAI(model=llm_model, temperature=llm_temperature)

    def get_tool(self):
        max_results = self.max_results
        llm = self.llm

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

                search_results = "\n".join(formatted)
                prompt = PromptTemplate(
                    template="""You are a summarizer. You are given:

User question:
{question}

Web search results:
{search_results}

Task:
Summarize the information from the search results in a clear, concise answer that directly addresses the user question. Do not include irrelevant details or speculate beyond the given information. Cite details only if they appear in the search results.
""",
                    input_variables=["question", "search_results"],
                )
                return llm.invoke(
                    prompt.format(question=query, search_results=search_results)
                ).content

            except Exception as e:
                return f"Search error: {str(e)}"

        return search_web_for_movies
