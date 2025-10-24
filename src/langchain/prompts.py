from langchain.prompts import PromptTemplate, FewShotPromptTemplate

# Zero-shot
ZERO_SHOT_QA_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are a helpful movie assistant. Answer the user's question based on the provided movie information.

Instructions:
- Answer based ONLY on the provided information
- Mention specific movies when relevant
- If information is insufficient, say so
- Be concise but informative

Retrieved Information: {context}

User question: {question}

Answer:""",
)

# Few-shot examples
EXAMPLES = [
    {
        "question": "I want a mind-bending sci-fi movie",
        "answer": "I recommend:\n1. Inception (2010) - Dream heists with nested realities\n2. The Matrix (1999) - Simulated reality with action\n3. Primer (2004) - Complex time travel",
    },
    {
        "question": "Movies like Shawshank Redemption",
        "answer": "I recommend:\n1. The Green Mile (1999) - Prison drama with redemption themes\n2. The Pursuit of Happyness (2006) - Inspiring perseverance\n3. Good Will Hunting (1997) - Personal growth and potential",
    },
]

EXAMPLE_TEMPLATE = PromptTemplate(
    input_variables=["question", "answer"],
    template="User: {question}\nAssistant: {answer}",
)

FEW_SHOT_QA_PROMPT = FewShotPromptTemplate(
    examples=EXAMPLES,
    example_prompt=EXAMPLE_TEMPLATE,
    prefix="""You are a movie expert. Use the context to answer the question.

Instructions:
- Answer based ONLY on the provided information
- Mention specific movies when relevant
- If information is insufficient, say so
- Be concise but informative 

Examples:""",
    suffix="""
    Context:
    {context}

    User: {question}
    Assistant:""",
    input_variables=["question", "context"],
)  # just randomly trying user assistant instead of question answer


# Streaming prompt with citations
STREAM_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are a helpful movie assistant. Answer the user's question based on the provided movie information.

IMPORTANT: When referencing information from a specific source, add inline numeric citations EXACTLY like [1], [2], etc.

Instructions:
- Answer based ONLY on the provided information
- Mention specific movies when relevant
- If information is insufficient, say so
- Be concise but informative
- Each citation must correspond to one of the numbered sources below

Retrieved Information (numbered sources): {context}

User question: {question}

Answer with inline citations::""",
)


# HyDE prompt for hypothetical answer generation
HYDE_PROMPT = PromptTemplate(
    input_variables=["pre_hyde_query"],
    template="""Write a detailed paragraph that perfectly answers this question about movies.
Write as if you're a movie expert giving the ideal response.

Question: {pre_hyde_query}

Expert Answer:""",
)

# Self-RAG critique prompt
SELF_RAG_CRITIQUE_PROMPT = PromptTemplate(
    input_variables=["question", "answer"],
    template="""Evaluate this answer. Respond with ONE word: GOOD or BAD

Question: {question}
Answer: {answer}

Is it:
- Complete (answers all aspects)?
- Specific (uses details, not vague)?
- Clear (well-explained)?

ONE WORD: """,
)

# Self-RAG refine prompt
SELF_RAG_REFINE_PROMPT = PromptTemplate(
    input_variables=["question", "answer", "sources_text"],
    template="""The answer below needs improvement. Make it more complete and specific using ONLY information from the sources.

Question: {question}

Current answer: {answer}

Sources to use:
{sources_text}

Improved answer (be specific, use details from sources):""",
)

# multimodal router prompt
ROUTER_PROMPT = PromptTemplate(
    input_variables=["query"],
    template="""You are a router for a movie QA system. Given the user query, decide if it needs:
- text retrieval,
- visual retrieval, or
- both.

Query: "{query}"

Answer with one of TEXT, VISUAL, BOTH:""",
)

# multimodal visual prompt
VISUAL_RAG_PROMPT = PromptTemplate(
    input_variables=["question", "movies_desc"],
    template="""You are a helpful movie assistant. The following are visual search results for the query {question}.
Each result includes a poster description.

Results:
{movies_desc}

Summarize what these posters have in common and what kind of films they likely represent.""",
)

# multimodal combined prompt
COMBINED_RAG_PROMPT = PromptTemplate(
    input_variables=["question", "text_answer", "movies_desc"],
    template="""You are a helpful movie assistant. The question is: "{question}".

Textual answer (from plot/reviews):
{text_answer}

Visually matching movies (from poster analysis):
{movies_desc}

Answer the user's question by combining BOTH the textual information and visual matches. Explain which movies align both textually and visually, and why.
""",
)

# sql tool prompt
SQL_TOOL_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""Convert this question to a SQL query.

Database schema:
Table: movies
- rotten_tomatoes_link (TEXT, primary key)
- movie_title (TEXT)
- release_year (INTEGER)
- streaming_release_date (TEXT)
- imdb_rating, tomatometer_rating, audience_rating (REAL)
- runtime (REAL, in minutes)
- genres, directors, actors, authors (TEXT, comma-separated)
- awards, box_office, content_rating, production_company (TEXT)
- tomatometer_status, audience_status (TEXT)
- audience_count, tomatometer_count, tomatometer_top_critics_count, tomatometer_fresh_critics_count, tomatometer_rotten_critics_count (INTEGER)

Notes:
- For genre/director/actor searches, use: WHERE genres LIKE '%Sci-Fi%'
- Ratings are 0-10 scale (imdb) or 0-100 (tomatometer/audience)

Question: {question}

SQL query only:""",
)

# plan execute - planning prompt
PLANNING_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""You are a planning assistant for a movie expert system.

Available tools:
1. search_movies_by_content
   - Use for: Search for movies by plot, themes, story, or reviews from plots and reviews with metadata documents.
   - Example: "What is X about?", "Movies about time travel", "What themes are in X?, "What do critics say about X?", "Movies about [element]", 
   "Analysis of characters or directors", "recommend movies based on [plot line], [genre], [staff], [reviews]".

2. search_movies_by_visual
   - Use for: Search for movies by visual style, aesthetics, colors, or poster appearance.
   - Example: "dark moody films", "bright colorful movies", "movies with neon colors", "gloomy atmosphere", "visually similar to X".

3. search_movies_by_content_and_visual
   - Use for: Queries needing BOTH content (themes, plot, reviews) and visual (style, aesthetics, colors) understanding.
   If using this tool, you do NOT need to also use the individual content or visual tools separately. 
   This tool ALWAYS retrieves from both modalities and intelligently combines them.
   - Example: "Sci-fi movies that look like X", "Dark themed films with moody visuals", "Movies similar to X in both story and style".            

4. search_movies_by_metadata
   - Use for: searching, filtering, movie statistics using structured metadata such as ratings, years, counts.
   - Example: "How many movies...", "What percentage...", "movies rated above 8", "movies from the 1990s", "Top 10 highest rated", 
   "Average rating of sci-fi movies", "Which has more reviews, X or Y?"

5. recommend_by_similar_taste
   - Use for: **item-based collaborative filtering** to find movies that share similar critic rating patterns with the given titles. 
   It’s best used when you want recommendations based on movies the user liked.
   - Example: "I loved [X, Y] — what else might I like?", "Find movies similar to [X] based on ratings", "What movies have similar critic rating patterns to [X]?"

User question: "{question}"

Create a step-by-step plan to answer the question. Each step must:
- Specify which tool to use, 1 per step
- Explain what information to get
- Be clear and specific
- Only use the available tools
- Only use as many steps as needed

Format:
1. Use [tool_name] to [get specific information]
2. Use [tool_name] to [get specific information]
3. Synthesize results to answer the question

Now create a plan for the user's question.
Plan:""",
)
