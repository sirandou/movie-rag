# src/langchain/prompts.py
"""Prompt templates for movie recommendations."""

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


# System prompt for LLM
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
