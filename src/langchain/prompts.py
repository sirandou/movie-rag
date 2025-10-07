# src/langchain/prompts.py
"""Prompt templates for movie recommendations."""

from langchain.prompts import PromptTemplate, FewShotPromptTemplate

# Zero-shot
ZERO_SHOT_QA_PROMPT = PromptTemplate(
    input_variables=["context", "query"],
    template="""You are a movie expert. Use the context to answer the question.

Instructions:
- Answer based ONLY on the provided information
- Mention specific movies when relevant
- If information is insufficient, say so
- Be concise but informative

Context: {context}

Question: {query}

Answer:""",
)

# Few-shot examples
EXAMPLES = [
    {
        "query": "I want a mind-bending sci-fi movie",
        "answer": "I recommend:\n1. Inception (2010) - Dream heists with nested realities\n2. The Matrix (1999) - Simulated reality with action\n3. Primer (2004) - Complex time travel",
    },
    {
        "query": "Movies like Shawshank Redemption",
        "answer": "I recommend:\n1. The Green Mile (1999) - Prison drama with redemption themes\n2. The Pursuit of Happyness (2006) - Inspiring perseverance\n3. Good Will Hunting (1997) - Personal growth and potential",
    },
]

EXAMPLE_TEMPLATE = PromptTemplate(
    input_variables=["query", "answer"], template="User: {query}\nAssistant: {answer}"
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

    User: {query}
    Assistant:""",
    input_variables=["query", "context"],
)  # just randomly trying user assistant instead of question answer
