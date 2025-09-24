# RAG Pipeline Instructions

## Async Patterns
- Use async/await for I/O operations
- Implement proper connection pooling
- Handle timeouts and retries gracefully

## Document Processing
- Implement chunking strategies for large documents
- Use overlap between chunks for context preservation
- Handle different document formats consistently

## Vector Operations
- Prefer batch operations for embeddings
- Include similarity score thresholds
- Handle embedding model versioning

## LLM Integration
- Manage context window limits
- Implement prompt templates
- Include fallback strategies for API failures

## Common Patterns
```python
# Async retrieval pattern
async def retrieve_context(query: str, top_k: int = 5) -> List[Document]:
    # Use connection pools and proper error handling
```

## Error Handling
- Circuit breakers for external APIs
- Graceful degradation when retrieval fails
- Proper logging with correlation IDs