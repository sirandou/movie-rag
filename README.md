# Movie RAG System

A comprehensive RAG system for movie information, built with multi-modal data from Rotten Tomatoes and OMDB. Features custom retrievers, advanced RAG patterns, and an intelligent agent that combines multiple search strategies.

## Project Components

### 1. Datasets
- **8,000+ movies** with critic reviews from Rotten Tomatoes
- **6,000+ plot summaries** from OMDB
- **6,000+ poster images** from OMDB
- **SQLite database** with structured movie metadata (ratings, genres, cast, etc.)

### 2. Retrievers
**Text Retrievers** (plots and reviews):
- **Dense**: FAISS or in-memory with configurable embeddings (Sentence Transformers, OpenAI)
- **Sparse**: BM25 keyword search
- **Hybrid**: Weighted combination of dense + sparse

**Visual Retriever**:
- CLIP-based embeddings with configurable text/image fusion weights

### 3. RAG Chains & Patterns
**Base Chains**:
- **Text RAG**: Question answering over plot summaries and reviews
- **Multimodal RAG**: Combined text + visual search with image inputs

**Advanced Patterns**:
- **HyDE**: Hypothetical Document Embeddings for query expansion
- **Reranking**: Cross-encoder reranking for improved relevance
- **Self-RAG**: Self-critique and refinement
- **Streaming**: Token-by-token streaming responses
- **Evaluation**: RAGAS metrics for RAG quality assessment

### 4. AI Agents
**ReAct Agent** (LangGraph) with multiple tools:
- Text RAG search
- Visual RAG search
- Multimodal (text + image) search
- SQL queries over structured metadata
- Collaborative filtering recommendations (item-based, using critic rating patterns)

**Plan-Execute Agent** (LangGraph):
- Multistep planning for complex queries
- Executes planned steps using available tools

## Quick Start

### Installation

1. Clone the repository:
   ```bash
   git clone git@github.com:sirandou/movie-rag.git
   cd movie-rag
   ```
   
2. Install Poetry (if not already installed):
   ```bash
   curl -sSL https://install.python-poetry.org | python3 -
   ```

3. Create conda environment from environment.yml:
   ```bash
   conda env create -f environment.yml
   conda activate movie-rag
   ```

4. Install dependencies with poetry:
   ```bash
   # Base installation
   make install

   # With ML dependencies (includes PyTorch, Lightning, W&B)
   make install-ml

   # Development installation (includes testing and formatting tools)
   make install-dev

   # Full installation (all dependencies including RAG components)
   make install-all
   ```

### Configure Notebook Paths

Before running any notebooks, update the file paths defined at the beginning of each notebook to match your local environment.

### Dataset Setup

Follow the steps in [datasets/rotten-tomatoes-reviews/README.md](datasets/rotten-tomatoes-reviews/README.md) to prepare the datasets.

### API Keys

**Never commit API keys to the repository.** Store them as environment variables:

```bash
# Required
export OPENAI_API_KEY="your-key-here"
export OMDB_API_KEY="your-key-here"
export POSTER_API_KEY="your-key-here"

# Optional (for LangSmith tracing)
export LANGCHAIN_API_KEY="your-key-here"
export LANGCHAIN_TRACING_V2="true"
export LANGCHAIN_ENDPOINT="https://api.smith.langchain.com"
```

## Codebase Overview

- **`notebooks/data_prep/`**: Scripts for creating dataset files (plots, reviews, posters, SQLite DB)
- **`notebooks/`**: Demos and examples for retrievers, RAG chains, agents, and evaluation
- **`src/data/`**: Document creation, chunking strategies, and SQLite database setup
- **`src/retrievers/`**: Core retriever implementations (dense, sparse, hybrid, visual)
- **`src/utils/`**: Embedding models and LLM utilities
- **`src/langchain/`**: RAG infrastructure
  - Document loaders, chunking, and retriever wrappers
  - RAG chains (text, multimodal) and advanced patterns (HyDE, Self-RAG, reranking)
  - Prompts, streaming, and evaluation (RAGAS)
- **`src/agents/`**: ReAct and Plan-Execute agents with tools (RAG search, SQL, collaborative filtering)

## Usage

### Examples
- **Retrieval**: [notebooks/3-retrieval-text.ipynb](notebooks/3-retrieval-text.ipynb), [notebooks/5-retrieval-visual.ipynb](notebooks/5-retrieval-visual.ipynb)
- **RAG Chains**: [notebooks/6-langchain-rag.ipynb](notebooks/6-langchain-rag.ipynb), [notebooks/10-multimodal-full-chain.ipynb](notebooks/10-multimodal-full-chain.ipynb)
- **Advanced Patterns**: [notebooks/7-langchain-rerank.ipynb](notebooks/7-langchain-rerank.ipynb), [notebooks/8-hyde-stream-langsmith.ipynb](notebooks/8-hyde-stream-langsmith.ipynb), [notebooks/9-selfrag-ragas.ipynb](notebooks/9-selfrag-ragas.ipynb)
- **Agents**: [notebooks/11-multimodal-react-agent.ipynb](notebooks/11-multimodal-react-agent.ipynb), [notebooks/12-plan_execute.ipynb](notebooks/12-plan_execute.ipynb)

## Development

### Available Commands

```bash
# Testing
make test                 # Run all tests
pytest path/to/test.py   # Run specific test

# Code Quality
make format              # Format code with Ruff

# Jupyter
make jupyter             # Start Jupyter Lab

# Cleanup
make clean               # Remove cache and temp files

# Git Hooks
make setup-hooks         # Setup pre-commit hooks
```

## License

- This project uses data from: 
  - [Rotten Tomatoes Movies and Reviews Dataset](https://www.kaggle.com/datasets/stefanoleone992/rotten-tomatoes-movies-and-critic-reviews-dataset) (Kaggle)
  - OMDB API for plot summaries and poster images (requires free or paid API key)
- OpenAI API for embeddings and LLM (requires OpenAI API key, paid)
- [Hugging Face](https://huggingface.co/) for Sentence Transformers and CLIP models (free)
