# Movie RAG System

A Retrieval-Augmented Generation (RAG) system for movie information using multi-modal data from Rotten Tomatoes 
and OMDB. 

## Features

- **Hybrid Retrieval for Text**: Configurable dense, sparse, or hybrid retrieval strategies
  - **Flexible Dense Backends**: FAISS or in-memory
  - **Multiple Embedding Models**: Support for Sentence Transformers and OpenAI embeddings
- **Visual Search**: CLIP-based fused poster image + movie info text embeddings for visual search
- **Comprehensive Dataset**: 8,000+ movies with reviews, 6,000+ plot summaries, 6000+ poster images

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

### Dataset Setup

Follow the steps in [datasets/rotten-tomatoes-reviews/README.md](datasets/rotten-tomatoes-reviews/README.md) to prepare the datasets:

## Architecture

### Retriever Types

- **Dense Retriever** (`FaissDenseRetriever`, `InMemoryDenseRetriever`): Semantic search using embeddings
- **Sparse Retriever** (`BM25SparseRetriever`): Keyword-based search using BM25 algorithm
- **Hybrid Retriever** (`HybridRetriever`): Combines dense and sparse with configurable weighting
- **Visual Retriever** (`VisualRetriever`): CLIP-based poster image + movie info text search with configurable 
weight for text and image fusion

### Document Processing

- **Document Creators**: Convert DataFrames to RAG-ready documents (`src/data/document_creators.py`)
- **Chunking**: Text chunking for optimal retrieval (`src/data/chunk.py`)
- **Embeddings**: Support for multiple embedding providers (`src/utils/embeddings.py`, `src/utils/clip_embeddings.py`)

## Usage

### Retrieval

Examples for full retrieval pipeline: [notebooks/3-retrieval-text.ipynb](notebooks/3-retrieval-text.ipynb), [notebooks/5-retrieval-visual.ipynb](notebooks/5-retrieval-visual.ipynb)

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

### Project Structure

```
movie-rag/
├── src/
│   ├── data/               # Document creation and chunking
│   ├── retrievers/         # Retriever implementations
│   └── utils/              # Embedding utilities
├── datasets/               # Movie datasets (reviews, plots, posters)
├── notebooks/              # Jupyter notebooks for experimentation
├── tests/                  # Test suite
└── pyproject.toml          # Poetry configuration
```

## License

- This project uses data from: 
  - [Rotten Tomatoes Movies and Reviews Dataset](https://www.kaggle.com/datasets/stefanoleone992/rotten-tomatoes-movies-and-critic-reviews-dataset) (Kaggle)
  - OMDB API for plot summaries and poster images (requires free or paid API key)
- OpenAI API for embeddings and LLM (requires OpenAI API key, paid)
- [Hugging Face](https://huggingface.co/) for Sentence Transformers and CLIP models (free)
