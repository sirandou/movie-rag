# GitHub Copilot Instructions

## Project Overview
Python 3.12 project using Poetry for dependency management. Multi-component codebase with ML, RAG, and deployment components.

## General Guidelines
- Use Black formatter (88 char line length)
- Type hints required (mypy configured)
- Poetry for dependencies (`poetry add` not `pip install`)
- Pathlib for file operations
- Structured logging over print statements

## Code Quality
- Write docstrings for functions and classes
- Use descriptive variable names
- Include error handling
- Write tests with pytest

See component-specific instruction files for detailed guidelines.