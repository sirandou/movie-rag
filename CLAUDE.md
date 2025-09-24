# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Python 3.12 ML project using Poetry for dependency management. The project is configured for machine learning development with multi-component architecture supporting ML, RAG, and deployment components.

## Development Commands

### Installation and Setup
- `make install` - Install base dependencies
- `make install-ml` - Install with ML dependencies (torch, lightning, wandb)
- `make install-dev` - Install with development and ML dependencies
- `make setup-hooks` - Setup pre-commit hooks

### Testing and Quality
- `make test` or `pytest` - Run all tests
- `pytest path/to/test_file.py` - Run specific test file
- `pytest path/to/test_file.py::test_function` - Run specific test
- `make format` - Run Black formatter, flake8 linter, and mypy type checker

### Development Environment
- `make jupyter` or `jupyter lab` - Start Jupyter Lab
- `make clean` - Clean cache and temporary files

## Code Standards

### Dependency Management
- Use `poetry add` for new dependencies, not `pip install`
- Separate dependencies into appropriate groups (dev, ml) in pyproject.toml

### Code Quality Requirements
- Black formatter with 88-character line length
- Type hints required (mypy configured for Python 3.12)
- Use pathlib for file operations
- Structured logging over print statements
- Write docstrings for functions and classes
- Include error handling in production code

### Project Structure
- Source code in `src/` directory
- Tests should be created in `tests/` directory (currently empty)
- Configuration managed through pyproject.toml and Makefile

## Environment Options

The project supports both Poetry and Conda environments:
- Poetry: Primary dependency management (pyproject.toml)
- Conda: Alternative setup via environment.yml

## ML Dependencies

Core ML stack includes:
- Data: numpy, pandas, scipy
- ML: scikit-learn, torch, pytorch-lightning
- Visualization: matplotlib, seaborn
- Experiment tracking: wandb
- Notebooks: jupyter, ipykernel