.PHONY: install install-ml install-dev test format clean

# Install dependencies
install:
	poetry install

# Install with ML dependencies
install-ml:
	poetry install --with ml

# Install for development
install-dev:
	poetry install --with dev,ml

# Run tests
test:
	pytest

# Format and lint code
format:
	isort src tests
	black src tests
	flake8 src tests
	mypy src

# Clean cache and temp files
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf .pytest_cache

# Run jupyter lab
jupyter:
	jupyter lab

# Setup pre-commit hooks
setup-hooks:
	pre-commit install