# Contributing to Ragatanga

Thank you for your interest in contributing to Ragatanga! This document provides guidelines and instructions for contributing to the project.

## Code of Conduct

Please read and follow our [Code of Conduct](https://github.com/jquant/ragatanga/blob/main/CODE_OF_CONDUCT.md) to ensure a positive and inclusive environment for everyone.

## Getting Started

### Setting Up Development Environment

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/yourusername/ragatanga.git
   cd ragatanga
   ```
3. Set up a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
4. Install development dependencies:
   ```bash
   pip install -e ".[dev]"
   ```

### Running Tests

We use pytest for testing:

```bash
# Run all tests
pytest

# Run specific tests
pytest tests/test_ontology.py

# Run with coverage
pytest --cov=ragatanga
```

## Development Workflow

### Branching Strategy

- `main`: Stable release branch
- `develop`: Development branch for next release
- Feature branches: Create from `develop` with format `feature/your-feature-name`
- Bugfix branches: Create from `develop` with format `bugfix/issue-description`

### Pull Request Process

1. Create a new branch from `develop`
2. Make your changes
3. Run tests and ensure they pass
4. Update documentation if necessary
5. Submit a pull request to the `develop` branch
6. Wait for code review and address any feedback

## Coding Standards

### Code Style

We follow PEP 8 and use Black for code formatting:

```bash
# Format code
black ragatanga tests

# Check code style
flake8 ragatanga tests
```

### Type Hints

We use type hints throughout the codebase:

```python
def example_function(param1: str, param2: int) -> bool:
    """
    Example function with type hints.
    
    Args:
        param1: Description of param1
        param2: Description of param2
        
    Returns:
        Description of return value
    """
    return True
```

### Documentation

- Use docstrings for all public classes and methods
- Follow Google style docstrings
- Update documentation when adding or changing features

## Adding Features

### Adding a New Embedding Provider

1. Create a new file in `ragatanga/core/embeddings/providers/`
2. Implement the `EmbeddingProvider` interface
3. Register the provider in `ragatanga/core/embeddings/manager.py`
4. Add tests in `tests/core/embeddings/`
5. Update documentation

### Adding a New LLM Provider

1. Create a new file in `ragatanga/core/llm/providers/`
2. Implement the `LLMProvider` interface
3. Register the provider in `ragatanga/core/llm/manager.py`
4. Add tests in `tests/core/llm/`
5. Update documentation

## Release Process

1. Update version in `pyproject.toml`
2. Update `CHANGELOG.md`
3. Create a pull request from `develop` to `main`
4. After approval and merge, create a new release on GitHub
5. CI/CD will automatically publish to PyPI

## Getting Help

If you need help or have questions:

- Open an issue on GitHub
- Join our community discussions
- Reach out to the maintainers

Thank you for contributing to Ragatanga! 