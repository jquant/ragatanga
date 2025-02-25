# Ragatanga

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Version](https://img.shields.io/badge/version-0.3.1-blue.svg)](https://github.com/jquant/ragatanga/releases/tag/v0.3.1)

Ragatanga is a hybrid retrieval system that combines ontology-based reasoning with semantic search for powerful knowledge retrieval.

## Features

- **💪 Hybrid Retrieval**: Combines SPARQL queries against an ontology with semantic search for comprehensive knowledge retrieval
- **🧠 Adaptive Parameters**: Dynamically adjusts retrieval parameters based on query complexity and type
- **🔄 Multiple Embedding Providers**: Support for OpenAI, HuggingFace, and Sentence Transformers embeddings
- **💬 Multiple LLM Providers**: Support for OpenAI, HuggingFace, Ollama, and Anthropic LLMs
- **🌐 Comprehensive API**: FastAPI endpoints for querying and managing knowledge
- **📊 Confidence Scoring**: Ranks results with confidence scores for higher quality answers
- **🌍 Multilingual Support**: Translates queries to match your ontology's language
- **⚙️ Flexible Configuration**: Comprehensive configuration options through environment variables and config module

## Quick Start

```bash
# Install the latest version from PyPI
pip install ragatanga
```

```python
import asyncio
from ragatanga.core.ontology import OntologyManager
from ragatanga.core.retrieval import AdaptiveRetriever

async def main():
    # Initialize with your ontology file
    ontology_manager = OntologyManager("path/to/ontology.ttl")
    await ontology_manager.load_and_materialize()
    
    # Create retriever
    retriever = AdaptiveRetriever(ontology_manager)
    
    # Query your knowledge base
    results = await retriever.retrieve("What is Ragatanga?")
    print(results)

if __name__ == "__main__":
    asyncio.run(main())
```

## Documentation

Explore the documentation to learn more about Ragatanga:

- [Getting Started](getting-started.md): Installation and basic setup
- [Usage Guide](usage.md): Detailed usage examples
- [Architecture](architecture.md): Technical overview of Ragatanga's design
- [API Reference](api-reference.md): Detailed API documentation
- [Contributing](contributing.md): Guidelines for contributing to Ragatanga 