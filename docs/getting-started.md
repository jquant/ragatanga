# Getting Started with Ragatanga

This guide will help you get started with Ragatanga, from installation to your first query.

## Installation

### Prerequisites

Before installing Ragatanga, ensure you have:

- Python 3.8 or higher
- pip (Python package installer)
- (Optional) A virtual environment tool like venv or conda

### Installing from PyPI

The easiest way to install Ragatanga is from PyPI:

```bash
pip install ragatanga
```

### Installing from Source

To install the latest development version:

```bash
git clone https://github.com/jquant/ragatanga.git
cd ragatanga
pip install -e .
```

## Basic Setup

### Setting Up Your Ontology

Ragatanga requires an ontology file in Turtle (.ttl) format. If you don't have one, you can use a sample ontology:

```python
import os
from ragatanga.utils.samples import download_sample_ontology

# Download a sample ontology
ontology_path = download_sample_ontology()
print(f"Sample ontology downloaded to: {ontology_path}")
```

### Creating a Knowledge Base

You can create a knowledge base from Markdown files or other text sources:

```python
import asyncio
from ragatanga.core.ontology import OntologyManager
from ragatanga.core.knowledge import KnowledgeBaseBuilder

async def build_kb():
    # Initialize ontology
    ontology_path = "path/to/ontology.ttl"
    ontology_manager = OntologyManager(ontology_path)
    await ontology_manager.load_and_materialize()
    
    # Build knowledge base from Markdown
    kb_builder = KnowledgeBaseBuilder(ontology_manager)
    await kb_builder.add_markdown_file("path/to/knowledge.md")
    
    # Save knowledge base
    await kb_builder.save("knowledge_base.md")

if __name__ == "__main__":
    asyncio.run(build_kb())
```

## Your First Query

Here's a simple example to get you started with querying:

```python
import asyncio
from ragatanga.core.ontology import OntologyManager
from ragatanga.core.retrieval import AdaptiveRetriever

async def first_query():
    # Initialize with ontology
    ontology_path = "path/to/ontology.ttl"
    ontology_manager = OntologyManager(ontology_path)
    await ontology_manager.load_and_materialize()
    
    # Create retriever
    retriever = AdaptiveRetriever(ontology_manager)
    
    # Make a simple query
    query = "What is Ragatanga?"
    results = await retriever.retrieve(query)
    
    # Print results
    print(f"Query: {query}")
    for i, result in enumerate(results):
        print(f"\nResult {i+1} (Confidence: {result.confidence:.2f}):")
        print(f"Content: {result.content[:150]}...")

if __name__ == "__main__":
    asyncio.run(first_query())
```

## Configuration

### Basic Configuration

You can configure Ragatanga using environment variables or a configuration file:

```bash
# Set API keys
export OPENAI_API_KEY="your-openai-api-key"

# Configure embedding model
export RAGATANGA_EMBEDDING_PROVIDER="openai"
export RAGATANGA_EMBEDDING_MODEL="text-embedding-ada-002"
```

### Using a Configuration File

Create a `.env` file in your project root:

```
OPENAI_API_KEY=your-openai-api-key
RAGATANGA_EMBEDDING_PROVIDER=openai
RAGATANGA_EMBEDDING_MODEL=text-embedding-ada-002
RAGATANGA_LLM_PROVIDER=openai
RAGATANGA_LLM_MODEL=gpt-3.5-turbo
```

## Next Steps

Now that you have Ragatanga set up, you can:

- Explore the [Usage Guide](usage.md) for more detailed examples
- Learn about the [Architecture](architecture.md) to understand how Ragatanga works
- Check out the [API Reference](api-reference.md) for detailed documentation of all classes and methods 