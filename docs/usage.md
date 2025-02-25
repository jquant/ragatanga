# Ragatanga Usage Guide

This document provides comprehensive usage guidance for the Ragatanga hybrid retrieval system.

## Basic Usage

### Setting Up Ragatanga

```python
import asyncio
from ragatanga.core.ontology import OntologyManager
from ragatanga.core.retrieval import AdaptiveRetriever
from ragatanga.core.query import generate_structured_answer

async def setup_ragatanga(ontology_path, kb_path=None):
    """Set up Ragatanga with ontology and knowledge base."""
    # Initialize ontology manager
    ontology_manager = OntologyManager(ontology_path)
    await ontology_manager.load_and_materialize()
    
    # Initialize adaptive retriever
    retriever = AdaptiveRetriever(ontology_manager)
    
    return ontology_manager, retriever
```

### Processing Queries

```python
async def process_query(query, retriever):
    """Process a natural language query."""
    # Retrieve relevant information
    retrieved_texts, confidence_scores = await retriever.retrieve(query)
    
    # Generate a structured answer
    answer = await generate_structured_answer(query, retrieved_texts, confidence_scores)
    
    return answer
```

### Complete Example

```python
import asyncio
from ragatanga.core.ontology import OntologyManager
from ragatanga.core.retrieval import AdaptiveRetriever
from ragatanga.core.query import generate_structured_answer

async def main():
    # Set up Ragatanga
    ontology_path = "path/to/ontology.ttl"
    ontology_manager = OntologyManager(ontology_path)
    await ontology_manager.load_and_materialize()
    
    retriever = AdaptiveRetriever(ontology_manager)
    
    # Process queries
    queries = [
        "What is the price of the Plus plan?",
        "Which units are in Belo Horizonte?",
        "What's the difference between Premium and Basic plans?"
    ]
    
    for query in queries:
        print(f"\nProcessing query: {query}")
        
        # Retrieve information
        retrieved_texts, confidence_scores = await retriever.retrieve(query)
        
        # Generate answer
        answer = await generate_structured_answer(query, retrieved_texts, confidence_scores)
        
        # Print answer
        print(f"\nAnswer: {answer.answer}")

if __name__ == "__main__":
    asyncio.run(main())
```

## Advanced Usage

### Customizing Embedding Providers

Ragatanga supports multiple embedding providers that can be configured through environment variables or directly in code:

```python
import os
from ragatanga.utils.embeddings import (
    EmbeddingProvider,
    OpenAIEmbeddingProvider,
    HuggingFaceEmbeddingProvider,
    SentenceTransformersEmbeddingProvider
)

# Method 1: Set environment variable
os.environ["EMBEDDING_PROVIDER"] = "huggingface"
provider = EmbeddingProvider.get_provider()

# Method 2: Specify provider directly
provider = EmbeddingProvider.get_provider("sentence-transformers")

# Method 3: Create provider instance directly with custom parameters
provider = HuggingFaceEmbeddingProvider(
    model="sentence-transformers/all-mpnet-base-v2",
    device="cuda"
)
```

### Customizing LLM Providers

Similar to embedding providers, Ragatanga supports multiple LLM providers:

```python
import os
from ragatanga.core.llm import (
    LLMProvider,
    OpenAIProvider,
    HuggingFaceProvider,
    OllamaProvider,
    AnthropicProvider
)

# Method 1: Set environment variable
os.environ["LLM_PROVIDER"] = "anthropic"
provider = LLMProvider.get_provider()

# Method 2: Specify provider directly
provider = LLMProvider.get_provider("huggingface")

# Method 3: Create provider instance directly with custom parameters
provider = HuggingFaceProvider(
    model="mistralai/Mistral-7B-Instruct-v0.2",
    local=True,
    device="cuda"
)

# Method 4: Use Ollama for local models
provider = OllamaProvider(
    model="llama3",
    api_url="http://localhost:11434"
)
```

### Working with Ontologies

Ragatanga uses ontologies in Turtle (.ttl) format. Here's how to work with them:

```python
from ragatanga.core.ontology import OntologyManager, extract_relevant_schema

# Initialize with ontology file
manager = OntologyManager("path/to/ontology.ttl")

# Load and materialize inferences
await manager.load_and_materialize()

# Get statistics about the ontology
stats = manager.get_ontology_statistics()
print(f"Classes: {stats['statistics']['total_classes']}")
print(f"Individuals: {stats['statistics']['total_individuals']}")
print(f"Properties: {stats['statistics']['total_properties']}")

# Execute a SPARQL query
results = await manager.execute_sparql("""
    PREFIX : <http://example.org/ontology#>
    SELECT ?entity ?label
    WHERE {
        ?entity a :Class ;
                rdfs:label ?label .
    }
""")

# Get properties of an individual
properties = await manager.get_individual_properties("http://example.org/ontology#individual1")

# Extract schema relevant to a specific query
schema = await extract_relevant_schema("What is the price of Plan X?", manager.owl_file_path)
```

### Working with Knowledge Bases

Ragatanga can use text knowledge bases in addition to ontologies:

```python
from ragatanga.core.semantic import SemanticSearch

# Initialize semantic search
semantic_search = SemanticSearch()

# Load a knowledge base file
await semantic_search.load_knowledge_base("path/to/knowledge_base.md")

# Search the knowledge base
results = await semantic_search.search("What are the gym hours?", k=5)

# Search with similarity scores
results, scores = await semantic_search.search_with_scores("What are the gym hours?", k=5)
```

## Advanced Configuration

### Environment Variables

```bash
# API keys
export OPENAI_API_KEY="your-openai-api-key"
export ANTHROPIC_API_KEY="your-anthropic-api-key"
export HF_API_KEY="your-huggingface-api-key"

# Provider configuration
export EMBEDDING_PROVIDER="openai"  # openai, huggingface, sentence-transformers
export LLM_PROVIDER="openai"  # openai, huggingface, ollama, anthropic

# Model configuration
export OPENAI_MODEL="gpt-4o"
export OPENAI_EMBEDDING_MODEL="text-embedding-3-large"

# Path configuration
export ONTOLOGY_PATH="/path/to/your/ontology.ttl"
export KNOWLEDGE_BASE_PATH="/path/to/your/knowledge_base.md"
```

### Configuration Module

The `config.py` module provides centralized configuration for the entire system:

```python
from ragatanga.config import (
    ONTOLOGY_PATH,
    KNOWLEDGE_BASE_PATH,
    TEMPERATURE,
    MAX_TOKENS,
    DEFAULT_LLM_MODEL,
    DEFAULT_PORT
)

# Use configuration values
print(f"Using ontology at: {ONTOLOGY_PATH}")
print(f"Using knowledge base at: {KNOWLEDGE_BASE_PATH}")
print(f"Default LLM model: {DEFAULT_LLM_MODEL}")
```

### Version Information

You can access the package version information programmatically:

```python
from ragatanga._version import __version__

print(f"Ragatanga version: {__version__}")
```

### Multilingual Support

Ragatanga includes translation capabilities for multilingual support:

```python
from ragatanga.utils.translation import translate_query_to_ontology_language

# Translate a query to the ontology language (default is English)
translated_query = translate_query_to_ontology_language(
    "Quais unidades estão em Belo Horizonte?",  # Portuguese query
    target_language="en"  # Translate to English
)

print(f"Translated query: {translated_query}")
# Output: "Which units are in Belo Horizonte?"
```

### API Configuration

When running as an API server, Ragatanga can be configured with:

```bash
# Start server on custom port
python -m ragatanga.main --port 9000 --host 127.0.0.1
```

## Performance Tuning

### Caching Strategies

Ragatanga doesn't include built-in caching, but you can implement it using standard Python techniques:

```python
import functools
from typing import Dict, Any, Tuple

# Simple in-memory cache for SPARQL queries
_sparql_cache: Dict[str, Any] = {}

# Decorator for caching
def cached(func):
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        # Create a cache key from args and kwargs
        key = str(args) + str(sorted(kwargs.items()))
        
        if key in _sparql_cache:
            return _sparql_cache[key]
        
        result = await func(*args, **kwargs)
        _sparql_cache[key] = result
        return result
    return wrapper

# Apply to functions
@cached
async def my_expensive_function(arg1, arg2):
    # Expensive operation here
    return result
```

### Optimizing Ontology Loading

For large ontologies, you might want to optimize loading:

```python
# Load only when needed (lazy loading)
ontology_manager = OntologyManager("path/to/ontology.ttl")
# Only materialize when needed
await ontology_manager.load_and_materialize(force_rebuild=False)
```

## Troubleshooting

### Common Issues

1. **SPARQL query errors**: Check that your ontology is properly formatted and that your queries use the correct prefixes and properties.

2. **Embedding errors**: Ensure you have the appropriate API keys set for the embedding provider you're using.

3. **LLM generation failures**: Check your API keys and network connection. Also ensure your prompts aren't too long for the model's context window.

4. **Performance issues**: For large ontologies, consider:
   - Using caching
   - Reducing the number of inferences materialized
   - Using more efficient embedding models

### Logging

Ragatanga uses `loguru` for logging. You can configure it for more detailed output:

```python
from loguru import logger

# Set log level
logger.level("INFO")

# Add file output
logger.add("ragatanga.log", rotation="500 MB")

# Debug specific components
logger.debug("Detailed debugging information")
```