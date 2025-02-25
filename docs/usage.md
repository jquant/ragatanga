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
    """Process a natural language query and return results."""
    # Retrieve relevant information
    results = await retriever.retrieve(query)
    
    # Generate a structured answer
    answer = await generate_structured_answer(query, results)
    
    return {
        "query": query,
        "results": results,
        "answer": answer
    }
```

### Complete Example

```python
import asyncio
from ragatanga.core.ontology import OntologyManager
from ragatanga.core.retrieval import AdaptiveRetriever
from ragatanga.core.query import generate_structured_answer

async def main():
    # Initialize with ontology
    ontology_path = "path/to/ontology.ttl"
    ontology_manager = OntologyManager(ontology_path)
    await ontology_manager.load_and_materialize()
    
    # Create retriever
    retriever = AdaptiveRetriever(ontology_manager)
    
    # Process a query
    query = "What are the main features of Ragatanga?"
    results = await retriever.retrieve(query)
    
    # Generate answer
    answer = await generate_structured_answer(query, results)
    
    print(f"Query: {query}")
    print(f"Answer: {answer}")
    
    # Display results
    for i, result in enumerate(results):
        print(f"\nResult {i+1} (Confidence: {result.confidence:.2f}):")
        print(f"Source: {result.source}")
        print(f"Content: {result.content[:100]}...")

if __name__ == "__main__":
    asyncio.run(main())
```

## Advanced Usage

### Customizing Retrieval Parameters

```python
from ragatanga.core.config import RetrievalConfig

# Create custom retrieval configuration
config = RetrievalConfig(
    semantic_search_weight=0.7,
    ontology_search_weight=0.3,
    max_results=10,
    confidence_threshold=0.6
)

# Initialize retriever with custom config
retriever = AdaptiveRetriever(ontology_manager, config=config)
```

### Using Different Embedding Providers

```python
from ragatanga.core.embeddings import EmbeddingManager
from ragatanga.core.config import EmbeddingConfig

# Configure embedding provider
embedding_config = EmbeddingConfig(
    provider="sentence_transformers",
    model_name="all-MiniLM-L6-v2"
)

# Create embedding manager
embedding_manager = EmbeddingManager(config=embedding_config)

# Initialize retriever with custom embedding manager
retriever = AdaptiveRetriever(
    ontology_manager, 
    embedding_manager=embedding_manager
)
```

### Using Different LLM Providers

```python
from ragatanga.core.llm import LLMManager
from ragatanga.core.config import LLMConfig

# Configure LLM provider
llm_config = LLMConfig(
    provider="huggingface",
    model_name="mistralai/Mistral-7B-Instruct-v0.1"
)

# Create LLM manager
llm_manager = LLMManager(config=llm_config)

# Initialize retriever with custom LLM manager
retriever = AdaptiveRetriever(
    ontology_manager, 
    llm_manager=llm_manager
)
```

## API Usage

### Starting the API Server

```bash
# Start the API server
python -m ragatanga.api.server --ontology path/to/ontology.ttl --port 8000
```

### Making API Requests

```python
import requests
import json

# Query endpoint
response = requests.post(
    "http://localhost:8000/query",
    json={"query": "What is Ragatanga?"}
)

# Parse response
result = response.json()
print(json.dumps(result, indent=2))
```

## Environment Variables

Ragatanga can be configured using environment variables:

```bash
# Set OpenAI API key
export OPENAI_API_KEY="your-api-key"

# Configure embedding provider
export RAGATANGA_EMBEDDING_PROVIDER="openai"
export RAGATANGA_EMBEDDING_MODEL="text-embedding-ada-002"

# Configure LLM provider
export RAGATANGA_LLM_PROVIDER="openai"
export RAGATANGA_LLM_MODEL="gpt-3.5-turbo"

# Configure retrieval parameters
export RAGATANGA_MAX_RESULTS=5
export RAGATANGA_CONFIDENCE_THRESHOLD=0.7
``` 