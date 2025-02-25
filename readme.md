# Ragatanga

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Version](https://img.shields.io/badge/version-0.3.0-blue.svg)](https://github.com/jquant/ragatanga/releases/tag/v0.3.0)

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

## Installation

```bash
# Install the latest version from PyPI
pip install ragatanga

# Install a specific version
pip install ragatanga==0.3.0

# Install from source
git clone https://github.com/yourusername/ragatanga.git
cd ragatanga
pip install -e .
```

## Quick Start

```python
import asyncio
from ragatanga.core.ontology import OntologyManager
from ragatanga.core.retrieval import AdaptiveRetriever
from ragatanga.core.query import generate_structured_answer

async def main():
    # Initialize ontology manager
    ontology_manager = OntologyManager("path/to/ontology.ttl")
    await ontology_manager.load_and_materialize()
    
    # Initialize adaptive retriever
    retriever = AdaptiveRetriever(ontology_manager)
    
    # Retrieve information for a query
    query = "What units are in Belo Horizonte?"
    retrieved_texts, confidence_scores = await retriever.retrieve(query)
    
    # Generate an answer
    answer = await generate_structured_answer(query, retrieved_texts, confidence_scores)
    
    # Print the answer
    print(answer.answer)

if __name__ == "__main__":
    asyncio.run(main())
```

## API Usage

Start the API server:

```bash
python -m ragatanga.main
```

Then, query the API:

```bash
curl -X POST "http://localhost:8000/query" \
     -H "Content-Type: application/json" \
     -d '{"query": "What units are in Belo Horizonte?"}'
```

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/query` | POST | Process a natural language query |
| `/upload/ontology` | POST | Upload a new ontology file |
| `/download/ontology` | GET | Download the current ontology file |
| `/upload/kb` | POST | Upload a new knowledge base file |
| `/download/kb` | GET | Download the current knowledge base |
| `/describe_ontology` | GET | Get detailed statistics about the ontology |

## Configuration

Ragatanga can be configured through environment variables:

- `OPENAI_API_KEY`: Your OpenAI API key (required for OpenAI providers)
- `ANTHROPIC_API_KEY`: Your Anthropic API key (required for Anthropic provider)
- `HF_API_KEY`: Your HuggingFace API key (required for HuggingFace API)
- `EMBEDDING_PROVIDER`: Embedding provider to use (openai, huggingface, sentence-transformers)
- `LLM_PROVIDER`: LLM provider to use (openai, huggingface, ollama, anthropic)

## Architecture

Ragatanga's modular architecture includes:

- **Core**: Core functionality for ontology management, retrieval, and query processing
  - `ontology.py`: Ontology loading, inference, and SPARQL query execution
  - `semantic.py`: Semantic search using vector embeddings
  - `retrieval.py`: Hybrid retrieval combining ontology and semantic search
  - `query.py`: Query analysis and answer generation
  - `llm.py`: Abstraction for different LLM providers

- **API**: FastAPI application and endpoints
  - `app.py`: FastAPI application and lifecycle management
  - `routes.py`: API endpoint definitions
  - `models.py`: Pydantic models for request/response validation

- **Utils**: Utility functions for embeddings, SPARQL, and translation
  - `embeddings.py`: Embedding providers and utilities
  - `sparql.py`: SPARQL query generation and utilities
  - `translation.py`: Multilingual support for query translation

- **System**: Configuration and version management
  - `config.py`: System-wide configuration settings
  - `_version.py`: Version tracking and information

## Advanced Usage

### Using Different Embedding Providers

```python
from ragatanga.utils.embeddings import EmbeddingProvider

# Get a specific provider
embed_provider = EmbeddingProvider.get_provider("sentence-transformers")

# Embed a query
query_embedding = await embed_provider.embed_query("What units are in Belo Horizonte?")
```

Available embedding providers:
- `openai`: OpenAI's text-embedding models
- `huggingface`: HuggingFace's embedding models
- `sentence-transformers`: SentenceTransformers embedding models

### Using Different LLM Providers

```python
from ragatanga.core.llm import LLMProvider

# Get a specific provider
llm_provider = LLMProvider.get_provider("huggingface", model="mistralai/Mistral-7B-Instruct-v0.2")

# Generate text
response = await llm_provider.generate_text(
    prompt="What are the benefits of regular exercise?",
    system_prompt="You are a fitness expert."
)
```

Available LLM providers:
- `openai`: OpenAI's GPT models
- `huggingface`: HuggingFace's language models (API or local)
- `ollama`: Local models via Ollama
- `anthropic`: Anthropic's Claude models

### Customizing Ontology Management

```python
from ragatanga.core.ontology import OntologyManager

# Initialize with an ontology file
manager = OntologyManager("path/to/ontology.ttl")

# Load and materialize inferences
await manager.load_and_materialize()

# Execute a SPARQL query
results = await manager.execute_sparql("""
    PREFIX : <http://example.org/ontology#>
    SELECT ?entity ?label
    WHERE {
        ?entity a :SomeClass ;
                rdfs:label ?label .
    }
""")
```

### Adaptive Retrieval Configuration

```python
from ragatanga.core.retrieval import AdaptiveRetriever

# Initialize with custom base parameters
retriever = AdaptiveRetriever(
    ontology_manager,
    base_top_k=15  # Adjust base number of results
)

# The retriever automatically adapts parameters based on query type and complexity
results, scores = await retriever.retrieve("What's the difference between Plan A and Plan B?")
```

## Knowledge Base Format

Ragatanga accepts knowledge base files in markdown format, with entries separated by blank lines:

```markdown
# Entity Name 1

This is information about Entity 1. The system will chunk this
content for retrieval later.

# Entity Name 2

This is information about Entity 2. Each chunk will be embedded
and retrievable through semantic search.
```

## Ontology Format

Ragatanga works with OWL/RDF ontologies in Turtle (.ttl) format:

```turtle
@prefix : <http://example.org/ontology#> .
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .

:Entity1 rdf:type owl:Class ;
         rdfs:label "Entity 1" .

:property1 rdf:type owl:ObjectProperty ;
           rdfs:domain :Entity1 ;
           rdfs:range :Entity2 .
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.