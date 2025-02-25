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
from ragatanga.config import ONTOLOGY_PATH, KNOWLEDGE_BASE_PATH

async def main():
    # Initialize ontology manager with the sample ontology
    # The package includes a sample ontology file that's loaded by default
    ontology_manager = OntologyManager(ONTOLOGY_PATH)
    await ontology_manager.load_and_materialize()
    
    # Initialize adaptive retriever
    retriever = AdaptiveRetriever(ontology_manager)
    
    # Retrieve information for a query
    query = "What classes are defined in the sample ontology?"
    retrieved_texts, confidence_scores = await retriever.retrieve(query)
    
    # Generate an answer
    answer = await generate_structured_answer(query, retrieved_texts, confidence_scores)
    
    # Print the answer
    print(answer.answer)
    
    # You can try additional queries related to the sample data
    sample_queries = [
        "What properties does Class1 have?",
        "How many individuals are in the ontology?",
        "Describe the relationship between Class1 and Class2"
    ]
    
    for sample_query in sample_queries:
        print(f"\nQuery: {sample_query}")
        texts, scores = await retriever.retrieve(sample_query)
        result = await generate_structured_answer(sample_query, texts, scores)
        print(f"Answer: {result.answer}")

if __name__ == "__main__":
    asyncio.run(main())
```

## API Usage

Start the API server (which will use the sample data files by default):

```bash
python -m ragatanga.main
```

Then, query the API with questions about the sample data:

```bash
# Query about the sample ontology classes
curl -X POST "http://localhost:8000/query" \
     -H "Content-Type: application/json" \
     -d '{"query": "What classes are defined in the sample ontology?"}'

# Query about sample ontology properties
curl -X POST "http://localhost:8000/query" \
     -H "Content-Type: application/json" \
     -d '{"query": "What properties does Class1 have?"}'

# Get statistics about the sample ontology
curl -X GET "http://localhost:8000/describe_ontology"
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
- `ONTOLOGY_PATH`: Path to your custom ontology file
- `KNOWLEDGE_BASE_PATH`: Path to your custom knowledge base file

## Sample Data

Ragatanga comes with sample data to help you get started immediately:

### Sample Ontology (`sample_ontology.ttl`)

A minimal ontology demonstrating the basic structure with:
- Classes representing key concepts
- Properties defining relationships between classes
- Individuals (instances) of the classes
- Labels and descriptions for improved readability

This sample ontology uses standard OWL/RDF patterns and can be used as a template for building your own domain-specific ontologies.

### Sample Knowledge Base (`sample_knowledge_base.md`)

A markdown file with text descriptions that complement the ontology:
- Detailed explanations of concepts
- Usage examples
- FAQs about the domain
- Additional unstructured information

This sample knowledge base demonstrates how to structure markdown for optimal chunking and retrieval.

You can replace these sample files with your own data by setting the `ONTOLOGY_PATH` and `KNOWLEDGE_BASE_PATH` environment variables.

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
from ragatanga.config import KNOWLEDGE_BASE_PATH

# Get a specific provider
embed_provider = EmbeddingProvider.get_provider("sentence-transformers")

# Embed a query about the sample data
query_embedding = await embed_provider.embed_query("What classes are defined in the sample ontology?")

# You can also embed the entire knowledge base
with open(KNOWLEDGE_BASE_PATH, "r") as f:
    kb_text = f.read()
    kb_embedding = await embed_provider.embed_query(kb_text)
```

Available embedding providers:
- `openai`: OpenAI's text-embedding models (requires `OPENAI_API_KEY`)
- `huggingface`: HuggingFace's embedding models (requires `HF_API_KEY`)
- `sentence-transformers`: SentenceTransformers embedding models (requires `SENTENCE_TRANSFORMERS_API_KEY`)
- `anthropic`: Anthropic's embedding models (requires `ANTHROPIC_API_KEY`)
### Using Different LLM Providers

```python
from ragatanga.core.llm import LLMProvider
from ragatanga.config import ONTOLOGY_PATH
import rdflib

# Get a specific provider
llm_provider = LLMProvider.get_provider("huggingface", model="mistralai/Mistral-7B-Instruct-v0.2")

# Load the sample ontology
g = rdflib.Graph()
g.parse(ONTOLOGY_PATH, format="turtle")

# Get all classes from the sample ontology
query = """
PREFIX owl: <http://www.w3.org/2002/07/owl#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
SELECT ?class ?label
WHERE {
  ?class a owl:Class .
  OPTIONAL { ?class rdfs:label ?label }
}
"""
results = g.query(query)

# Generate a summary of the classes in the sample ontology
classes_text = "\n".join([f"Class: {str(row.class)}, Label: {str(row.label)}" for row in results])
system_prompt = "You are an ontology expert helping to document an OWL ontology."
prompt = f"Here are the classes in our sample ontology:\n{classes_text}\n\nPlease generate a brief summary of these classes."

response = await llm_provider.generate_text(
    prompt=prompt,
    system_prompt=system_prompt
)
```

### Customizing Ontology Management

```python
from ragatanga.core.ontology import OntologyManager, extract_relevant_schema
from ragatanga.config import ONTOLOGY_PATH

# Initialize with the sample ontology file
manager = OntologyManager(ONTOLOGY_PATH)

# Load and materialize inferences
await manager.load_and_materialize()

# Get statistics about the sample ontology
stats = manager.get_ontology_statistics()
print(f"Classes: {stats['statistics']['total_classes']}")
print(f"Individuals: {stats['statistics']['total_individuals']}")
print(f"Properties: {stats['statistics']['total_properties']}")

# Execute a SPARQL query against the sample ontology
results = await manager.execute_sparql("""
    PREFIX owl: <http://www.w3.org/2002/07/owl#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    SELECT ?class ?label
    WHERE {
        ?class a owl:Class ;
              rdfs:label ?label .
    }
""")

# Extract schema relevant to a query using the sample ontology
schema = await extract_relevant_schema("What are the properties of Class1?", ONTOLOGY_PATH)
```

### Working with Knowledge Bases

Ragatanga can use text knowledge bases in addition to ontologies:

```python
from ragatanga.core.semantic import SemanticSearch
from ragatanga.config import KNOWLEDGE_BASE_PATH

# Initialize semantic search
semantic_search = SemanticSearch()

# Load the sample knowledge base file
await semantic_search.load_knowledge_base(KNOWLEDGE_BASE_PATH)

# Search the sample knowledge base for information
results = await semantic_search.search("What information is available about sample topics?", k=5)

# Search with similarity scores
results, scores = await semantic_search.search_with_scores("Tell me about the sample classes", k=5)

# Print the results
for i, (text, score) in enumerate(zip(results, scores)):
    print(f"Result {i+1} (Confidence: {score:.2f}):")
    print(f"{text}\n")
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