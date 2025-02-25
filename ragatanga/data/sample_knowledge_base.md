# Ragatanga Knowledge Base

## What is Ragatanga?

Ragatanga is a hybrid knowledge retrieval system that combines semantic search with SPARQL queries to provide accurate and relevant answers to user queries. It leverages both unstructured text data and structured ontological knowledge to deliver comprehensive responses. The name "Ragatanga" is inspired by the idea of bringing together different elements (semantic search and SPARQL queries) to create a harmonious whole, much like how different musical elements come together in a song.

## Core Components

Ragatanga consists of several core components that work together to provide its functionality:

1. **Semantic Search Component**: Performs semantic search using embeddings to find relevant information in the knowledge base. Handles vector-based semantic search using embeddings, loading and processing knowledge base texts, creating and managing vector embeddings, and searching for similar content.

2. **SPARQL Query Component**: Executes SPARQL queries against the ontology to retrieve structured information. Transforms natural language queries into SPARQL format and executes them against the materialized ontology.

3. **Ontology Manager Component**: Manages the ontology, including loading, materializing inferences, and executing SPARQL queries. Responsible for loading ontology files (.ttl, .owl), materializing inferences using reasoning, executing SPARQL queries, and extracting schema information.

4. **Adaptive Retriever Component**: Combines results from semantic search and SPARQL queries to provide the most relevant information. Analyzes query complexity and specificity, determines optimal retrieval parameters, executes hybrid retrieval, and weights and ranks results.

5. **Query Processing Component**: Handles query analysis and answer generation. Analyzes query type (factual, descriptive, etc.), generates system prompts, and creates structured answers from retrieved facts.

6. **LLM Providers**: Abstracts different LLM providers for text generation. Provides a unified interface to different LLMs, handles structured output generation, and supports different provider-specific features.

## Key Features

Ragatanga offers several key features:

- **Hybrid retrieval**: Combines semantic search and SPARQL queries for comprehensive results
- **Adaptive weighting**: Dynamically adjusts retrieval strategies based on query characteristics
- **Multiple embedding providers**: Supports OpenAI, HuggingFace, and SentenceTransformers
- **Multiple LLM providers**: Works with OpenAI, Anthropic, HuggingFace, and Ollama
- **Ontology materialization**: Performs inference to derive additional facts from the ontology
- **Performance optimization**: Includes caching of materialized ontologies for better performance
- **REST API**: Provides easy integration with other systems
- **Asynchronous processing**: Built with asyncio for efficient handling of I/O operations
- **Extensible architecture**: Designed with clear extension points for customization

## Technical Architecture

Ragatanga is built using Python and leverages several key libraries:

- **FastAPI**: Powers the REST API with async support and automatic documentation
- **RDFLib**: Provides tools for working with RDF data and executing SPARQL queries
- **OwlReady2**: Handles OWL ontology processing and inference
- **FAISS**: Enables efficient vector search for semantic retrieval
- **Pydantic**: Handles data validation and settings management
- **Loguru**: Provides comprehensive logging capabilities
- **Various embedding and LLM libraries**: Depending on the provider (OpenAI, HuggingFace, etc.)

The system follows a modular architecture with clear separation of concerns:

- **Core modules**: Handle the fundamental functionality (ontology.py, semantic.py, retrieval.py, query.py, llm.py)
- **API modules**: Provide the REST interface (app.py, routes.py, models.py)
- **Utility modules**: Provide common functionality used across the system (embeddings.py, sparql.py)

## Data Flow in Ragatanga

1. **User Query Flow**:
   User Query → API → AdaptiveRetriever → [Ontology Query + Semantic Search] → Merge Results → Generate Answer → Response

2. **Ontology Processing Flow**:
   Ontology File → Load → Materialize Inferences → Execute SPARQL Queries → Results

3. **Knowledge Base Processing Flow**:
   KB File → Chunking → Embedding Generation → Index Creation → Semantic Search → Results

4. **Answer Generation Flow**:
   Retrieved Facts → Analyze Query Type → Generate System Prompt → LLM Generation → Structured Answer

## Design Patterns

Ragatanga implements several design patterns:

- **Factory Pattern**: Used for creating embedding and LLM providers
- **Strategy Pattern**: Used for different retrieval strategies that can be swapped out
- **Adapter Pattern**: Used to provide a consistent interface to different LLM and embedding providers
- **Repository Pattern**: Used for abstracting data access to ontology and knowledge base data

## Getting Started

To use Ragatanga, you need to:

1. Install the package using pip: `pip install ragatanga`
2. Set up your environment variables (API keys, etc.)
3. Prepare your ontology and knowledge base
4. Start the API server: `python -m ragatanga.main`

Basic usage in Python:

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
    
    # Process a query
    query = "What is Ragatanga?"
    retrieved_texts, confidence_scores = await retriever.retrieve(query)
    answer = await generate_structured_answer(query, retrieved_texts, confidence_scores)
    
    print(f"Answer: {answer.answer}")

if __name__ == "__main__":
    asyncio.run(main())
```

## Configuration

Ragatanga can be configured using environment variables or by modifying the config.py file. Key configuration options include:

- **ONTOLOGY_PATH**: Path to the ontology file
- **KNOWLEDGE_BASE_PATH**: Path to the knowledge base file
- **EMBEDDING_PROVIDER**: Provider to use for embeddings (openai, huggingface, sentence-transformers)
- **LLM_PROVIDER**: Provider to use for LLMs (openai, anthropic, huggingface, ollama)
- **DEFAULT_PORT**: Port for the API server
- **OPENAI_API_KEY**: API key for OpenAI services
- **ANTHROPIC_API_KEY**: API key for Anthropic services
- **HF_API_KEY**: API key for HuggingFace services

## API Endpoints

Ragatanga provides several API endpoints:

- **POST /query**: Submit a query and get an answer
- **GET /status**: Check the status of the system
- **GET /describe_ontology**: Get a description of the loaded ontology

## Customization and Extension

Ragatanga is designed to be extensible in several ways:

1. **New Embedding Providers**: Implement the `EmbeddingProvider` interface and add to the factory method
2. **New LLM Providers**: Implement the `LLMProvider` interface and add to the factory method
3. **Enhanced Retrieval Strategies**: Extend or replace `AdaptiveRetriever` with custom retrieval logic
4. **New Answer Generation Methods**: Extend the `generate_structured_answer` function and add new query type handlers

## Performance Considerations

For optimal performance, consider the following:

- Use a pre-materialized ontology to avoid the overhead of materialization at startup
- Use a cached knowledge base index to avoid rebuilding the index at startup
- Adjust the chunk size and overlap for knowledge base processing based on your content
- Choose appropriate embedding and LLM providers based on your requirements and resources
- Implement caching for expensive operations like SPARQL queries
- Use async processing for heavy I/O operations

## Troubleshooting

Common issues and their solutions:

- **Knowledge base index not initialized**: Ensure that the knowledge base file exists and is properly formatted
- **Ontology not found**: Ensure that the ontology file exists and is a valid OWL/RDF file
- **API key not set**: Set the appropriate environment variables for your chosen providers
- **SPARQL query errors**: Check that your ontology is properly formatted and that your queries use the correct prefixes
- **Embedding errors**: Ensure you have the appropriate API keys and models set for the embedding provider
- **LLM generation failures**: Check API keys, network connection, and prompt length compatibility with the model
