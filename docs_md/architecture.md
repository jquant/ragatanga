# Ragatanga Architecture

This document provides a technical overview of Ragatanga's architecture, components, and design decisions.

## System Overview

Ragatanga is a hybrid retrieval system that combines ontology-based reasoning with semantic search. This combination provides several advantages:

1. **Structure + Unstructured**: Combines the precision of ontology/SPARQL queries with the flexibility of semantic search
2. **Adaptive Parameters**: Adjusts retrieval strategies based on query characteristics
3. **Extensible Design**: Supports multiple embedding and LLM providers
4. **Confidence Scoring**: Ranks results by confidence for better answers

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                      Ragatanga System                       │
└──────────────────────────────┬──────────────────────────────┘
                               │
          ┌───────────────────┼───────────────────┐
          │                   │                   │
┌─────────▼─────────┐ ┌───────▼──────┐   ┌────────▼────────┐
│      Core         │ │    Config    │   │     Version     │
└─────────┬─────────┘ └──────────────┘   └─────────────────┘
          │                                        
┌─────────▼─────────┐                   ┌────────────────────┐
│  ┌─────────────┐  │                   │        API         │
│  │  Ontology   │◄─┼───────────────────┤                    │
│  └─────┬───────┘  │                   │  ┌─────────────┐   │
│        │          │                   │  │    App      │   │
│  ┌─────▼───────┐  │                   │  └─────────────┘   │
│  │  Semantic   │◄─┼───────────────────┤  ┌─────────────┐   │
│  └─────┬───────┘  │                   │  │   Routes    │◄──┼────┐
│        │          │                   │  └─────────────┘   │    │
│  ┌─────▼───────┐  │                   │  ┌─────────────┐   │    │
│  │  Retrieval  │◄─┼───────────────────┤  │   Models    │◄──┼────┘
│  └─────┬───────┘  │                   └────────────────────┘
│        │          │
│  ┌─────▼───────┐  │                   ┌────────────────────┐
│  │   Query     │◄─┼───────────────────┤      Utils        │
│  └─────┬───────┘  │                   │ ┌───────────────┐  │
│        │          │                   │ │  Embeddings   │◄─┼──┐
│  ┌─────▼───────┐  │                   │ └───────────────┘  │  │
│  │    LLM      │◄─┼───────────────────┤ ┌───────────────┐  │  │
│  └─────────────┘  │                   │ │    SPARQL     │◄─┼──┼─┐
└───────────────────┘                   │ └───────────────┘  │  │ │
                                        │ ┌───────────────┐  │  │ │
                                        │ │  Translation  │◄─┼──┘ │
                                        │ └───────────────┘  │    │
                                        └────────────────────┘    │
                                                                 
```

## Component Details

### Core Components

#### OntologyManager (`core/ontology.py`)

Responsible for loading and managing the ontology, and executing SPARQL queries.

**Key responsibilities:**
- Loading ontology files (.ttl, .owl)
- Materializing inferences using reasoning
- Executing SPARQL queries
- Extracting schema information

#### SemanticSearch (`core/semantic.py`)

Handles vector-based semantic search using embeddings.

**Key responsibilities:**
- Loading and processing knowledge base texts
- Creating and managing vector embeddings
- Searching for similar content in the knowledge base

#### AdaptiveRetriever (`core/retrieval.py`)

Combines ontology queries and semantic search, with adaptive parameters.

**Key responsibilities:**
- Analyzing query complexity and specificity
- Determining optimal retrieval parameters
- Executing hybrid retrieval
- Weighting and ranking results

#### Query Processing (`core/query.py`)

Handles query analysis and answer generation.

**Key responsibilities:**
- Analyzing query type (factual, descriptive, etc.)
- Generating system prompts
- Creating structured answers from retrieved facts

#### LLM Providers (`core/llm.py`)

Abstracts different LLM providers for text generation.

**Key responsibilities:**
- Providing a unified interface to different LLMs
- Handling structured output generation
- Supporting different provider-specific features

### API Components

#### FastAPI Application (`api/app.py`)

Main API application with lifecycle management.

**Key responsibilities:**
- FastAPI application setup
- Lifecycle management (startup/shutdown)
- Dependency management

#### API Routes (`api/routes.py`)

API endpoints and request handlers.

**Key responsibilities:**
- Defining API endpoints
- Handling requests
- Integration with core components

#### API Models (`api/models.py`)

Pydantic models for request/response validation.

**Key responsibilities:**
- Defining request/response schemas
- Input validation
- Documentation

### Utility Components

#### Embedding Providers (`utils/embeddings.py`)

Abstracts different embedding providers.

**Key responsibilities:**
- Providing a unified interface for embeddings
- Supporting different embedding models
- Vector normalization and processing

#### SPARQL Utilities (`utils/sparql.py`)

SPARQL generation and utility functions.

**Key responsibilities:**
- Generating SPARQL queries from natural language
- SPARQL query validation
- Text similarity calculations

#### Translation Utilities (`utils/translation.py`)

Provides language translation capabilities for multilingual support.

**Key responsibilities:**
- Translating user queries to the ontology's language
- Supporting multiple translation providers (Google Translate, OpenAI)
- Fallback mechanisms for reliable translation

### System Components

#### Configuration Management (`config.py`)

Manages global configuration settings for the entire system.

**Key responsibilities:**
- Setting default paths for data files
- Configuring semantic search parameters
- Managing API settings
- Defaulting to sample files if needed
- Environment variable integration

#### Version Management (`_version.py`)

Handles version tracking for the package.

**Key responsibilities:**
- Defining the current version number
- Providing version information for the package
- Supporting setuptools-scm version tracking

## Data Flow

1. **User Query Flow**:
   ```
   User Query → API → AdaptiveRetriever → [Ontology Query + Semantic Search] → Merge Results → Generate Answer → Response
   ```

2. **Ontology Processing Flow**:
   ```
   Ontology File → Load → Materialize Inferences → Execute SPARQL Queries → Results
   ```

3. **Knowledge Base Processing Flow**:
   ```
   KB File → Chunking → Embedding Generation → Index Creation → Semantic Search → Results
   ```

4. **Answer Generation Flow**:
   ```
   Retrieved Facts → Analyze Query Type → Generate System Prompt → LLM Generation → Structured Answer
   ```

## Design Patterns

### Factory Pattern

Used for creating embedding and LLM providers:

```python
# Factory method in EmbeddingProvider
@staticmethod
def get_provider(provider_name: str = None, **kwargs) -> "EmbeddingProvider":
    # Create appropriate provider based on name
```

### Strategy Pattern

Used for different retrieval strategies that can be swapped out:

```python
# Different strategies for retrieval
async def retrieve(self, query: str) -> Tuple[List[str], List[float]]:
    # Strategy selection based on query characteristics
    sparql_weight, semantic_weight, top_k = self._calculate_parameters(
        query_complexity, query_specificity, query_type)
```

### Adapter Pattern

Used to provide a consistent interface to different LLM and embedding providers:

```python
# Abstract interface
class LLMProvider(abc.ABC):
    @abc.abstractmethod
    async def generate_text(self, prompt: str, ...) -> str:
        pass
```

### Repository Pattern

Used for abstracting data access:

```python
# OntologyManager as a repository for ontology data
class OntologyManager:
    async def execute_sparql(self, sparql_query: str) -> List[str]:
        # Data access code
```

## Performance Considerations

### Caching

Caching is used throughout the system to improve performance:
- Ontology materialization is cached
- SPARQL queries can be cached
- Embedding computations should be cached

### Asynchronous Processing

The entire system is built around asynchronous processing:
- All heavy I/O operations are async
- External API calls use asyncio
- Parallel processing where possible

### Resource Management

Resources are managed carefully:
- LLM and embedding providers can be configured
- Tensor operations are optimized
- Memory-intensive operations use batching

## Extension Points

Ragatanga is designed to be extensible in several ways:

1. **New Embedding Providers**:
   - Implement the `EmbeddingProvider` interface
   - Add to the factory method

2. **New LLM Providers**:
   - Implement the `LLMProvider` interface
   - Add to the factory method

3. **Enhanced Retrieval Strategies**:
   - Extend or replace `AdaptiveRetriever`
   - Implement custom retrieval logic

4. **New Answer Generation Methods**:
   - Extend the `generate_structured_answer` function
   - Add new query type handlers

## Dependency Graph

```
ragatanga.main
├── ragatanga.config
├── ragatanga.api.app
│   ├── ragatanga.core.ontology.OntologyManager
│   ├── ragatanga.core.retrieval.AdaptiveRetriever
│   ├── ragatanga.core.semantic.SemanticSearch
│   └── ragatanga.api.routes
│       ├── ragatanga.api.models
│       ├── ragatanga.core.query.generate_structured_answer
│       └── ragatanga.core.ontology.OntologyManager
├── ragatanga.core.ontology
│   └── ragatanga.utils.sparql
├── ragatanga.core.semantic
│   └── ragatanga.utils.embeddings
├── ragatanga.core.retrieval
│   ├── ragatanga.core.ontology
│   ├── ragatanga.core.semantic
│   └── ragatanga.utils.sparql
└── ragatanga.core.query
    └── ragatanga.core.llm
```

## Configuration Management

Ragatanga uses a combination of:
- Environment variables
- Configuration module (`config.py`)
- Command-line arguments
- API parameters

This layered approach allows for flexibility in deployment scenarios.

## Future Architectural Considerations

1. **Distributed Processing**:
   - Potential for distributing LLM and embedding tasks
   - Integrating with distributed vector databases

2. **Plugin Architecture**:
   - More formal plugin architecture for providers
   - Dynamic loading of extensions

3. **Event-Driven Model**:
   - Moving to an event-driven model for better integration
   - Supporting webhooks and event subscriptions

4. **Multi-Tenancy**:
   - Supporting multiple separate knowledge bases
   - User-specific configurations and models