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
```

## Core Components

### Ontology Manager

The `OntologyManager` is responsible for:

- Loading and parsing ontology files (TTL format)
- Materializing inferences using RDFS/OWL reasoning
- Providing SPARQL query capabilities
- Managing the knowledge graph

### Adaptive Retriever

The `AdaptiveRetriever` is the central component that:

- Analyzes queries to determine the best retrieval strategy
- Combines ontology-based and semantic search results
- Ranks and scores results based on confidence
- Provides a unified interface for all retrieval operations

### Embedding Manager

The `EmbeddingManager` handles:

- Text embedding generation using various providers
- Vector storage and retrieval
- Similarity search operations
- Caching of embeddings for performance

### LLM Manager

The `LLMManager` provides:

- Integration with multiple LLM providers
- Query enhancement and reformulation
- Structured answer generation
- Confidence estimation

## Data Flow

1. **Query Analysis**: Incoming queries are analyzed to determine complexity and type
2. **Retrieval Strategy Selection**: Based on analysis, the system selects appropriate retrieval methods
3. **Parallel Execution**: Ontology and semantic search operations run in parallel
4. **Result Fusion**: Results from different sources are combined and ranked
5. **Answer Generation**: Final answers are generated with confidence scores

## Extension Points

Ragatanga is designed to be extensible in several ways:

- **Custom Embedding Providers**: Add support for new embedding models
- **Custom LLM Providers**: Integrate with additional LLM services
- **Custom Retrieval Strategies**: Implement specialized retrieval methods
- **Custom Scoring Functions**: Define domain-specific relevance scoring 