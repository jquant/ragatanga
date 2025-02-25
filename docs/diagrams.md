# Diagrams

This page demonstrates how to use diagrams to better explain Ragatanga's architecture and workflow.

## System Architecture

The following diagram illustrates the high-level architecture of Ragatanga:

```mermaid
graph TD
    User[User/Client] -->|Query| API[API Layer]
    API -->|Process Query| Core[Core Components]
    
    subgraph Core Components
        Retriever[Adaptive Retriever] -->|Query| Ontology[Ontology Manager]
        Retriever -->|Search| Embeddings[Embedding Manager]
        Retriever -->|Generate Answer| LLM[LLM Manager]
    end
    
    Ontology -->|Load| OntologyFile[(Ontology File)]
    Embeddings -->|Access| KnowledgeBase[(Knowledge Base)]
    
    Core -->|Results| API
    API -->|Response| User
```

## Data Flow

The following diagram illustrates the data flow during a query:

```mermaid
sequenceDiagram
    participant User
    participant API
    participant Retriever
    participant Ontology
    participant Embeddings
    participant LLM
    
    User->>API: Submit Query
    API->>Retriever: Process Query
    
    par Parallel Processing
        Retriever->>Ontology: SPARQL Query
        Ontology-->>Retriever: Ontology Results
        
        Retriever->>Embeddings: Semantic Search
        Embeddings-->>Retriever: Semantic Results
    end
    
    Retriever->>Retriever: Combine Results
    Retriever->>LLM: Generate Structured Answer
    LLM-->>Retriever: Structured Answer
    
    Retriever-->>API: Complete Results
    API-->>User: Response
```

## Configuration Components

The following diagram shows the configuration components:

```mermaid
classDiagram
    class RetrievalConfig {
        +float semantic_search_weight
        +float ontology_search_weight
        +int max_results
        +float confidence_threshold
        +apply_defaults()
    }
    
    class OntologyConfig {
        +string reasoning_level
        +bool cache_materialized
        +dict prefixes
        +validate()
    }
    
    class EmbeddingConfig {
        +string provider
        +string model_name
        +int batch_size
        +bool cache_embeddings
        +get_provider()
    }
    
    class LLMConfig {
        +string provider
        +string model_name
        +float temperature
        +int max_tokens
        +string system_prompt
        +get_provider()
    }
    
    RetrievalConfig --> OntologyConfig
    RetrievalConfig --> EmbeddingConfig
    RetrievalConfig --> LLMConfig
```

## Adding Your Own Diagrams

You can add your own diagrams to the documentation using Mermaid.js. Here's how:

1. Add the Mermaid extension to your `mkdocs.yml`:

```yaml
markdown_extensions:
  - pymdownx.superfences:
      custom_fences:
        - name: mermaid
          class: mermaid
          format: !!python/name:pymdownx.superfences.fence_code_format
```

2. Create your diagram using Mermaid syntax:

```
```mermaid
graph TD
    A[Start] --> B[Process]
    B --> C[End]
```
```

3. You can create various types of diagrams:
   - Flowcharts (`graph TD` or `graph LR`)
   - Sequence diagrams (`sequenceDiagram`)
   - Class diagrams (`classDiagram`)
   - Entity Relationship diagrams (`erDiagram`)
   - Gantt charts (`gantt`)
   - State diagrams (`stateDiagram-v2`) 