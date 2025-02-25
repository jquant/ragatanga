# API Reference

This page provides documentation for Ragatanga's API. The documentation is automatically generated from docstrings in the codebase.

## Core Module

### OntologyManager

```python
# Example usage
from ragatanga.core.ontology import OntologyManager

ontology_manager = OntologyManager("path/to/ontology.ttl")
await ontology_manager.load_and_materialize()
```

### AdaptiveRetriever

```python
# Example usage
from ragatanga.core.retrieval import AdaptiveRetriever

retriever = AdaptiveRetriever(ontology_manager)
results = await retriever.retrieve("What is Ragatanga?")
```

## Configuration Module

### RetrievalConfig

```python
# Example usage
from ragatanga.core.config import RetrievalConfig

config = RetrievalConfig(
    semantic_search_weight=0.7,
    ontology_search_weight=0.3,
    max_results=10
)
```

## Utility Module

### KnowledgeBaseBuilder

```python
# Example usage
from ragatanga.core.knowledge import KnowledgeBaseBuilder

kb_builder = KnowledgeBaseBuilder(ontology_manager)
await kb_builder.add_markdown_file("path/to/knowledge.md")
await kb_builder.save("knowledge_base.md")
```

!!! note "Note on API Documentation"
    As the codebase evolves, make sure to update the documentation by adding proper docstrings to your code. Uncomment the API reference sections above when the corresponding modules are available. 