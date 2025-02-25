# Configuration Guide

This document provides detailed information about configuring Ragatanga for your specific needs.

## Configuration Methods

Ragatanga can be configured in several ways:

1. **Environment Variables**: Set configuration through environment variables
2. **Configuration Files**: Use `.env` files for persistent configuration
3. **Programmatic Configuration**: Configure components directly in code
4. **Configuration Objects**: Use configuration classes for fine-grained control

## Environment Variables

### Core Configuration

| Variable | Description | Default | Example |
|----------|-------------|---------|---------|
| `RAGATANGA_LOG_LEVEL` | Logging level | `INFO` | `DEBUG` |
| `RAGATANGA_CACHE_DIR` | Directory for caching | `~/.ragatanga/cache` | `/tmp/ragatanga_cache` |
| `RAGATANGA_MAX_RESULTS` | Maximum results to return | `5` | `10` |
| `RAGATANGA_CONFIDENCE_THRESHOLD` | Minimum confidence score | `0.6` | `0.8` |

### Embedding Configuration

| Variable | Description | Default | Example |
|----------|-------------|---------|---------|
| `RAGATANGA_EMBEDDING_PROVIDER` | Embedding provider | `openai` | `sentence_transformers` |
| `RAGATANGA_EMBEDDING_MODEL` | Embedding model name | `text-embedding-ada-002` | `all-MiniLM-L6-v2` |
| `RAGATANGA_EMBEDDING_BATCH_SIZE` | Batch size for embeddings | `32` | `64` |
| `RAGATANGA_EMBEDDING_CACHE` | Enable embedding cache | `true` | `false` |

### LLM Configuration

| Variable | Description | Default | Example |
|----------|-------------|---------|---------|
| `RAGATANGA_LLM_PROVIDER` | LLM provider | `openai` | `anthropic` |
| `RAGATANGA_LLM_MODEL` | LLM model name | `gpt-3.5-turbo` | `claude-2` |
| `RAGATANGA_LLM_TEMPERATURE` | LLM temperature | `0.0` | `0.7` |
| `RAGATANGA_LLM_MAX_TOKENS` | Maximum tokens for LLM | `1024` | `2048` |

### API Keys

| Variable | Description | Example |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key | `sk-...` |
| `ANTHROPIC_API_KEY` | Anthropic API key | `sk-ant-...` |
| `HUGGINGFACE_API_KEY` | HuggingFace API key | `hf_...` |

## Configuration File

You can create a `.env` file in your project root with the following format:

```
# Core Configuration
RAGATANGA_LOG_LEVEL=INFO
RAGATANGA_CACHE_DIR=~/.ragatanga/cache
RAGATANGA_MAX_RESULTS=5
RAGATANGA_CONFIDENCE_THRESHOLD=0.6

# Embedding Configuration
RAGATANGA_EMBEDDING_PROVIDER=openai
RAGATANGA_EMBEDDING_MODEL=text-embedding-ada-002
RAGATANGA_EMBEDDING_BATCH_SIZE=32
RAGATANGA_EMBEDDING_CACHE=true

# LLM Configuration
RAGATANGA_LLM_PROVIDER=openai
RAGATANGA_LLM_MODEL=gpt-3.5-turbo
RAGATANGA_LLM_TEMPERATURE=0.0
RAGATANGA_LLM_MAX_TOKENS=1024

# API Keys
OPENAI_API_KEY=your-openai-api-key
```

## Programmatic Configuration

### Configuring the Ontology Manager

```python
from ragatanga.core.ontology import OntologyManager
from ragatanga.core.config import OntologyConfig

# Create configuration
ontology_config = OntologyConfig(
    reasoning_level="RDFS",
    cache_materialized=True,
    prefixes={
        "ex": "http://example.org/",
        "schema": "http://schema.org/"
    }
)

# Initialize with configuration
ontology_manager = OntologyManager(
    "path/to/ontology.ttl",
    config=ontology_config
)
```

### Configuring the Embedding Manager

```python
from ragatanga.core.embeddings import EmbeddingManager
from ragatanga.core.config import EmbeddingConfig

# Create configuration
embedding_config = EmbeddingConfig(
    provider="sentence_transformers",
    model_name="all-MiniLM-L6-v2",
    batch_size=64,
    cache_embeddings=True
)

# Initialize with configuration
embedding_manager = EmbeddingManager(config=embedding_config)
```

### Configuring the LLM Manager

```python
from ragatanga.core.llm import LLMManager
from ragatanga.core.config import LLMConfig

# Create configuration
llm_config = LLMConfig(
    provider="openai",
    model_name="gpt-4",
    temperature=0.2,
    max_tokens=2048,
    system_prompt="You are a helpful assistant specialized in knowledge retrieval."
)

# Initialize with configuration
llm_manager = LLMManager(config=llm_config)
```

### Configuring the Retriever

```python
from ragatanga.core.retrieval import AdaptiveRetriever
from ragatanga.core.config import RetrievalConfig

# Create configuration
retrieval_config = RetrievalConfig(
    semantic_search_weight=0.7,
    ontology_search_weight=0.3,
    max_results=10,
    confidence_threshold=0.6,
    use_query_enhancement=True,
    use_multilingual=True
)

# Initialize with configuration
retriever = AdaptiveRetriever(
    ontology_manager,
    embedding_manager=embedding_manager,
    llm_manager=llm_manager,
    config=retrieval_config
)
```

## Configuration Classes

Ragatanga provides several configuration classes for fine-grained control:

- `OntologyConfig`: Configuration for the ontology manager
- `EmbeddingConfig`: Configuration for embedding generation and storage
- `LLMConfig`: Configuration for language model interactions
- `RetrievalConfig`: Configuration for retrieval operations
- `APIConfig`: Configuration for the API server

See the [API Reference](api-reference.md) for detailed documentation of these classes. 