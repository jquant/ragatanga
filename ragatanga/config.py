import os
from pathlib import Path
from ragatanga._version import __version__ as VERSION

# Get the base directory of the project
BASE_DIR = Path(__file__).resolve().parent

# Paths for data files
DATA_DIR = os.path.join(BASE_DIR, "data")

# Default to sample files if the real files don't exist
ONTOLOGY_PATH = os.getenv("ONTOLOGY_PATH") or os.path.join(DATA_DIR, "sample_ontology.ttl")
KNOWLEDGE_BASE_PATH = os.getenv("KNOWLEDGE_BASE_PATH") or os.path.join(DATA_DIR, "sample_knowledge_base.md")

# If the sample files exist but the real files don't, use the sample files
if not os.path.exists(ONTOLOGY_PATH) or os.path.getsize(ONTOLOGY_PATH) == 0:
    sample_path = os.path.join(DATA_DIR, "sample_ontology.ttl")
    if os.path.exists(sample_path) and os.path.getsize(sample_path) > 0:
        ONTOLOGY_PATH = sample_path

if not os.path.exists(KNOWLEDGE_BASE_PATH) or os.path.getsize(KNOWLEDGE_BASE_PATH) == 0:
    sample_path = os.path.join(DATA_DIR, "sample_knowledge_base.md")
    if os.path.exists(sample_path) and os.path.getsize(sample_path) > 0:
        KNOWLEDGE_BASE_PATH = sample_path

KBASE_INDEX_PATH = os.path.join(DATA_DIR, "kbase_index.pkl")

# SPARQL configuration
SPARQL_ENDPOINT_MEMORY = "memory://"
SPARQL_ENDPOINT_FILE = f"file://{ONTOLOGY_PATH}"

# Semantic search configuration
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Question answering configuration
MAX_TOKENS = 8000
TEMPERATURE = 0.7 

# Embedding configuration
EMBED_MODEL = "text-embedding-3-large"
BATCH_SIZE = 16
DIMENSIONS = 3072

# LLM configuration
DEFAULT_LLM_MODEL = "gpt-4o"

# API configuration
DEFAULT_PORT = 8000