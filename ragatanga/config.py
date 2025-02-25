import os
from pathlib import Path
from ragatanga._version import version as VERSION

# Get the base directory of the project
BASE_DIR = Path(__file__).resolve().parent

# Paths for data files
DATA_DIR = os.path.join(BASE_DIR, "data")
ONTOLOGY_PATH = os.path.join(DATA_DIR, "ontology.ttl")
KNOWLEDGE_BASE_PATH = os.path.join(DATA_DIR, "knowledge_base.txt")
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