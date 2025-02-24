import os
from pathlib import Path

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