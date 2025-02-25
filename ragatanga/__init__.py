"""
Ragatanga - A hybrid semantic knowledge base and query system.
"""

try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"

# Don't import app here since it requires environment variables
from ragatanga._version import version as __version__

# Make core components available at package level
from ragatanga.core.ontology import OntologyManager
from ragatanga.core.retrieval import AdaptiveRetriever
from ragatanga.core.semantic import SemanticSearch
from ragatanga.core.query import generate_structured_answer

__all__ = [
    "__version__",
    "OntologyManager",
    "AdaptiveRetriever",
    "SemanticSearch",
    "generate_structured_answer"
]