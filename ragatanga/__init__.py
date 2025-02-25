"""
Ragatanga - A hybrid semantic knowledge base and query system.
"""

__version__ = "0.3.0.dev1"

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