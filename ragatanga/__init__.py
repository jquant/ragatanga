"""
Ragatanga - A hybrid semantic knowledge base and query system.
"""

try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"

from .main import app, handle_query

__all__ = ["app", "handle_query"] 