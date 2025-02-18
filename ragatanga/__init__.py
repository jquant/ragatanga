"""
Ragatanga - A hybrid semantic knowledge base and query system.
"""

try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"

# Don't import app here since it requires environment variables
__all__ = ["__version__"] 