"""MSSearch: Dual-Path Multi-Source Visual Search Engine."""

__version__ = "1.0.0"

from .config import MSSearchConfig

try:
    from .MSSearch import MultiSourceSearchEngine, get_mssearch_response
    __all__ = ["MSSearchConfig", "MultiSourceSearchEngine", "get_mssearch_response"]
except ImportError:
    __all__ = ["MSSearchConfig"]

