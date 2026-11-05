"""MSSearch: Dual-Path Multi-Source Visual Search Engine."""

__version__ = "1.0.0"

import warnings
import logging

# Suppress harmless upstream library deprecation and checkpoint configuration mismatch warnings
warnings.filterwarnings("ignore", category=FutureWarning, message=r".*min_pixels.*max_pixels.*")
warnings.filterwarnings("ignore", message=r".*bos_token_id.*")
warnings.filterwarnings("ignore", message=r".*eos_token_id.*")
warnings.filterwarnings("ignore", message=r".*`torch_dtype` is deprecated.*")
warnings.filterwarnings("ignore", category=UserWarning, message=r".*the number of connected components of the connectivity matrix.*")
logging.getLogger("transformers.configuration_utils").setLevel(logging.ERROR)

from .config import MSSearchConfig


def __getattr__(name):
    if name in ("MultiSourceSearchEngine", "get_mssearch_response"):
        from .MSSearch import MultiSourceSearchEngine, get_mssearch_response
        return locals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["MSSearchConfig", "MultiSourceSearchEngine", "get_mssearch_response"]

