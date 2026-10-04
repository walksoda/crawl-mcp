"""
Search Processing Module - Backward compatibility facade.
Implementation moved to crawl4ai_mcp.processors.web_search
"""

from .processors.web_search import *  # noqa: F401, F403
from .processors.web_search import GoogleSearchProcessor, WebSearchProcessor  # noqa: F401
from .processors.search_engines import (  # noqa: F401
    BraveSearchClient,
    CustomSearchAPIClient,
    ExaSearchClient,
    RateLimiter,
    SearchRequest,
)
