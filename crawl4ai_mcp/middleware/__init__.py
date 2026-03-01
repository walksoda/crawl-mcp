"""Middleware package for Crawl4AI MCP Server.

Provides composable middleware pipeline for cross-cutting concerns:
validation, content slicing, token limiting, caching, etc.
"""

from .pipeline import PipelineContext, Middleware, ToolPipeline
from .module_check import ModuleCheckMiddleware
from .validation import ValidationMiddleware
from .content_slicing import ContentSlicingMiddleware
from .search_slicing import SearchSlicingMiddleware
from .token_limiter import TokenLimitMiddleware
from .response_transform import ResponseTransformMiddleware
from .cache import SearchCacheMiddleware, clear_search_cache
from .error_handler import ErrorHandlingMiddleware

__all__ = [
    "PipelineContext",
    "Middleware",
    "ToolPipeline",
    "ModuleCheckMiddleware",
    "ValidationMiddleware",
    "ContentSlicingMiddleware",
    "SearchSlicingMiddleware",
    "TokenLimitMiddleware",
    "ResponseTransformMiddleware",
    "SearchCacheMiddleware",
    "ErrorHandlingMiddleware",
    "clear_search_cache",
]
