"""
Core business logic for Crawl4AI MCP Server.

MCP-free modules containing crawling, extraction, search, YouTube, and file processing logic.
"""

from .crawler_summarizer import summarize_web_content, MAX_RESPONSE_CHARS
from .crawler_core import _internal_crawl_url
from .crawl_url import crawl_url, deep_crawl_site
from .extraction_intelligent import _internal_intelligent_extract, intelligent_extract
from .extraction_entity import (
    _regex_worker,
    _safe_regex_findall,
    _internal_extract_entities,
    _internal_llm_extract_entities,
    extract_entities,
)
from .extraction_structured import _internal_extract_structured_data, extract_structured_data


def __getattr__(name):
    """Lazy imports for modules that may not be loaded yet."""
    _lazy_imports = {
        'crawl_url_with_fallback': '.crawler_fallback',
    }
    if name in _lazy_imports:
        import importlib
        module = importlib.import_module(_lazy_imports[name], package=__name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
