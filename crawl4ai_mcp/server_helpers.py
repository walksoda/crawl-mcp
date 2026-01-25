"""Server helper functions for Crawl4AI MCP Server.

This module contains utility functions used by the main server for:
- Module loading and initialization
- Browser setup and diagnostics
- Result processing and content handling
- Fallback logic
"""

import glob
import platform
from pathlib import Path
from typing import Optional, Tuple, List

# Global state for lazy loading
_heavy_imports_loaded = False
_browser_setup_done = False
_browser_setup_failed = False
_tools_imported = False

# Global module references
asyncio = None
json = None
AsyncWebCrawler = None
web_crawling = None
search = None
youtube = None
file_processing = None
utilities = None


def _load_heavy_imports() -> None:
    """Load heavy imports only when tools are actually used."""
    global _heavy_imports_loaded
    if _heavy_imports_loaded:
        return

    global asyncio, json, AsyncWebCrawler

    import asyncio as _asyncio
    import json as _json
    from crawl4ai import AsyncWebCrawler as _AsyncWebCrawler

    asyncio = _asyncio
    json = _json
    AsyncWebCrawler = _AsyncWebCrawler

    _heavy_imports_loaded = True


def _load_tool_modules() -> bool:
    """
    Load tool modules only when needed.

    Returns:
        True if modules were loaded successfully, False otherwise.
    """
    global _tools_imported
    if _tools_imported:
        return True

    global web_crawling, search, youtube, file_processing, utilities

    try:
        from .tools import web_crawling as _wc
        from .tools import search as _s
        from .tools import youtube as _yt
        from .tools import file_processing as _fp
        from .tools import utilities as _ut

        web_crawling = _wc
        search = _s
        youtube = _yt
        file_processing = _fp
        utilities = _ut
        _tools_imported = True
        return True
    except ImportError:
        # Fallback for absolute imports
        try:
            from crawl4ai_mcp.tools import web_crawling as _wc
            from crawl4ai_mcp.tools import search as _s
            from crawl4ai_mcp.tools import youtube as _yt
            from crawl4ai_mcp.tools import file_processing as _fp
            from crawl4ai_mcp.tools import utilities as _ut

            web_crawling = _wc
            search = _s
            youtube = _yt
            file_processing = _fp
            utilities = _ut
            _tools_imported = True
            return True
        except ImportError:
            _tools_imported = False
            return False


def _ensure_browser_setup() -> bool:
    """
    Browser setup with lazy loading.

    Returns:
        True if browser is set up, False otherwise.
    """
    global _browser_setup_done, _browser_setup_failed

    if _browser_setup_done:
        return True
    if _browser_setup_failed:
        return False

    try:
        # Quick browser cache check
        cache_pattern = str(Path.home() / ".cache" / "ms-playwright" / "chromium-*")
        if glob.glob(cache_pattern):
            _browser_setup_done = True
            return True
        else:
            _browser_setup_failed = True
            return False
    except Exception:
        _browser_setup_failed = True
        return False


def get_system_diagnostics() -> dict:
    """Get system diagnostics for troubleshooting browser and environment issues."""
    _load_heavy_imports()

    # Check browser cache
    cache_pattern = str(Path.home() / ".cache" / "ms-playwright" / "chromium-*")
    cache_dirs = glob.glob(cache_pattern)

    return {
        "status": "FastMCP 2.0 Server - Clean STDIO communication",
        "platform": platform.system(),
        "python_version": platform.python_version(),
        "fastmcp_version": "2.0.0",
        "browser_cache_found": len(cache_dirs) > 0,
        "cache_directories": cache_dirs,
        "recommendations": [
            "Install Playwright browsers: pip install playwright && playwright install webkit",
            "For UVX: uvx --with playwright playwright install webkit"
        ]
    }


def _convert_result_to_dict(result) -> dict:
    """Convert CrawlResponse or similar object to dict."""
    if hasattr(result, 'model_dump'):
        return result.model_dump()
    elif hasattr(result, 'dict'):
        return result.dict()
    return result


def _process_content_fields(
    result_dict: dict,
    include_cleaned_html: bool,
    generate_markdown: bool
) -> dict:
    """Process content fields based on flags and add warnings if needed."""
    # Preserve existing warnings from crawler
    warnings = result_dict.get("warnings", [])
    if not isinstance(warnings, list):
        warnings = [warnings] if warnings else []

    # Handle content field based on include_cleaned_html flag
    if not include_cleaned_html and 'content' in result_dict:
        # Only remove if markdown is available or generate_markdown is True
        if generate_markdown and result_dict.get("markdown", "").strip():
            del result_dict['content']
            warnings.append(
                "HTML content removed to save tokens. Use include_cleaned_html=true to include it."
            )
        elif not generate_markdown:
            # Keep content when markdown generation is disabled
            warnings.append("HTML content preserved because generate_markdown=false.")

    if warnings:
        result_dict["warnings"] = warnings

    return result_dict


def _should_trigger_fallback(
    result_dict: dict,
    generate_markdown: bool
) -> Tuple[bool, str]:
    """
    Determine if fallback should be triggered and return reason.

    Args:
        result_dict: The crawl result dictionary
        generate_markdown: Whether markdown generation was requested

    Returns:
        Tuple of (should_trigger, reason_message)
    """
    # Check explicit failure
    if not result_dict.get("success", True):
        error_msg = result_dict.get("error", "Unknown error")
        return True, f"Initial crawl failed: {error_msg}"

    # Check for meaningful content based on what was requested
    has_markdown = bool(result_dict.get("markdown", "").strip())
    has_content = bool(result_dict.get("content", "").strip())
    has_raw_content = bool(result_dict.get("raw_content", "").strip())

    # If markdown was requested but is empty, and no other content exists
    if generate_markdown and not has_markdown:
        if has_content or has_raw_content:
            # Has some content, don't fallback - might be intentional (PDF, etc.)
            return False, ""
        return True, "Empty markdown and no alternative content available"

    # If markdown was not requested, check for any content
    if not generate_markdown and not has_content and not has_raw_content:
        return True, "No content available in response"

    return False, ""


def get_tool_modules():
    """
    Get the loaded tool modules.

    Returns:
        Tuple of (web_crawling, search, youtube, file_processing, utilities)
        or (None, None, None, None, None) if not loaded.
    """
    if not _tools_imported:
        _load_tool_modules()
    return web_crawling, search, youtube, file_processing, utilities


def is_heavy_imports_loaded() -> bool:
    """Check if heavy imports are loaded."""
    return _heavy_imports_loaded


def is_tools_imported() -> bool:
    """Check if tool modules are imported."""
    return _tools_imported


def is_browser_setup_done() -> bool:
    """Check if browser setup is done."""
    return _browser_setup_done


def _apply_content_slicing(
    result_dict: dict,
    content_limit: int,
    content_offset: int
) -> dict:
    """
    Apply content slicing to retrieve partial content.

    Args:
        result_dict: The crawl result dictionary
        content_limit: Maximum characters to return (0=unlimited)
        content_offset: Start position for content (0-indexed)

    Returns:
        Result dictionary with slicing_info added

    Note:
        slicing_info reflects the state after slicing but before token limit.
        The actual returned content may be further truncated by token limits.
    """
    # Defensive: ensure non-negative values
    content_limit = max(0, content_limit)
    content_offset = max(0, content_offset)

    slicing_info = {}

    for field in ['markdown', 'content']:
        if field in result_dict and result_dict[field]:
            text = result_dict[field]
            original_length = len(text)

            # Apply slicing
            sliced = text[content_offset:]
            if content_limit > 0:
                sliced = sliced[:content_limit]
            result_dict[field] = sliced

            # Calculate effective_limit: the actual upper bound applied
            if content_limit > 0:
                effective_limit = content_limit
            else:
                effective_limit = max(0, original_length - content_offset)

            slicing_info[field] = {
                'original_length': original_length,
                'offset': content_offset,
                'limit': content_limit,  # 0 means unlimited
                'effective_limit': effective_limit,
                'returned_length': len(sliced),
                'offset_exceeded': content_offset >= original_length
            }
        else:
            # Field does not exist or is empty - still return info for consistency
            slicing_info[field] = {
                'present': False,
                'offset': content_offset,
                'limit': content_limit
            }

    result_dict['slicing_info'] = slicing_info
    return result_dict


# =============================================================================
# Search Result Cache Mechanism (LRU + TTL)
# =============================================================================

from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import time as _time

# Cache settings
_SEARCH_CACHE_MAX_SIZE = 5
_SEARCH_CACHE_TTL_SECONDS = 3600  # 1 hour


@dataclass
class _CacheEntry:
    """Cache entry with data and timestamp."""
    data: dict
    timestamp: float


# Search result cache (LRU + TTL)
_search_result_cache: OrderedDict[str, _CacheEntry] = OrderedDict()


def _get_search_cache_key(request: dict) -> str:
    """Generate cache key for search query including all relevant parameters.
    
    Normalizes defaults to match actual search behavior and avoid cache misses.
    """
    # Normalize num_results (clamp to valid range 1-100, default 10)
    num_results = request.get('num_results', 10)
    if num_results is None:
        num_results = 10
    num_results = max(1, min(100, int(num_results)))
    
    # Normalize language (default 'en')
    language = request.get('language') or 'en'
    
    # Normalize region (default 'us')
    region = request.get('region') or 'us'
    
    # Normalize recent_days (None/0/'' all mean no filter)
    recent_days = request.get('recent_days')
    recent_days_str = str(recent_days) if recent_days else ''
    
    # Normalize safe_search (default True)
    safe_search = request.get('safe_search', True)
    if safe_search is None:
        safe_search = True
    
    # Normalize search_genre (None/'' both mean no genre filter)
    search_genre = request.get('search_genre') or ''
    
    # Build cache key with all parameters that affect search results
    key_parts = [
        request.get('query', ''),
        str(num_results),
        search_genre,
        language,
        region,
        recent_days_str,
        str(safe_search),
    ]
    key_str = ":".join(key_parts)
    return hashlib.md5(key_str.encode()).hexdigest()


def _cleanup_expired_cache() -> None:
    """Remove expired cache entries."""
    now = _time.time()
    expired_keys = [
        key for key, entry in _search_result_cache.items()
        if now - entry.timestamp > _SEARCH_CACHE_TTL_SECONDS
    ]
    for key in expired_keys:
        del _search_result_cache[key]


def _get_cached_search_result(cache_key: str) -> Optional[dict]:
    """Get search result from cache (LRU update, TTL check)."""
    _cleanup_expired_cache()  # Clean up expired entries first

    if cache_key in _search_result_cache:
        entry = _search_result_cache[cache_key]
        # TTL check
        if _time.time() - entry.timestamp <= _SEARCH_CACHE_TTL_SECONDS:
            _search_result_cache.move_to_end(cache_key)
            return entry.data
        else:
            # Expired, remove it
            del _search_result_cache[cache_key]
    return None


def _cache_search_result(cache_key: str, result: dict) -> None:
    """Store search result in cache."""
    _cleanup_expired_cache()  # Clean up before storing

    if cache_key in _search_result_cache:
        _search_result_cache.move_to_end(cache_key)
        _search_result_cache[cache_key] = _CacheEntry(data=result, timestamp=_time.time())
    else:
        if len(_search_result_cache) >= _SEARCH_CACHE_MAX_SIZE:
            _search_result_cache.popitem(last=False)  # Remove oldest entry
        _search_result_cache[cache_key] = _CacheEntry(data=result, timestamp=_time.time())


def clear_search_cache() -> None:
    """Clear the search result cache. Useful for testing."""
    _search_result_cache.clear()


def _apply_search_content_slicing(
    result_dict: dict,
    content_limit: int,
    content_offset: int
) -> dict:
    """
    Apply content slicing to search results by combining snippets.

    Args:
        result_dict: The search result dictionary
        content_limit: Maximum characters to return (0=unlimited)
        content_offset: Start position for content (0-indexed)

    Returns:
        Result dictionary with content field and slicing_info added
    """
    # Defensive: ensure non-negative values
    content_limit = max(0, content_limit)
    content_offset = max(0, content_offset)

    # Null-safe handling of results
    results = result_dict.get('results') or []
    if not isinstance(results, list):
        results = []
    content_parts = []

    for i, item in enumerate(results, 1):
        # Skip non-dict items to prevent AttributeError
        if not isinstance(item, dict):
            continue
        title = item.get('title', '') or ''
        snippet = item.get('snippet', '') or ''
        url = item.get('url', '') or ''
        part = f"[{i}] {title}\n{url}\n{snippet}\n"
        content_parts.append(part)

    combined_content = "\n".join(content_parts)
    original_length = len(combined_content)

    # Apply slicing
    sliced = combined_content[content_offset:]
    if content_limit > 0:
        sliced = sliced[:content_limit]

    result_dict['content'] = sliced

    # Calculate effective_limit
    if content_limit > 0:
        effective_limit = content_limit
    else:
        effective_limit = max(0, original_length - content_offset)

    result_dict['slicing_info'] = {
        'content': {
            'original_length': original_length,
            'offset': content_offset,
            'limit': content_limit,
            'effective_limit': effective_limit,
            'returned_length': len(sliced),
            'offset_exceeded': content_offset >= original_length,
            'source': 'combined_search_snippets',
            'result_count': len(results)
        }
    }

    return result_dict
